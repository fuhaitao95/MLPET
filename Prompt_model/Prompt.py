import torch
from torch import nn
from torch.nn import LayerNorm as ESM1bLayerNorm

from .modules import LearnedPositionalEmbedding, TransformerLayer


class LearnablePrompt(nn.Module):
    def __init__(self, args, num_prompts):
        """
        初始化可学习的提示。

        Args:
            num_prompts (int): 提示的数量。
            embed_dim (int): 嵌入的维度。
        """
        super(LearnablePrompt, self).__init__()
        self.num_prompts = num_prompts
        self.embed_dim = args.embed_dim

        # 使用 register_parameter 动态注册参数
        for i in range(num_prompts):
            self.register_parameter(f"{i}", nn.Parameter(torch.randn(1, self.embed_dim)))

    def forward(self, task_id):
        """
        根据任务 ID 返回对应的 prompt。

        Args:
            task_id (int): 当前任务的 ID。

        Returns:
            torch.Tensor: 指定任务的 prompt。
        """
        if not (0 <= task_id < self.num_prompts):
            raise ValueError(f"task_id 应在 [0, {self.num_prompts - 1}] 范围内，当前为 {task_id}")

        # 动态获取注册的参数
        return getattr(self, f"{task_id}")


# Prompt-tuning module (for combining prompts)
class PromptTuningModule(nn.Module):
    def __init__(self, embed_size, num_prompt):
        super(PromptTuningModule, self).__init__()
        self.linear = nn.Linear(num_prompt * embed_size, embed_size)  # Combine SMILES and edge prompts

    def forward(self, prompts):
        # Concatenate and combine the SMILES and edge prompts
        combined_prompt = torch.cat([param for param in prompts], dim=-1)
        return self.linear(combined_prompt)  # Output combined prompt representation


class Prompt_SMILES(nn.Module):
    def __init__(self, args, alphabet) -> None:
        super().__init__()
        self.args = args
        self.max_position_num = args.max_position_num
        self.layer_num = args.layer_num
        self.attention_head_num = args.attention_head_num
        self.dropout = args.transformer_dropout
        self.embed_dim = args.embed_dim
        self.ffn_embed_dim = args.ffn_embed_dim
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.mask2_idx = alphabet.mask2_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.emb_layer_norm_before = getattr(self.args, "emb_layer_norm_before", False)

        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.embed_dim, padding_idx=self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    self.ffn_embed_dim,
                    self.attention_head_num,
                    self.dropout,
                    add_bias_kv=False,
                )
                for _ in range(self.layer_num)
            ]
        )
        self.embed_scale = 1
        self.embed_positions = LearnedPositionalEmbedding(
            self.max_position_num, self.embed_dim, self.padding_idx
        )
        self.emb_layer_norm_before = (
            ESM1bLayerNorm(self.embed_dim) if self.emb_layer_norm_before else None
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

    def forward(self, tokens, repr_layers=[], need_head_weights=False, with_prompt_num=0,
                learnable_prompt=None):
        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T
        #  字符填空任务的 mask 需要对key_attention_mask进行处理，使它可以查询别人，别人不能查询它
        mlm_mask = tokens.eq(self.mask2_idx)
        if with_prompt_num != 0 or learnable_prompt != None:
            x = self.embed_scale * self.embed_tokens(tokens)
            if with_prompt_num != 0:
                x[:, :-with_prompt_num, :] = x[:, :-with_prompt_num, :] + self.embed_positions(
                    tokens[:, :-with_prompt_num])
            else:
                x = x + self.embed_positions(tokens)
            if learnable_prompt is not None:
                learned_prompt_num = learnable_prompt.size(0)
                learnable_prompt = learnable_prompt.repeat(x.size(0), 1, 1)
                x = torch.cat([x, learnable_prompt], 1)
                padding_mask = torch.cat([padding_mask,
                                          torch.zeros((x.size(0), learned_prompt_num), dtype=padding_mask.dtype,
                                                      device=padding_mask.device)], dim=1)
                mlm_mask = torch.cat([mlm_mask,
                                          torch.zeros((x.size(0), learned_prompt_num), dtype=mlm_mask.dtype,
                                                      device=mlm_mask.device)], dim=1)
                with_prompt_num += learned_prompt_num
        else:
            x = self.embed_scale * self.embed_tokens(tokens)
            if getattr(self.args, 'token_dropout', True):
                x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
                # x: B x T x C
                mask_ratio_train = 0.15 * 0.8
                src_lengths = (~padding_mask).sum(-1)
                mask_ratio_observed = (tokens == self.mask_idx).sum(-1).float() / src_lengths
                x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
                x = x.to(next(self.embed_tokens.parameters()).dtype)
            x = x + self.embed_positions(tokens)
        attention_mask = padding_mask.repeat_interleave(padding_mask.size(1) * self.attention_head_num, dim=0).reshape(
            -1,
            padding_mask.size(1),
            padding_mask.size(1))
        # attention_mask中 padding_mask和mlm_mask区别对待，mlm_mask可以从别人那获取特征，但是不能散播特征
        # 正好与prompt相反
        if with_prompt_num != 0:
            attention_mask[:, -with_prompt_num:, :-with_prompt_num] = True
        if self.emb_layer_norm_before:
            x = self.emb_layer_norm_before(x)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x
        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)
        self_attn_padding_mask = padding_mask | mlm_mask
        if not self_attn_padding_mask.any():
            self_attn_padding_mask = None
        if not padding_mask.any():
            padding_mask = None
        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x, self_attn_mask=attention_mask, self_attn_padding_mask=self_attn_padding_mask,
                need_head_weights=need_head_weights,
                with_prompt_num=with_prompt_num
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)
        # last hidden representation should have layer norm applied
        if (layer_idx + 2) in repr_layers:
            hidden_representations[layer_idx + 2] = x

        if with_prompt_num != 0:
            x = x[:, :-with_prompt_num, :]

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
        return result
