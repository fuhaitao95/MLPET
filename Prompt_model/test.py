import argparse
from collections import OrderedDict

import torch

from Prompt_model.Prompt import Prompt_SMILES
from Prompt_model.converter import PromptConverter
from Prompt_model.token_SMILES import Alphabet


def main():
    parser = argparse.ArgumentParser(description='(GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)

    parser.add_argument('--max_position_num', type=int, default=29, help="QM9 max position 之后根据下游数据集一起计算")
    parser.add_argument('--layer_num', type=int, default=5, help="33 layers in paper")
    parser.add_argument('--attention_head_num', type=int, default=5, help="20 heads in paper")
    parser.add_argument('--embed_dim', type=int, default=300, help="1280 dims in paper")
    parser.add_argument('--ffn_embed_dim', type=int, default=300)
    parser.add_argument("--emb_layer_norm_before", action="store_true", default=False)
    parser.add_argument("--token_dropout", action="store_true", default=True)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.005, help="1 × 10−4 in papers")

    parser.add_argument('--epochs', type=int, default=1, help="270k steps of updates in papers")
    parser.add_argument('--eval_steps', type=int, default=0)
    parser.add_argument('--runs', type=int, default=1)

    # save model
    parser.add_argument('--model_save_path', type=str, default='save/model.pth',
                        help='the directory used to save models')
    parser.add_argument('--model_encoder_load_path', type=str, default='',
                        help='the path of trained encoder')

    # load trained model for test
    parser.add_argument('--model_load_path', type=str, default='',
                        help='the path of trained model')
    parser.add_argument('--model_direct_load_path', type=str, default='',
                        help='the path of trained model')
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dictionary = Alphabet.build_alphabet()

    model = Prompt_SMILES(args, dictionary).to(device)

    converter = PromptConverter(dictionary)

    data = [
        "C#CC#CC(=O)C#N",
        "C#Cc1ccc([nH]1)N",
        "C[NH2+]CC([O-])=O"
    ]

    prompts = ['<seq>']

    encoded_sequences = converter(data, prompt_toks=prompts)

    with torch.no_grad():
        logits = model(encoded_sequences, with_prompt_num=1)['logits']
    print(logits)


if __name__ == '__main__':
    main()
