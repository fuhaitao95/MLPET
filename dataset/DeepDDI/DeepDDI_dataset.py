import os
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, SDMolSupplier
import torch
from torch_geometric.data import Data, InMemoryDataset
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from Utils.featureUtil import smiles_to_indices, sdf_to_smiles, mol_to_pyg


class DrugDDIDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DrugDDIDataset, self).__init__(root, transform, pre_transform)

        self.task_type = "classification"

        self.num_tasks = 86
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # List all files that will be processed
        return ['KnownDDI.csv', 'Interaction_information.csv', 'Approved_drug_Information.txt']

    @property
    def processed_file_names(self):
        # Output file after processing
        return ['processed_data.pt']

    def process(self):
        RDLogger.DisableLog('rdApp.warning')

        # Load DDI information (edges)
        known_ddi = pd.read_csv(os.path.join(self.raw_dir, 'KnownDDI.csv'))

        # 创建一个药物ID映射表，确保每个药物有一个唯一的数值索引
        drug_to_index = {drug_id: idx for idx, drug_id in
                         enumerate(known_ddi['Drug1'].unique().tolist() + known_ddi['Drug2'].unique().tolist())}

        processed_drugs = set()
        data_list = []

        # Iterate over the drug pairs in KnownDDI.csv
        for idx, row in tqdm(known_ddi.iterrows(), total=len(known_ddi), desc="Processing DDI pairs"):
            drug1, drug2, label = row['Drug1'], row['Drug2'], row['Label']

            # Construct SDF file paths
            sdf_file1 = os.path.join(self.raw_dir, f'DrugBank5.0_Approved_drugs/{drug1}.sdf')
            sdf_file2 = os.path.join(self.raw_dir, f'DrugBank5.0_Approved_drugs/{drug2}.sdf')

            # Try to load the SDF files for both drugs
            drug1_smiles, drug1_mol = sdf_to_smiles(sdf_file1)
            drug2_smiles, drug2_mol = sdf_to_smiles(sdf_file2)

            # Initialize variables for this pair's data
            # drug1_smiles_indices = smiles_to_indices(drug1_smiles, self.smiles_vocab, self.max_len)
            # drug1_pyg_data = mol_to_pyg(drug1_mol)
            # drug2_smiles_indices = smiles_to_indices(drug2_smiles, self.smiles_vocab, self.max_len)
            # drug2_pyg_data = mol_to_pyg(drug2_mol)

            # Drug pair edge index
            drug1_idx = drug_to_index[drug1]
            drug2_idx = drug_to_index[drug2]
            edge_index = torch.tensor([[drug1_idx], [drug2_idx]], dtype=torch.long)

            # Create the Data object for this drug pair
            data = Data(
                drug1_smiles=drug1_smiles,  # Drug1 SMILES
                drug2_smiles=drug2_smiles,  # Drug2 SMILES
                edge_index=edge_index,              # Edge between drug1 and drug2
                label=torch.tensor([label-1], dtype=torch.long)  # Label (DDI interaction type)
            )

            data_list.append(data)

        # 保存预处理后的数据
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def extract_fingerprint(self, mol):
        """ Extract molecular fingerprint as feature from RDKit molecule object. """
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        return torch.tensor(list(fingerprint), dtype=torch.float)

    def combine_node_features(self, mol, drug_info_row):
        """
        Combine molecular structure features and drug information features.
        If features are missing, use placeholder values (e.g., zeros).
        """

        # Extract molecular fingerprint (or other structural features)
        if mol is not None:
            structural_features = self.extract_fingerprint(mol)  # Example: a 1024-dim fingerprint vector
        else:
            # print("Molecule is None, using zero vector for structural features.")
            structural_features = torch.zeros(1024)  # Assuming fingerprint length is 1024

        # Extract drug information features (e.g., from drug_info_row)
        if drug_info_row is not None:
            text_features = torch.tensor([
                len(drug_info_row['Name']),  # Name length as a simple feature
                len(drug_info_row['Category']),  # Category length
            ], dtype=torch.float)
        else:
            # print("Drug info is missing, using zero vector for text features.")
            text_features = torch.zeros(2)  # Assuming 4 text-based features

            # Combine structural and text features
        combined_features = torch.cat([structural_features, text_features])

        return combined_features

    def combine_edge_features(self, interaction_data):
        """
        Combine different edge features like DDI type, description, and subject.
        If features are missing, use placeholder values (e.g., zeros).
        """

        # Example: Use one-hot encoding for DDI type (assuming categorical field)
        ddi_type = interaction_data.get('DDI type', None)
        description = interaction_data.get('Description', "")
        subject = interaction_data.get('Subject', None)

        # DDI type one-hot encoding
        ddi_type_mapping = {
            'inhibition': [1, 0, 0],
            'synergy': [0, 1, 0],
            'antagonism': [0, 0, 1]
        }
        ddi_type_vector = ddi_type_mapping.get(ddi_type, [0, 0, 0])

        # Subject one-hot encoding
        subject_mapping = {
            'metabolism': [1, 0],
            'receptor': [0, 1]
        }
        subject_vector = subject_mapping.get(subject, [0, 0])

        # Convert description to numerical feature (e.g., description length)
        description_length = len(description)

        # Combine all features into a single vector
        combined_features = torch.tensor(ddi_type_vector + subject_vector + [description_length], dtype=torch.float)

        return combined_features

# Usage:
# dataset = DrugDDIDataset(root='../../dataset/DeepDDI')
# data = dataset[0]
# print(data)
