import io
import sqlite3
from functools import partial

import numpy as np
import pandas as pd
import torch
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from torch.utils.data import Dataset

from utils import integer_label_protein

    
class DTIDataset(Dataset):
    """Dataset class for Drug-Target Interaction (DTI) prediction."""
    
    def __init__(
        self, 
        list_IDs, 
        df,
        protein_language_model_embedding_dim,
        smiles_db_path,
        protein_db_path,
        is_LigandGCN=False, 
        is_unimol_Ligand=False, 
        is_ProteinCNN=False, 
        is_esm_Protein=False,
        max_drug_nodes=300,
        protein_padding_length=1200,
        drug_embedding_dim=512
    ):

        """
        Initialize the DTI dataset.
        
        Parameters:
        -----------
        list_IDs : list
            List of indices for data samples
        df : pandas.DataFrame
            DataFrame containing drug-target data
        is_LigandGCN : bool, optional
            Whether to use GCN for ligand features
        is_unimol_Ligand : bool, optional
            Whether to use unimol for ligand features
        is_ProteinCNN : bool, optional
            Whether to use CNN for protein features
        is_esm_Protein : bool, optional
            Whether to use ESM for protein features
        protein_language_model_embedding_dim : int, optional
            Dimension of protein language model embeddings
        max_drug_nodes : int, optional
            Maximum number of nodes for drug graphs
        smiles_db_path : str, optional
            Path to the SQLite database containing SMILES embeddings
        protein_db_path : str, optional
            Path to the SQLite database containing protein embeddings
        protein_padding_length : int, optional
            Length to pad protein sequences to
        drug_embedding_dim : int, optional
            Dimension of drug embeddings
        """
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes
        self.is_LigandGCN = is_LigandGCN
        self.is_ProteinCNN = is_ProteinCNN
        self.is_unimol_Ligand = is_unimol_Ligand
        self.is_esm_Protein = is_esm_Protein
        self.protein_embedding_dim = protein_language_model_embedding_dim
        self.protein_padding_length = protein_padding_length
        self.drug_embedding_dim = drug_embedding_dim

        # Initialize featurizers for ligand GCN
        if is_LigandGCN:
            self.atom_featurizer = CanonicalAtomFeaturizer()
            self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
            self.fc = partial(smiles_to_bigraph, add_self_loop=True)

        # Initialize SQLite connections
        self.conn_sqlite_smiles = sqlite3.connect(smiles_db_path)
        self.conn_sqlite_protein = sqlite3.connect(protein_db_path)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Parameters:
        -----------
        index : int
            Index of the sample
            
        Returns:
        --------
        tuple
            (drug_graph, drug_embedding, protein_cnn, protein_esm, label)
        """
        # try:
            # Get actual index from list_IDs
        actual_index = self.list_IDs[index]
        sample = self.df.iloc[actual_index]
        smiles = sample['SMILES']
        protein_sequence = sample['Protein']
        y = sample["Y"]
        
        # Process drug features
        v_d = self._process_ligand_gcn(smiles) if self.is_LigandGCN else None
        v_d_atomic_embedding = self._process_unimol_ligand(smiles) if self.is_unimol_Ligand else None
        
        # Process protein features
        v_p_cnn = self._process_protein_cnn(protein_sequence) if self.is_ProteinCNN else None
        v_p_esm = self._process_esm_protein(protein_sequence) if self.is_esm_Protein else None

        return v_d, v_d_atomic_embedding, v_p_cnn, v_p_esm, y

    # except Exception as e:
    #     print(f"Error processing sample at index {index}: {e}")
    #     # Skip problematic sample and move to the next one
        # return self.__getitem__(index + 1)

    def _process_ligand_gcn(self, smiles):
        """Process SMILES string into a graph for GCN."""
        v_d = self.fc(smiles=smiles, 
                      node_featurizer=self.atom_featurizer, 
                      edge_featurizer=self.bond_featurizer)
        
        # Extract node features
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        
        # Add padding bit to distinguish real vs. virtual nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        
        # Add virtual nodes with padding bit set to 1
        virtual_node_feat = torch.cat((
            torch.zeros(num_virtual_nodes, 74), 
            torch.ones(num_virtual_nodes, 1)
        ), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        
        # Add self loops
        v_d = v_d.add_self_loop()
        return v_d

    def _process_unimol_ligand(self, smiles):
        """Get UniMol embedding for a SMILES string and pad if necessary."""
        v_d_atomic_embedding = self.get_smiles_embedding(smiles)
        v_d_atomic_embedding = torch.tensor(v_d_atomic_embedding)
        
        # Pad if necessary
        if v_d_atomic_embedding.shape[0] < self.max_drug_nodes:
            padding = torch.zeros((
                self.max_drug_nodes - v_d_atomic_embedding.shape[0], 
                self.drug_embedding_dim
            ))
            v_d_atomic_embedding = torch.cat([v_d_atomic_embedding, padding])
            
        return v_d_atomic_embedding

    def _process_protein_cnn(self, sequence):
        """Convert protein sequence to integer encoding for CNN."""
        v_p_cnn = integer_label_protein(sequence)
        return torch.tensor(v_p_cnn)

    def _process_esm_protein(self, sequence):
        """Get ESM embedding for a protein sequence and pad if necessary."""
        v_p_esm_embedding = self.get_protein_embedding(sequence)
        v_p_esm = torch.tensor(v_p_esm_embedding)
        
        # Pad if necessary
        if v_p_esm.shape[0] < self.protein_padding_length:
            padding = torch.zeros((
                self.protein_padding_length - v_p_esm.shape[0], 
                self.protein_embedding_dim
            ))
            v_p_esm = torch.cat([v_p_esm, padding])
            
        return v_p_esm

    def get_protein_embedding(self, protein_sequence):
        """
        Retrieve protein embedding from SQLite database.
        
        Parameters:
        -----------
        protein_sequence : str
            Protein sequence to get embedding for
            
        Returns:
        --------
        numpy.ndarray
            Protein embedding
        """
        cursor = self.conn_sqlite_protein.cursor()
        cursor.execute(
            'SELECT embedding FROM protein_embeddings WHERE protein_sequence=?', 
            (protein_sequence,)
        )
        result = cursor.fetchone()

        if result is None:
            raise ValueError(f"Protein sequence {protein_sequence} not found in database")

        buffer = io.BytesIO(result[0])
        embedding = np.load(buffer, allow_pickle=True)
        embedding = np.array(embedding, dtype=np.float32).reshape((-1, self.protein_embedding_dim))

        return embedding

    def get_smiles_embedding(self, smiles):
        """
        Retrieve SMILES embedding from SQLite database.
        
        Parameters:
        -----------
        smiles : str
            SMILES string to get embedding for
            
        Returns:
        --------
        numpy.ndarray
            SMILES embedding
        """
        cursor = self.conn_sqlite_smiles.cursor()
        cursor.execute(
            'SELECT atomic_embedding FROM molecule_atomic_embeddings WHERE smiles=?', 
            (smiles,)
        )
        result = cursor.fetchone()

        if result is None:
            raise ValueError(f"SMILES {smiles} not found in database")

        atomic_embedding = np.load(io.BytesIO(result[0]), allow_pickle=True)
        return atomic_embedding
        
    def __del__(self):
        """Clean up database connections when the object is deleted."""
        if hasattr(self, 'conn_sqlite_smiles'):
            self.conn_sqlite_smiles.close()
        if hasattr(self, 'conn_sqlite_protein'):
            self.conn_sqlite_protein.close()

