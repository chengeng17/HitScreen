import argparse
import os
import warnings
from time import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataloader_sql import DTIDataset
from models import HitScreen
from trainer import Trainer
from utils import set_seed, graph_collate_func, mkdir


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="HitScreen for DTI prediction")
    
    # Data parameters
    parser.add_argument('--data', 
                        default="./data/data_augmentation_chembl_33_duplicate_target_dataset", 
                        type=str, help='dataset path')
    parser.add_argument('--result_output_dir', default="./result", type=str, 
                       help="Output directory for results")
    parser.add_argument('--result_save_model', default=True, type=bool, 
                       help="Whether to save the model")
    
    # Model architecture parameters
    add_model_args(parser)
    
    # Training parameters
    parser.add_argument('--max_epoch', default=20, type=int, help="Maximum number of epochs")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size")
    parser.add_argument('--num_workers', default=0, type=int, help="Number of workers")
    parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
    parser.add_argument('--seed', default=2048, type=int, help="Random seed")
    parser.add_argument('--run_device', default='cuda:0', type=str, 
                        help="compute device (e.g., 'cpu', 'cuda:0')")
    
    return parser.parse_args()


def add_model_args(parser):
    """Add model architecture arguments."""
    # CoAttention parameters
    parser.add_argument('--layer', default=1, type=int, help="Number of layers")
    parser.add_argument('--hidden_size', default=128, type=int, help="Hidden size")
    parser.add_argument('--multi_head', default=8, type=int, help="Number of multi heads")
    parser.add_argument('--dropout', default=0.2, type=float, help="Dropout rate")

    # Drug feature extractor
    parser.add_argument('--is_LigandGCN', type=bool, default=True, 
                       help="Whether to use GCN for extracting ligand features")
    parser.add_argument('--is_unimol_Ligand', type=bool, default=True, 
                       help="Whether to use unimol for extracting ligand features")
    parser.add_argument('--ligand_node_in_feats', default=75, type=int, 
                       help="Number of input features for ligand nodes")
    parser.add_argument('--ligand_padding', default=True, type=bool, 
                       help="Whether to pad ligand features")
    parser.add_argument('--ligand_hidden_layers', default=[128, 128, 128], type=list, 
                       help="Hidden layers for ligand feature extractor")
    parser.add_argument('--gcn_ligand_node_in_embedding', default=128, type=int, 
                       help="Input embedding size for ligand nodes")
    parser.add_argument('--unimol_ligand_node_in_embedding', default=512, type=int, 
                       help="Input embedding size for ligand nodes")
    parser.add_argument('--ligand_max_nodes', default=300, type=int, 
                       help="Maximum number of nodes for ligands")
    parser.add_argument('--smiles_db_path', 
                       default='./unimol_molecule_embeddings_no_h_chembl33.db', 
                       type=str, help="Path to SQLite database with SMILES embeddings")
    parser.add_argument('--drug_embedding_dim', default=512, type=int,
                       help="Dimension of drug embeddings")

    # Protein feature extractor
    parser.add_argument('--is_ProteinCNN', type=bool, default=False, 
                       help="Whether to use CNN for extracting protein features")
    parser.add_argument('--is_esm_Protein', type=bool, default=True, 
                       help="Whether to use ESM for extracting protein features")
    parser.add_argument('--protein_num_filters', default=[128, 128, 128], type=list, 
                       help="Number of filters for protein feature extractor")
    parser.add_argument('--protein_kernel_size', default=[3, 6, 9], type=list, 
                       help="Kernel sizes for protein feature extractor")
    parser.add_argument('--protein_embedding_dim', default=128, type=int, 
                       help="Embedding dimension for protein features")
    parser.add_argument('--protein_language_model_embedding_dim', default=1536, type=int, 
                       help="Embedding dimension for protein language models: Ankh = 1536, ProtT5 = 1024, ESM-2 150M = 640")
    parser.add_argument('--protein_padding', default=True, type=bool, 
                       help="Whether to pad protein features")
    parser.add_argument('--protein_db_path', 
                       default='./ankh_protein_embedding_1536.db', 
                       type=str, help="Path to SQLite database with protein embeddings")
    parser.add_argument('--protein_padding_length', default=1200, type=int,
                       help="Length to pad protein sequences to")

    # MLP decoder
    parser.add_argument('--decoder_name', default="MLP", type=str, help="Decoder name")
    parser.add_argument('--decoder_in_dim', default=256, type=int, help="Input dimension for decoder")
    parser.add_argument('--decoder_hidden_dim', default=128, type=int, help="Hidden dimension for decoder")
    parser.add_argument('--decoder_out_dim', default=64, type=int, help="Output dimension for decoder")
    parser.add_argument('--decoder_binary', default=1, type=int, help="Binary setting for decoder")

def setup_device(device_str):
    """Set up the computing device."""
    if torch.cuda.is_available() and 'cuda' in device_str:
        return torch.device(device_str)
    return torch.device('cpu')


def load_datasets(data_path, args):
    """
    Load training and validation datasets.
    
    Parameters:
    -----------
    data_path : str
        Path to the directory containing train.csv and val.csv
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    tuple
        (train_dataset, val_dataset)
    """
    train_path = os.path.join(data_path, 'train.csv')
    val_path = os.path.join(data_path, "val.csv")
    
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    train_dataset = DTIDataset(
        list_IDs=df_train.index.values,
        df=df_train,
        protein_language_model_embedding_dim=args.protein_language_model_embedding_dim,  
        smiles_db_path=args.smiles_db_path,
        protein_db_path=args.protein_db_path,
        is_LigandGCN=args.is_LigandGCN, 
        is_unimol_Ligand=args.is_unimol_Ligand, 
        is_ProteinCNN=args.is_ProteinCNN, 
        is_esm_Protein=args.is_esm_Protein,
        max_drug_nodes=args.ligand_max_nodes, 
        protein_padding_length=args.protein_padding_length, 
        drug_embedding_dim=args.drug_embedding_dim
    )

    val_dataset = DTIDataset(
        list_IDs=df_val.index.values,
        df=df_val,
        protein_language_model_embedding_dim=args.protein_language_model_embedding_dim,  
        smiles_db_path=args.smiles_db_path,                      
        protein_db_path=args.protein_db_path,                                           
        is_LigandGCN=args.is_LigandGCN, 
        is_unimol_Ligand=args.is_unimol_Ligand, 
        is_ProteinCNN=args.is_ProteinCNN, 
        is_esm_Protein=args.is_esm_Protein,
        max_drug_nodes=args.ligand_max_nodes, 
        protein_padding_length=args.protein_padding_length, 
        drug_embedding_dim=args.drug_embedding_dim
)

    
    return train_dataset, val_dataset


def create_data_loaders(train_dataset, val_dataset, args):
    """Create data loaders for training and validation."""
    train_params = {
        'batch_size': args.batch_size, 
        'shuffle': True, 
        'num_workers': args.num_workers,
        'drop_last': True, 
        'collate_fn': graph_collate_func
    }
    
    val_params = {
        'batch_size': args.batch_size, 
        'shuffle': False, 
        'num_workers': args.num_workers,
        'drop_last': False, 
        'collate_fn': graph_collate_func
    }
    
    train_loader = DataLoader(train_dataset, **train_params)
    val_loader = DataLoader(val_dataset, **val_params)
    
    return train_loader, val_loader


def main():
    """Main function to run the model training."""
    # Parse arguments and set up environment
    args = parse_arguments()
    device = setup_device(args.run_device)
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    set_seed(args.seed)
    
    # Set up output directory
    args.result_output_dir = "./results/" + args.data.split("/")[-1]
    mkdir(args.result_output_dir)
    
    # Print configuration information
    print(f"Hyperparameters: {dict(vars(args))}")
    print(f"Running on: {device}", end="\n\n")
    
    # Load datasets and create data loaders
    train_dataset, val_dataset = load_datasets(args.data, args)
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, args)
    
    # Initialize model, optimizer, and trainer
    model = HitScreen(args).to(device)
    # print(model)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    torch.backends.cudnn.benchmark = True
    
    trainer = Trainer(model, optimizer, device, train_loader, val_loader, args, experiment=None)
    
    # Train the model
    result = trainer.train()
    
    # Save model architecture
    with open(os.path.join(args.result_output_dir, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))
    
    print(f"Directory for saving result: {args.result_output_dir}")
    
    return result


if __name__ == '__main__':
    start_time = time()
    result = main()
    end_time = time()
    print(f"Total running time: {round(end_time - start_time, 2)}s")