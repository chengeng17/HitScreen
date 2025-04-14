import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from models import HitScreen, binary_cross_entropy
from utils import graph_collate_func
from dataloader_sql import DTIDataset
from tqdm import tqdm

# =========================
# 1. Argument Parsing
# =========================
parser = argparse.ArgumentParser(description="HitScreen for DTI prediction")

# Dataset and model configuration
parser.add_argument('--test_file', default="./data/11betaHSD1/11betaHSD1.csv", type=str, help="Path to the test dataset")
parser.add_argument('--model_path', default="./model/Ankh_Large/Ankh_Large_model.pth", type=str, help="Path to the pre-trained model")

# embedding
parser.add_argument('--is_LigandGCN', type=bool, default=True, help="Whether to use GCN for extracting ligand features")
parser.add_argument('--is_unimol_Ligand', type=bool, default=True, help="Whether to use unimol for extracting ligand features")
parser.add_argument('--is_ProteinCNN', type=bool, default=False, help="Whether to use CNN for extracting protein features")
parser.add_argument('--is_esm_Protein', type=bool, default=True, help="Whether to use ESM for extracting protein features")
parser.add_argument('--ligand_max_nodes', default=300, type=int, help="Maximum number of nodes for ligands")

# Drug feature extractor
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


# Protein feature extractor
parser.add_argument('--protein_language_model_embedding_dim', default=1536, type=int, help="Embedding dimension for protein language models: Ankh = 1536, ProtT5 = 1024, ESM-2 150M = 640")
parser.add_argument('--protein_embedding_dim', default=128, type=int, help="Embedding dimension for protein features")
parser.add_argument('--protein_num_filters', default=[128, 128, 128], type=list, 
                       help="Number of filters for protein feature extractor")
parser.add_argument('--protein_kernel_size', default=[3, 6, 9], type=list, 
                       help="Kernel sizes for protein feature extractor")
parser.add_argument('--protein_padding', default=True, type=bool, 
                       help="Whether to pad protein features")

parser.add_argument('--layer', default=1, type=int, help="Number of layers")
parser.add_argument('--hidden_size', default=128, type=int, help="Hidden size")
parser.add_argument('--multi_head', default=8, type=int, help="Number of multi heads")
parser.add_argument('--dropout', default=0.2, type=float, help="Dropout rate")


# Database paths
parser.add_argument('--smiles_db_path', default='./data/11betaHSD1/Uni-Mol_molecule_embeddings_no_h_11betaHSD1_ligands.db', type=str, help="Path to SQLite database with SMILES embeddings")
parser.add_argument('--protein_db_path', default='./data/11betaHSD1/Ankh_Large_target_protein_embeddings_11betaHSD1.db', type=str, help="Path to SQLite database with protein embeddings")

# Padding and dimension parameters
parser.add_argument('--protein_padding_length', default=1200, type=int, help="Length to pad protein sequences to")
parser.add_argument('--drug_embedding_dim', default=512, type=int, help="Dimension of drug embeddings")

# Device configuration
parser.add_argument('--run_device', default='cuda:5', type=str, help="Compute device (e.g., 'cpu', 'cuda:0', 'cuda:1')")

# MLP decoder
parser.add_argument('--decoder_name', default="MLP", type=str, help="Decoder name")
parser.add_argument('--decoder_in_dim', default=256, type=int, help="Input dimension for decoder")
parser.add_argument('--decoder_hidden_dim', default=128, type=int, help="Hidden dimension for decoder")
parser.add_argument('--decoder_out_dim', default=64, type=int, help="Output dimension for decoder")
parser.add_argument('--decoder_binary', default=1, type=int, help="Binary setting for decoder")


args = parser.parse_args()

# =========================
# 2. Device Setup
# =========================
# Setup device (CUDA or CPU)
if torch.cuda.is_available() and 'cuda' in args.run_device:
    device = torch.device(args.run_device)
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

# =========================
# 3. Load Model
# =========================
# Load the ScreeningCPI model
model = HitScreen(args).to(device)
model.load_state_dict(torch.load(args.model_path))

# =========================
# 4. Prepare Test Data
# =========================
# Load the test dataset
df_test = pd.read_csv(args.test_file)

# Prepare the DTIDataset for testing
test_dataset = DTIDataset(
    list_IDs=df_test.index.values,
    df=df_test,
    is_LigandGCN=args.is_LigandGCN,
    is_unimol_Ligand=args.is_unimol_Ligand,
    is_ProteinCNN=args.is_ProteinCNN,
    is_esm_Protein=args.is_esm_Protein,
    protein_language_model_embedding_dim=args.protein_language_model_embedding_dim,
    max_drug_nodes=args.ligand_max_nodes,
    smiles_db_path=args.smiles_db_path,
    protein_db_path=args.protein_db_path,
    protein_padding_length=args.protein_padding_length,
    drug_embedding_dim=args.drug_embedding_dim
)

# Create DataLoader for the test dataset
params = {
    'batch_size': 1,
    'shuffle': False,
    'num_workers': 0,
    'drop_last': True,
    'collate_fn': graph_collate_func
}

training_generator = DataLoader(test_dataset, **params)

# Extract the list of IDs for saving predictions
id_list = df_test["ID"].tolist()

# Set the model to evaluation mode
model.eval()

# =========================
# 5. Inference and Save Results
# =========================
with torch.no_grad():
    with open('./data/11betaHSD1/11betaHSD1_screen_results.txt', 'w') as f:
        for (v_d_gcn, v_d_atomic_embedding, v_p_cnn, v_p_esm, labels), id in zip(tqdm(training_generator), id_list):
            v_d_gcn = v_d_gcn.to(device) if v_d_gcn is not None else None
            v_d_atomic_embedding = v_d_atomic_embedding.to(device) if v_d_atomic_embedding is not None else None
            v_p_cnn = v_p_cnn.to(device) if v_p_cnn is not None else None
            v_p_esm = v_p_esm.to(device) if v_p_esm is not None else None
            labels = labels.float().to(device) if labels is not None else None

            # Perform prediction
            _, score = model(v_d_gcn, v_d_atomic_embedding, v_p_cnn, v_p_esm)

            # Calculate loss
            n, loss = binary_cross_entropy(score, labels)

            # Collect labels and predictions
            y_label = labels.to("cpu").tolist()
            y_pred = n.to("cpu").tolist()

            # Write results to file
            f.write(f'{id}\t{y_pred}\t{y_label}\n')
