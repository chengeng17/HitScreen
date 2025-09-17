import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import argparse
import os
from typing import Tuple, Optional

def calculate_fingerprint(smiles: str) -> Optional[object]:
    """
    Calculate molecular fingerprint using Morgan algorithm.
    
    Args:
        smiles (str): SMILES string
        
    Returns:
        Optional[object]: Morgan fingerprint or None if invalid SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return fingerprint
    except Exception as e:
        print(f"Error calculating fingerprint for {smiles}: {e}")
        return None

def find_negative_sample(df_positive: pd.DataFrame, 
                        current_row: pd.Series, 
                        used_rows: set, 
                        similarity_threshold: float = 0.3) -> Optional[pd.Series]:
    """
    Find a suitable negative sample for data augmentation.
    
    Args:
        df_positive: DataFrame containing positive samples
        current_row: Current positive sample row
        used_rows: Set of already used row indices
        similarity_threshold: Maximum similarity threshold for negative samples
        
    Returns:
        Optional[pd.Series]: Negative sample row or None if not found
    """
    target_cluster = current_row['target_cluster']
    current_smiles = current_row['SMILES']
    current_protein = current_row['Protein']
    
    # Get samples from different target clusters and proteins
    df_different_cluster = df_positive[
        (df_positive['target_cluster'] != target_cluster) & 
        (df_positive['Protein'] != current_protein)
    ]
    
    # Filter out already used samples
    potential_matches = df_different_cluster[~df_different_cluster.index.isin(used_rows)]
    
    if potential_matches.empty:
        return None
    
    # Calculate current molecule fingerprint
    current_fingerprint = calculate_fingerprint(current_smiles)
    if current_fingerprint is None:
        return None
    
    # Try to find a suitable negative sample
    for _, neg_row in potential_matches.iterrows():
        neg_fingerprint = calculate_fingerprint(neg_row['SMILES'])
        if neg_fingerprint is None:
            continue
            
        similarity = DataStructs.TanimotoSimilarity(current_fingerprint, neg_fingerprint)
        
        if similarity < similarity_threshold:
            return neg_row
    
    return None

def augment_data(input_file: str, 
                output_file: str, 
                repeat_times: int = 1,
                similarity_threshold: float = 0.3,
                random_seed: int = 0) -> None:
    """
    Perform data augmentation by generating negative samples.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        repeat_times: Number of times to repeat the augmentation process
        similarity_threshold: Maximum similarity threshold for negative samples
        random_seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Read CSV file
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Extract positive samples (Y == 1)
    df_positive = df[df['Y'] == 1]
    print(f"Found {len(df_positive)} positive samples")
    
    # Create empty DataFrame for negative samples
    df_negative = pd.DataFrame(columns=df.columns)
    
    # Process each repeat
    for repeat_idx in range(repeat_times):
        print(f"Processing repeat {repeat_idx + 1}/{repeat_times}...")
        used_rows = set()
        
        for _, row in tqdm(df_positive.iterrows(), 
                          total=df_positive.shape[0], 
                          desc=f'Generating negative samples (repeat {repeat_idx + 1})'):
            
            # Find suitable negative sample
            negative_row = find_negative_sample(df_positive, row, used_rows, similarity_threshold)
            
            if negative_row is not None:
                # Create negative sample
                neg_row = negative_row.to_frame().T
                neg_row['SMILES'] = row['SMILES']  # Replace SMILES with current one
                neg_row['Y'] = 0  # Set label to 0 (negative)
                df_negative = pd.concat([df_negative, neg_row], ignore_index=True)
                used_rows.add(negative_row.name)
    
    # Combine original and augmented data
    df_augmented = pd.concat([df, df_negative], ignore_index=True)
    
    # Save augmented dataset
    print(f"Saving augmented data to {output_file}...")
    df_augmented.to_csv(output_file, index=False)
    
    print(f"Data augmentation completed!")
    print(f"Original samples: {len(df)}")
    print(f"Generated negative samples: {len(df_negative)}")
    print(f"Total augmented samples: {len(df_augmented)}")

def main():
    parser = argparse.ArgumentParser(description='Data augmentation for drug-target interaction prediction')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file path')
    parser.add_argument('--repeat', '-r', type=int, default=1, help='Number of repeat times (default: 1)')
    parser.add_argument('--threshold', '-t', type=float, default=0.3, 
                   help='Molecular similarity threshold for negative sample generation (default: 0.3). '
                        'If dataset is small, consider increasing this value to generate more negative samples')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed (default: 0)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found!")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run data augmentation
    augment_data(args.input, args.output, args.repeat, args.threshold, args.seed)

if __name__ == "__main__":
    main()