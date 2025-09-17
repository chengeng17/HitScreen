import argparse
import csv
import glob
from statistics import mean
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from rdkit.ML.Scoring import Scoring
from tqdm import tqdm


def main(file, file_result) -> None:
    ntb_top = []
    ntb_total = []
    auroc = []
    auprc = []
    bedroc = []

    with open(file, "r") as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
    id_to_pred = {line[0]: float(line[1].replace("[","").replace("]","")) for line in lines}

    pdbs = sorted(list(set([key.split("_")[0] for key in id_to_pred.keys()])))

    # Create empty DataFrame
    df = pd.DataFrame(columns=["target", "actives", "decoys", "EF0.1%", "EF0.5%", "EF1%", "EF5%", "AUROC", "AUPRC", "BEDROC"])

    for pdb in tqdm(pdbs, desc="Processing"):
        selected_keys = [key for key in id_to_pred.keys() if key.split("_")[0] == pdb]
        preds = [id_to_pred[key] for key in selected_keys]
        preds, selected_keys = zip(*sorted(zip(preds, selected_keys), reverse=True))
        active = "actives"
        true_binders = [key for key in selected_keys if key.split('_')[1] in active]
        ntb_top_pdb, ntb_total_pdb  = [], []
        ef = []
        for topn in [0.001, 0.005, 0.01, 0.05]:
            n = int(topn * len(selected_keys))
            top_keys = selected_keys[:n]
            n_top_true_binder = len(list(set(top_keys) & set(true_binders)))
            true_n = len(true_binders) * topn
            ntb_top_pdb.append(n_top_true_binder)
            ntb_total_pdb.append(len(true_binders) * topn)
            ef_c =  n_top_true_binder / true_n
            ef.append(round(ef_c, 3))  # Round to 3 decimal places
        ntb_top.append(ntb_top_pdb)
        ntb_total.append(ntb_total_pdb)

        # Calculate AUROC, AUPRC, BEDROC
        y_true = [1 if key in true_binders else 0 for key in selected_keys]
        y_scores = preds
        auroc.append(round(roc_auc_score(y_true, y_scores), 3))  # Round to 3 decimal places
        auprc.append(round(average_precision_score(y_true, y_scores), 3))  # Round to 3 decimal places
        bedroc_scores = [[score, truth] for score, truth in zip(y_scores, y_true)]
        bedroc.append(round(Scoring.CalcBEDROC(bedroc_scores, 1, 80.5), 3))  # Round to 3 decimal places

        # Add row to DataFrame
        new_row = [pdb, len(true_binders), len(selected_keys) - len(true_binders), ef[0], ef[1], ef[2], ef[3], auroc[-1], auprc[-1], bedroc[-1]]
        df.loc[len(df)] = new_row

    # After the loop
    # Calculate mean for each column from the second column onwards
    means = df.iloc[:, 1:].mean()

    # Create a new row with the label "average EF:"
    new_row = pd.Series(["average EF:"] + means.round(3).tolist(), index=df.columns)  # Round means to 3 decimal places

    # Set the new row in the DataFrame
    df.loc[len(df)] = new_row

    # Save DataFrame to CSV
    df.to_csv(file_result, index=False)

    print(f"Task completed. Results saved to {file_result}")


if __name__ == "__main__":
    file = "./pre_dude.txt"
    file_result = file.replace(".txt", "_EF.csv")
    main(file, file_result)





