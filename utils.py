import os
import random
import numpy as np
import torch
import dgl
import logging

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# def graph_collate_func(x):
#     d, p, y = zip(*x)
#     d = dgl.batch(d)
#     p_tensor = torch.stack(p)
#     return d, p_tensor, torch.tensor(y)


def graph_collate_func(batch):
    v_d, v_d_atomic_embedding, v_p_cnn, v_p_esm, y = zip(*batch)

    # Batch graph data if not None, else return None
    if any(item is not None for item in v_d):
        v_d_gcn = dgl.batch([item for item in v_d if item is not None])
    else:
        v_d_gcn = None

    # Stack embeddings if not None, else return None
    v_d_atomic_embedding_tensor = torch.stack([item for item in v_d_atomic_embedding if item is not None]) if any(item is not None for item in v_d_atomic_embedding) else None
    v_p_esm_tensor = torch.stack([item for item in v_p_esm if item is not None]) if any(item is not None for item in v_p_esm) else None
    v_p_cnn_tensor = torch.stack([item for item in v_p_cnn if item is not None]) if any(item is not None for item in v_p_cnn) else None

    # Convert labels to tensor
    y_tensor = torch.tensor(y)

    return v_d_gcn, v_d_atomic_embedding_tensor, v_p_cnn_tensor, v_p_esm_tensor, y_tensor

# def graph_collate_func(x):
#     v_d,v_d_atomic_embedding,v_d_cls_embedding, v_p, y = zip(*x)
#     d = torch.stack(v_d)
#     # d = dgl.batch(v_d)
#     # v_d_atomic_embedding_tensor = torch.stack(v_d_atomic_embedding)
#     # v_d_cls_embedding_tensor = torch.stack(v_d_cls_embedding)
#     v_p_tensor = torch.stack(v_p)
#     return d, v_p_tensor, torch.tensor(y)


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding
