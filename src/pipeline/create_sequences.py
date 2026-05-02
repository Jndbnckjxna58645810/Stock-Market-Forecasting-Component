import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger("pipeline.create_sequences")

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []

    y = np.asarray(y).ravel() 

    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i]) 
        y_seq.append(y[i])

    logger.info(f"Sequences for length={seq_len} created")

    return np.array(X_seq), np.array(y_seq)