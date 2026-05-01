import numpy as np

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []

    y = np.asarray(y).ravel() 

    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i]) 
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)