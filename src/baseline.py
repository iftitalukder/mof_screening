"""
baseline.py
-----------
Fix for #18: a simple, non-ML baseline. For every test MOF, find its
k nearest training MOFs by Tanimoto similarity of the linker Morgan
fingerprint, and predict the (similarity-weighted) fraction of
positive labels among those neighbors as a pseudo-probability. This
tests whether the classifiers in models.py are learning anything
beyond "look up the most chemically similar training example."
"""
import numpy as np
import pandas as pd
from rdkit import DataStructs
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import config


def _to_bitvect(fp_row: np.ndarray):
    bv = DataStructs.ExplicitBitVect(len(fp_row))
    on_bits = np.nonzero(fp_row)[0].tolist()
    bv.SetBitsFromList(on_bits)
    return bv


def tanimoto_knn_predict(X_train_fp: pd.DataFrame, y_train: np.ndarray,
                          X_test_fp: pd.DataFrame, k: int = 5):
    train_bvs = [_to_bitvect(row) for row in X_train_fp.values]
    y_train = np.asarray(y_train)

    probs = np.zeros(len(X_test_fp))
    for i, row in enumerate(X_test_fp.values):
        test_bv = _to_bitvect(row)
        sims = np.array(DataStructs.BulkTanimotoSimilarity(test_bv, train_bvs))
        top_k = np.argsort(sims)[::-1][:k]
        weights = sims[top_k]
        if weights.sum() == 0:
            probs[i] = y_train.mean()
        else:
            probs[i] = np.average(y_train[top_k], weights=weights)
    return probs


def evaluate_baseline(fp_feature_df: pd.DataFrame, master: pd.DataFrame,
                       label_col: str, split_idx: dict, k: int = 5):
    fp_cols = [c for c in fp_feature_df.columns if c.startswith("fp_")]
    X_train = fp_feature_df.loc[split_idx["train"], fp_cols]
    X_test = fp_feature_df.loc[split_idx["test"], fp_cols]
    y_train = master.loc[split_idx["train"], label_col].values
    y_test = master.loc[split_idx["test"], label_col].values

    probs = tanimoto_knn_predict(X_train, y_train, X_test, k=k)
    preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs) if len(set(y_test)) > 1 else float("nan"),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
