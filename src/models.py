"""
models.py
---------
Trains the 3 models (Logistic Regression, Random Forest, XGBoost) x
4 feature groups x 2 applications, on both the random split and the
metal-holdout split. Records accuracy/F1/ROC-AUC and wall-clock
train/inference time (co-author point #7, "low-compute comparison").

Circularity guard: assert_no_leakage() is called before every fit.
"""
import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb

import config
import data_prep
import splits as splits_mod

MODEL_BUILDERS = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, random_state=config.SEED
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, random_state=config.SEED, n_jobs=-1
    ),
    "xgboost": lambda: xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        random_state=config.SEED, eval_metric="logloss", n_jobs=-1,
    ),
}


def load_feature_group(name: str) -> pd.DataFrame:
    path = os.path.join(config.DATA_PROCESSED, f"features_{name}.parquet")
    return pd.read_parquet(path)


def train_and_eval(model_name, feature_df, master, label_col, split_idx):
    data_prep.assert_no_leakage(feature_df.columns, label_col.replace("label_", ""))

    X_train = feature_df.loc[split_idx["train"]].values
    y_train = master.loc[split_idx["train"], label_col].values
    X_test = feature_df.loc[split_idx["test"]].values
    y_test = master.loc[split_idx["test"], label_col].values

    model = MODEL_BUILDERS[model_name]()

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    inference_time = time.perf_counter() - t0

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else float("nan"),
        "train_time_sec": train_time,
        "inference_time_sec": inference_time,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
    return model, metrics


def run_ablation(master: pd.DataFrame):
    """
    Feature-group ablation (co-author point #2) x model x application,
    on the random split. Returns a tidy results dataframe and a dict of
    fitted models keyed by (application, feature_group, model_name) for
    downstream SHAP analysis.
    """
    feature_groups = {
        "precursor_only": load_feature_group("precursor_only"),
        "descriptor_only": load_feature_group("descriptor_only"),
        "precursor_descriptor": load_feature_group("precursor_descriptor"),
    }

    rows = []
    fitted = {}

    for app_name in config.APPLICATIONS:
        label_col = f"label_{app_name}"
        split_idx = splits_mod.random_split(master, label_col)

        for group_name, feat_df in feature_groups.items():
            for model_name in config.MODEL_NAMES:
                model, metrics = train_and_eval(
                    model_name, feat_df, master, label_col, split_idx
                )
                rows.append(
                    {
                        "application": app_name,
                        "feature_group": group_name,
                        "model": model_name,
                        **metrics,
                    }
                )
                fitted[(app_name, group_name, model_name)] = model
                print(
                    f"[{app_name}] {group_name:22s} {model_name:20s} "
                    f"acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} "
                    f"auc={metrics['roc_auc']:.3f}"
                )

    return pd.DataFrame(rows), fitted, feature_groups, split_idx


def run_metal_holdout_eval(master: pd.DataFrame, feature_groups: dict):
    """
    Same models, but evaluated on the metal-holdout split -- this is
    the leakage-aware generalization check (co-author points #3/#4).
    Uses the strongest feature group (precursor_descriptor) only, since
    this is a generalization stress-test, not a full ablation repeat.
    """
    rows = []
    feat_df = feature_groups["precursor_descriptor"]

    for app_name in config.APPLICATIONS:
        label_col = f"label_{app_name}"
        split_idx = splits_mod.metal_holdout_split(master, label_col)

        for model_name in config.MODEL_NAMES:
            model, metrics = train_and_eval(
                model_name, feat_df, master, label_col, split_idx
            )
            rows.append(
                {
                    "application": app_name,
                    "model": model_name,
                    "eval_type": "metal_holdout",
                    **metrics,
                }
            )
            print(
                f"[metal-holdout] [{app_name}] {model_name:20s} "
                f"acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} "
                f"auc={metrics['roc_auc']:.3f}"
            )
    return pd.DataFrame(rows)


def main():
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    master = pd.read_csv(os.path.join(config.DATA_PROCESSED, "master_table.csv"))

    print("=== Feature-group ablation (random split) ===")
    ablation_df, fitted_models, feature_groups, last_split = run_ablation(master)
    ablation_df.to_csv(
        os.path.join(config.RESULTS_TABLES, "ablation_results.csv"), index=False
    )

    print("\n=== Leakage-aware generalization check (metal-holdout split) ===")
    holdout_df = run_metal_holdout_eval(master, feature_groups)
    holdout_df.to_csv(
        os.path.join(config.RESULTS_TABLES, "metal_holdout_results.csv"), index=False
    )

    print("\n=== Low-compute comparison table ===")
    compute_rows = []
    for _, row in ablation_df[ablation_df.feature_group == "precursor_descriptor"].iterrows():
        compute_rows.append(
            {
                "model": row["model"],
                "training_time_sec": round(row["train_time_sec"], 4),
                "inference_time_sec_per_test_set": round(row["inference_time_sec"], 4),
                "gpu_needed": "no",
                "interpretability": {
                    "logistic_regression": "high",
                    "random_forest": "medium",
                    "xgboost": "SHAP-compatible",
                }[row["model"]],
            }
        )
    compute_df = pd.DataFrame(compute_rows).drop_duplicates(subset="model")
    compute_df.to_csv(
        os.path.join(config.RESULTS_TABLES, "compute_comparison.csv"), index=False
    )
    print(compute_df.to_string(index=False))

    print(f"\nSaved tables to {config.RESULTS_TABLES}/")
    return ablation_df, fitted_models, feature_groups, master


if __name__ == "__main__":
    main()
