"""
models.py (v2)
---------------
Fixes bundled here:
  #8  -- validation fold is now actually used, for hyperparameter selection.
  #9  -- small hyperparameter grids (config.HYPERPARAM_GRIDS), selected by
         validation AUC, not hardcoded blind.
  #10 -- StandardScaler (fit on train only) applied for Logistic Regression.
  #14 -- CalibratedClassifierCV (sigmoid/Platt), fit on the validation
         fold, wraps the final chosen model for the confidence table.
  #16 -- precision, recall, average precision (PR-AUC), and enrichment
         factor at top-10%/top-25% are now reported alongside
         accuracy/F1/ROC-AUC.
  #17 -- every result is averaged over config.SEEDS (mean +/- std), not
         a single lucky/unlucky split.
  #19 -- model size (serialized, KB) and peak memory during fit
         (tracemalloc) are recorded alongside wall-clock time, and
         inference time is reported per-sample as well as per-test-set.
"""
import os
import io
import pickle
import time
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    average_precision_score,
)
import xgboost as xgb

import config
import data_prep
import splits as splits_mod
import baseline as baseline_mod
import shap_analysis


def _build_model(model_name: str, params: dict):
    if model_name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=config.SEED, **params))
    elif model_name == "random_forest":
        return RandomForestClassifier(random_state=config.SEED, n_jobs=-1, **params)
    elif model_name == "xgboost":
        return xgb.XGBClassifier(random_state=config.SEED, eval_metric="logloss", n_jobs=-1, **params)
    raise ValueError(model_name)


def _enrichment_factor(y_true, y_proba, top_frac):
    n = len(y_true)
    n_top = max(1, int(n * top_frac))
    order = np.argsort(y_proba)[::-1][:n_top]
    hit_rate_top = y_true[order].mean()
    base_rate = y_true.mean()
    return hit_rate_top / base_rate if base_rate > 0 else float("nan")


def tune_and_fit(model_name, X_train, y_train, X_valid, y_valid):
    """Fix #8/#9: pick hyperparameters by validation AUC, then refit on train."""
    best_auc, best_params, best_model = -1, None, None
    for params in config.HYPERPARAM_GRIDS[model_name]:
        model = _build_model(model_name, params)
        model.fit(X_train, y_train)
        if len(set(y_valid)) > 1:
            proba = model.predict_proba(X_valid)[:, 1]
            auc = roc_auc_score(y_valid, proba)
        else:
            auc = 0.5
        if auc > best_auc:
            best_auc, best_params, best_model = auc, params, model
    return best_model, best_params, best_auc


def train_and_eval(model_name, feature_df, master, app_name, split_idx):
    data_prep.assert_no_leakage(feature_df.columns, app_name)

    labels, cutoff = splits_mod.make_labels_from_train_threshold(master, split_idx["train"], app_name)

    X_train = feature_df.loc[split_idx["train"]]
    y_train = labels.loc[split_idx["train"]].values
    X_valid = feature_df.loc[split_idx["valid"]]
    y_valid = labels.loc[split_idx["valid"]].values
    X_test = feature_df.loc[split_idx["test"]]
    y_test = labels.loc[split_idx["test"]].values

    tracemalloc.start()
    t0 = time.perf_counter()
    model, best_params, valid_auc = tune_and_fit(model_name, X_train, y_train, X_valid, y_valid)
    train_time = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    inference_time = time.perf_counter() - t0

    model_size_kb = len(pickle.dumps(model)) / 1024

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_test, y_proba) if len(set(y_test)) > 1 else float("nan"),
        "enrichment_top10": _enrichment_factor(y_test, y_proba, 0.10),
        "enrichment_top25": _enrichment_factor(y_test, y_proba, 0.25),
        "train_time_sec": train_time,
        "inference_time_sec_total": inference_time,
        "inference_time_sec_per_sample": inference_time / max(1, len(y_test)),
        "peak_train_memory_kb": peak_mem / 1024,
        "model_size_kb": model_size_kb,
        "best_params": str(best_params),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "label_cutoff_value": cutoff,
    }
    return model, metrics


def run_multiseed_ablation(master: pd.DataFrame, feature_groups: dict, split_fn, split_name: str,
                            seeds=None):
    seeds = seeds or config.SEEDS
    rows = []
    for app_name in config.APPLICATIONS:
        for group_name, feat_df in feature_groups.items():
            for model_name in config.MODEL_NAMES:
                per_seed = []
                for seed in seeds:
                    split_idx = split_fn(master, seed=seed)
                    _, metrics = train_and_eval(model_name, feat_df, master, app_name, split_idx)
                    per_seed.append(metrics)

                agg = {"application": app_name, "feature_group": group_name, "model": model_name,
                       "split_type": split_name, "n_seeds": len(seeds)}
                for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                            "enrichment_top10", "enrichment_top25", "train_time_sec",
                            "inference_time_sec_per_sample", "peak_train_memory_kb", "model_size_kb"]:
                    vals = [m[key] for m in per_seed]
                    agg[f"{key}_mean"] = np.mean(vals)
                    agg[f"{key}_std"] = np.std(vals)
                rows.append(agg)
                print(f"[{split_name}][{app_name}] {group_name:22s} {model_name:20s} "
                      f"AUC={agg['roc_auc_mean']:.3f}+/-{agg['roc_auc_std']:.3f} "
                      f"F1={agg['f1_mean']:.3f}+/-{agg['f1_std']:.3f}")
    return pd.DataFrame(rows)


def run_baseline_comparison(master: pd.DataFrame, feature_groups: dict, split_fn, split_name: str, seeds=None):
    """Fix #18: Tanimoto-kNN baseline alongside the ML models."""
    seeds = seeds or config.SEEDS
    fp_df = feature_groups["precursor_only"]
    rows = []
    for app_name in config.APPLICATIONS:
        per_seed = []
        for seed in seeds:
            split_idx = split_fn(master, seed=seed)
            labels, _ = splits_mod.make_labels_from_train_threshold(master, split_idx["train"], app_name)
            master_with_label = master.copy()
            master_with_label[f"label_{app_name}"] = labels
            m = baseline_mod.evaluate_baseline(fp_df, master_with_label, f"label_{app_name}", split_idx)
            per_seed.append(m)
        agg = {"application": app_name, "feature_group": "tanimoto_knn_baseline",
               "model": "5nn_tanimoto", "split_type": split_name, "n_seeds": len(seeds)}
        for key in ["accuracy", "f1", "roc_auc"]:
            vals = [m[key] for m in per_seed]
            agg[f"{key}_mean"] = np.mean(vals)
            agg[f"{key}_std"] = np.std(vals)
        rows.append(agg)
        print(f"[{split_name}][{app_name}] tanimoto_knn_baseline      "
              f"AUC={agg['roc_auc_mean']:.3f}+/-{agg['roc_auc_std']:.3f}")
    return pd.DataFrame(rows)


def main():
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    master = pd.read_csv(os.path.join(config.DATA_PROCESSED, "master_table.csv"))

    feature_groups = {
        "precursor_only": pd.read_parquet(os.path.join(config.DATA_PROCESSED, "features_precursor_only.parquet")),
        "descriptor_only": pd.read_parquet(os.path.join(config.DATA_PROCESSED, "features_descriptor_only.parquet")),
        "precursor_descriptor": pd.read_parquet(os.path.join(config.DATA_PROCESSED, "features_precursor_descriptor.parquet")),
    }

    # ---- Fix #11/#12: build shap_selected group from TRAIN-fold SHAP on
    # the primary (scaffold) split, then add it as a real 4th ablation arm.
    print("=== Building shap_selected feature group (train-fold SHAP only) ===")
    primary_split = splits_mod.scaffold_split(master, seed=config.SEED)
    shap_selected_frames = {}
    for app_name in config.APPLICATIONS:
        top_feats = shap_analysis.select_shap_features(
            app_name, master, feature_groups["precursor_descriptor"], primary_split
        )
        shap_selected_frames[app_name] = feature_groups["precursor_descriptor"][top_feats]
        print(f"  {app_name}: top {len(top_feats)} features selected from TRAIN fold only")

    print("\n=== PRIMARY evaluation: scaffold split (multi-seed) ===")
    all_results = []
    scaffold_results = run_multiseed_ablation(
        master, feature_groups, splits_mod.scaffold_split, "scaffold"
    )
    all_results.append(scaffold_results)

    # shap_selected evaluated separately per application (different feature sets)
    shap_rows = []
    for app_name in config.APPLICATIONS:
        feat_df = shap_selected_frames[app_name]
        for model_name in config.MODEL_NAMES:
            per_seed = []
            for seed in config.SEEDS:
                split_idx = splits_mod.scaffold_split(master, seed=seed)
                _, metrics = train_and_eval(model_name, feat_df, master, app_name, split_idx)
                per_seed.append(metrics)
            agg = {"application": app_name, "feature_group": "shap_selected", "model": model_name,
                   "split_type": "scaffold", "n_seeds": len(config.SEEDS)}
            for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                        "enrichment_top10", "enrichment_top25", "train_time_sec",
                        "inference_time_sec_per_sample", "peak_train_memory_kb", "model_size_kb"]:
                vals = [m[key] for m in per_seed]
                agg[f"{key}_mean"] = np.mean(vals)
                agg[f"{key}_std"] = np.std(vals)
            shap_rows.append(agg)
            print(f"[scaffold][{app_name}] shap_selected          {model_name:20s} "
                  f"AUC={agg['roc_auc_mean']:.3f}+/-{agg['roc_auc_std']:.3f}")
    all_results.append(pd.DataFrame(shap_rows))

    print("\n=== Baseline comparison: Tanimoto-kNN (scaffold split) ===")
    baseline_results = run_baseline_comparison(master, feature_groups, splits_mod.scaffold_split, "scaffold")
    all_results.append(baseline_results)

    print("\n=== Comparison split 1/2: group-aware random split (single reference seed) ===")
    random_results = run_multiseed_ablation(
        master, feature_groups, splits_mod.random_split, "random_group_aware", seeds=[config.SEED]
    )
    all_results.append(random_results)

    print("\n=== Comparison split 2/2: metal element-holdout split (single reference seed) ===")
    metal_results = run_multiseed_ablation(
        master, feature_groups, splits_mod.metal_element_holdout_split, "metal_element_holdout", seeds=[config.SEED]
    )
    all_results.append(metal_results)

    full_results = pd.concat(all_results, ignore_index=True)
    full_results.to_csv(os.path.join(config.RESULTS_TABLES, "ablation_results_full.csv"), index=False)
    print(f"\nSaved full results table -> {config.RESULTS_TABLES}/ablation_results_full.csv")

    return full_results, feature_groups, shap_selected_frames, master, primary_split


if __name__ == "__main__":
    main()
