"""
confidence_screen.py (v2)
--------------------------
FIX for #14: raw XGBoost probabilities are NOT used directly for the
"prioritize/test further/deprioritize" cutoffs. A CalibratedClassifierCV
(Platt/sigmoid scaling) is fit on the VALIDATION fold (never the test
fold), and we report Brier score before/after calibration so the
calibration quality itself is checked, not assumed.

FIX for #15: the full test set is scored and saved (not an arbitrary
first-25-rows slice). The console preview still only shows a handful
of rows for readability, but the CSV written to disk contains every
test-set MOF.
"""
import os
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN_ESTIMATOR = True
except ImportError:
    _HAS_FROZEN_ESTIMATOR = False
from sklearn.metrics import brier_score_loss
import xgboost as xgb
import shap

import config
import data_prep
import splits as splits_mod


def confidence_bucket(prob: float) -> str:
    if prob >= 0.7:
        return "prioritize"
    elif prob >= 0.4:
        return "needs further test"
    else:
        return "deprioritize"


def top_reason_for_row(shap_row_values, feature_names, k: int = 2):
    idx = np.argsort(np.abs(shap_row_values))[::-1][:k]
    parts = []
    for i in idx:
        sign = "+" if shap_row_values[i] > 0 else "-"
        parts.append(f"{sign}{feature_names[i]}")
    return " / ".join(parts)


def build_screening_table(app_name: str, master: pd.DataFrame):
    feat_df = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "features_precursor_descriptor.parquet"))
    split_idx = splits_mod.scaffold_split(master, seed=config.SEED)
    labels, cutoff = splits_mod.make_labels_from_train_threshold(master, split_idx["train"], app_name)

    data_prep.assert_no_leakage(feat_df.columns, app_name)

    X_train = feat_df.loc[split_idx["train"]]
    y_train = labels.loc[split_idx["train"]]
    X_valid = feat_df.loc[split_idx["valid"]]
    y_valid = labels.loc[split_idx["valid"]]
    X_test = feat_df.loc[split_idx["test"]]
    y_test = labels.loc[split_idx["test"]]

    raw_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=config.SEED, eval_metric="logloss", n_jobs=-1,
    )
    raw_model.fit(X_train, y_train)

    raw_proba_test = raw_model.predict_proba(X_test)[:, 1]
    brier_before = brier_score_loss(y_test, raw_proba_test)

    if _HAS_FROZEN_ESTIMATOR:
        calibrated = CalibratedClassifierCV(FrozenEstimator(raw_model), method=config.CALIBRATION_METHOD)
    else:
        calibrated = CalibratedClassifierCV(raw_model, method=config.CALIBRATION_METHOD, cv="prefit")
    calibrated.fit(X_valid, y_valid)
    calibrated_proba_test = calibrated.predict_proba(X_test)[:, 1]
    brier_after = brier_score_loss(y_test, calibrated_proba_test)

    print(f"  [{app_name}] Brier score before calibration: {brier_before:.4f}, "
          f"after calibration: {brier_after:.4f} "
          f"({'improved' if brier_after < brier_before else 'did not improve'})")

    explainer = shap.TreeExplainer(raw_model)
    shap_values = explainer(X_test)
    feature_names = X_test.columns.tolist()

    rows = []
    for row_pos, mof_idx in enumerate(split_idx["test"]):
        prob = calibrated_proba_test[row_pos]
        reason = top_reason_for_row(shap_values.values[row_pos], feature_names)
        rows.append({
            "mof_id": int(master.loc[mof_idx, "mof_id"]),
            "precursor": master.loc[mof_idx, "precursor"],
            "prediction": "promising" if prob >= 0.5 else "not promising",
            "calibrated_confidence": round(float(prob), 3),
            "raw_confidence": round(float(raw_proba_test[row_pos]), 3),
            "shap_reason": reason,
            "action": confidence_bucket(prob),
        })
    table = pd.DataFrame(rows).sort_values("calibrated_confidence", ascending=False).reset_index(drop=True)
    return table, brier_before, brier_after


def main():
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    master = pd.read_csv(os.path.join(config.DATA_PROCESSED, "master_table.csv"))

    calibration_summary = []
    for app_name in config.APPLICATIONS:
        print(f"\n=== Confidence-based screening: {app_name} (full test set, calibrated) ===")
        table, brier_before, brier_after = build_screening_table(app_name, master)
        table.to_csv(os.path.join(config.RESULTS_TABLES, f"confidence_screen_{app_name}.csv"), index=False)
        print(table.head(10).to_string(index=False))
        print(f"... ({len(table)} rows total, full table saved)")
        calibration_summary.append({
            "application": app_name, "brier_before_calibration": brier_before,
            "brier_after_calibration": brier_after,
        })

    pd.DataFrame(calibration_summary).to_csv(
        os.path.join(config.RESULTS_TABLES, "calibration_summary.csv"), index=False
    )
    print(f"\nSaved to {config.RESULTS_TABLES}/")


if __name__ == "__main__":
    main()
