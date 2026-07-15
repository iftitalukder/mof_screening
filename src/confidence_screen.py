"""
confidence_screen.py
---------------------
Co-author point #6. Turns model probability outputs into a practical
screening table: prioritize / needs further test / deprioritize,
each row annotated with its top SHAP-driving feature as a
human-readable "reason", matching the docx example table format.
"""
import os
import numpy as np
import pandas as pd

import config
import shap_analysis


def confidence_bucket(prob: float) -> str:
    if prob >= 0.7:
        return "prioritize"
    elif prob >= 0.4:
        return "needs further test"
    else:
        return "deprioritize"


def top_reason_for_row(shap_row_values, feature_names, k: int = 2):
    """Returns a short string naming the top-k SHAP-contributing features
    for a single prediction, signed by direction of contribution."""
    idx = np.argsort(np.abs(shap_row_values))[::-1][:k]
    parts = []
    for i in idx:
        sign = "+" if shap_row_values[i] > 0 else "-"
        parts.append(f"{sign}{feature_names[i]}")
    return " / ".join(parts)


def build_screening_table(app_name: str, master: pd.DataFrame, n_examples: int = 25):
    model, explainer, shap_values, X_test, split_idx = shap_analysis.compute_shap_for_application(
        app_name, master
    )

    proba = model.predict_proba(X_test)[:, 1]
    feature_names = X_test.columns.tolist()

    rows = []
    test_idx = split_idx["test"]
    for row_pos, mof_idx in enumerate(test_idx[: min(n_examples, len(test_idx))]):
        prob = proba[row_pos]
        reason = top_reason_for_row(shap_values.values[row_pos], feature_names)
        rows.append(
            {
                "mof_row_id": int(mof_idx),
                "precursor": master.loc[mof_idx, "precursor"],
                "prediction": "promising" if prob >= 0.5 else "not promising",
                "confidence": round(float(prob), 3),
                "shap_reason": reason,
                "action": confidence_bucket(prob),
            }
        )
    return pd.DataFrame(rows)


def main():
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    master = pd.read_csv(os.path.join(config.DATA_PROCESSED, "master_table.csv"))

    for app_name in config.APPLICATIONS:
        print(f"\n=== Confidence-based screening: {app_name} ===")
        table = build_screening_table(app_name, master)
        table.to_csv(
            os.path.join(config.RESULTS_TABLES, f"confidence_screen_{app_name}.csv"),
            index=False,
        )
        print(table.head(10).to_string(index=False))
        print(f"... ({len(table)} rows total)")

    print(f"\nSaved to {config.RESULTS_TABLES}/")


if __name__ == "__main__":
    main()
