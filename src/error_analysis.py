"""
error_analysis.py
------------------
Co-author point #8. Pulls false positives (model says promising, isn't)
and false negatives (model misses a truly promising MOF) from the test
set, and groups them by a likely cause using simple heuristics:
    - missing_descriptor : linker SMILES failed to parse (all-zero
                            descriptor row) -> model was flying blind
    - unusual_metal       : primary metal is rare in the training data
                            (outside the top-15 bucketed metals ->
                            landed in "metal_other")
    - out_of_domain_size  : linker MolWt is in the extreme tail
                            (<5th or >95th percentile) of the training
                            distribution
    - unexplained         : none of the above -- genuine model error
"""
import os
import pandas as pd

import config
import shap_analysis
import featurize


def classify_error_cause(row, mol_wt_low, mol_wt_high, top_metals: set) -> str:
    if row.get("MolWt", 0) == 0 and row.get("TPSA", 0) == 0:
        return "missing_descriptor"
    if row.get("primary_metal", None) not in top_metals:
        return "unusual_metal"
    mol_wt = row.get("MolWt", None)
    if mol_wt is not None and (mol_wt < mol_wt_low or mol_wt > mol_wt_high):
        return "out_of_domain_size"
    return "unexplained"


def build_error_table(app_name: str, master: pd.DataFrame):
    model, explainer, shap_values, X_test, split_idx = shap_analysis.compute_shap_for_application(
        app_name, master
    )
    label_col = f"label_{app_name}"

    y_true = master.loc[split_idx["test"], label_col].values
    y_pred = model.predict(X_test)

    train_desc = pd.read_parquet(
        os.path.join(config.DATA_PROCESSED, "features_descriptor_only.parquet")
    ).loc[split_idx["train"]]
    mol_wt_low = train_desc["MolWt"].quantile(0.05)
    mol_wt_high = train_desc["MolWt"].quantile(0.95)

    metal_feats = master.loc[split_idx["train"], "metal_frag"].apply(
        lambda m: __import__("metal_properties").metal_fragment_features(m)["primary_metal"]
    )
    top_metals = set(metal_feats.value_counts().head(15).index)

    desc_df = pd.read_parquet(
        os.path.join(config.DATA_PROCESSED, "features_descriptor_only.parquet")
    ).loc[split_idx["test"]]

    rows = []
    for row_pos, mof_idx in enumerate(split_idx["test"]):
        true_label = y_true[row_pos]
        pred_label = y_pred[row_pos]
        if true_label == pred_label:
            continue  # only record errors

        error_type = "false_positive" if pred_label == 1 and true_label == 0 else "false_negative"

        feat_row = desc_df.loc[mof_idx].to_dict()
        primary_metal = __import__("metal_properties").metal_fragment_features(
            master.loc[mof_idx, "metal_frag"]
        )["primary_metal"]
        feat_row["primary_metal"] = primary_metal

        cause = classify_error_cause(feat_row, mol_wt_low, mol_wt_high, top_metals)

        rows.append(
            {
                "mof_row_id": int(mof_idx),
                "precursor": master.loc[mof_idx, "precursor"],
                "error_type": error_type,
                "likely_cause": cause,
            }
        )
    return pd.DataFrame(rows)


def main():
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    master = pd.read_csv(os.path.join(config.DATA_PROCESSED, "master_table.csv"))

    for app_name in config.APPLICATIONS:
        print(f"\n=== Error analysis: {app_name} ===")
        err_df = build_error_table(app_name, master)
        err_df.to_csv(
            os.path.join(config.RESULTS_TABLES, f"error_analysis_{app_name}.csv"),
            index=False,
        )
        print(f"Total errors: {len(err_df)}")
        if len(err_df):
            print(err_df.groupby(["error_type", "likely_cause"]).size().to_string())

    print(f"\nSaved to {config.RESULTS_TABLES}/")


if __name__ == "__main__":
    main()
