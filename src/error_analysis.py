"""
error_analysis.py (v2)
------------------------
Same heuristic categories as v1, now run on the scaffold split (the
primary, rigorous split) instead of the old leaky random split, so the
error breakdown reflects genuine out-of-distribution generalization
failures rather than near-duplicate-chemistry misses.
"""
import os
import pandas as pd
import xgboost as xgb

import config
import data_prep
import splits as splits_mod
import metal_properties as mp


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
    feat_df = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "features_precursor_descriptor.parquet"))
    split_idx = splits_mod.scaffold_split(master, seed=config.SEED)
    labels, cutoff = splits_mod.make_labels_from_train_threshold(master, split_idx["train"], app_name)

    data_prep.assert_no_leakage(feat_df.columns, app_name)

    X_train = feat_df.loc[split_idx["train"]]
    y_train = labels.loc[split_idx["train"]]
    X_test = feat_df.loc[split_idx["test"]]
    y_test = labels.loc[split_idx["test"]].values

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=config.SEED, eval_metric="logloss", n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    desc_df = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "features_descriptor_only.parquet"))
    mol_wt_low = desc_df.loc[split_idx["train"], "MolWt"].quantile(0.05)
    mol_wt_high = desc_df.loc[split_idx["train"], "MolWt"].quantile(0.95)

    metal_feats = master.loc[split_idx["train"], "metal_frag"].apply(
        lambda m: mp.metal_fragment_features(m)["primary_metal"]
    )
    top_metals = set(metal_feats.value_counts().head(15).index)

    rows = []
    for row_pos, mof_idx in enumerate(split_idx["test"]):
        true_label = y_test[row_pos]
        pred_label = y_pred[row_pos]
        if true_label == pred_label:
            continue

        error_type = "false_positive" if pred_label == 1 and true_label == 0 else "false_negative"
        feat_row = desc_df.loc[mof_idx].to_dict()
        primary_metal = mp.metal_fragment_features(master.loc[mof_idx, "metal_frag"])["primary_metal"]
        feat_row["primary_metal"] = primary_metal
        cause = classify_error_cause(feat_row, mol_wt_low, mol_wt_high, top_metals)

        rows.append({
            "mof_id": int(master.loc[mof_idx, "mof_id"]),
            "precursor": master.loc[mof_idx, "precursor"],
            "error_type": error_type,
            "likely_cause": cause,
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    master = pd.read_csv(os.path.join(config.DATA_PROCESSED, "master_table.csv"))

    for app_name in config.APPLICATIONS:
        print(f"\n=== Error analysis (scaffold split): {app_name} ===")
        err_df = build_error_table(app_name, master)
        err_df.to_csv(os.path.join(config.RESULTS_TABLES, f"error_analysis_{app_name}.csv"), index=False)
        print(f"Total errors: {len(err_df)}")
        if len(err_df):
            print(err_df.groupby(["error_type", "likely_cause"]).size().to_string())

    print(f"\nSaved to {config.RESULTS_TABLES}/")


if __name__ == "__main__":
    main()
