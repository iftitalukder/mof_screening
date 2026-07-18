"""
run_shap_trend_report.py
--------------------------
Produces the human-readable SHAP trend table for the paper: fits the
primary model (scaffold split, precursor+descriptor, XGBoost) on TRAIN,
then explains it on TEST -- a legitimate post-hoc interpretation step
(does not feed back into any score reported elsewhere). Fingerprint
bits are decoded to an actual substructure where possible (fix #13).
"""
import os
import pandas as pd
import xgboost as xgb

import config
import data_prep
import splits as splits_mod
import shap_analysis


def main():
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    master = pd.read_csv(os.path.join(config.DATA_PROCESSED, "master_table.csv"))
    feat_df = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "features_precursor_descriptor.parquet"))

    for app_name in config.APPLICATIONS:
        print(f"\n=== SHAP trend table (scaffold split, test-fold interpretation): {app_name} ===")
        split_idx = splits_mod.scaffold_split(master, seed=config.SEED)
        labels, cutoff = splits_mod.make_labels_from_train_threshold(master, split_idx["train"], app_name)
        data_prep.assert_no_leakage(feat_df.columns, app_name)

        X_train = feat_df.loc[split_idx["train"]]
        y_train = labels.loc[split_idx["train"]]
        X_test = feat_df.loc[split_idx["test"]]

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=config.SEED, eval_metric="logloss", n_jobs=-1,
        )
        model.fit(X_train, y_train)

        sample_linkers = master.loc[split_idx["train"], "linker_smiles"].dropna().sample(
            min(500, len(split_idx["train"])), random_state=config.SEED
        ).tolist()

        trend_table = shap_analysis.build_trend_table_on_test(app_name, model, X_test, sample_linkers)
        trend_table.to_csv(os.path.join(config.RESULTS_TABLES, f"shap_trend_{app_name}.csv"), index=False)
        print(trend_table.to_string(index=False))

    print(f"\nSaved to {config.RESULTS_TABLES}/")


if __name__ == "__main__":
    main()
