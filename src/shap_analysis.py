"""
shap_analysis.py
-----------------
Co-author point #5. Runs SHAP on the best model (XGBoost,
precursor_descriptor feature group, per application), produces:
  - a global feature-importance / trend table (feature, SHAP direction,
    materials-meaning note) like the docx example table
  - the "shap_selected" feature group: subsets precursor_descriptor to
    its top-K SHAP features and saves it as a 4th feature parquet, so
    models.py's ablation table 0 can include it too.
"""
import os
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

import config
import data_prep
import splits as splits_mod

# hand-written, human-readable notes for the descriptor-side features;
# fingerprint bits (fp_###) get a generic note since individual bits
# aren't chemically nameable without decoding the Morgan hash.
FEATURE_NOTES = {
    "MolWt": "larger linkers -> bigger pores, more framework mass",
    "TPSA": "polar surface area -> proxy for framework polarity/CO2 affinity",
    "NumHDonors": "H-bond donors on linker -> polar adsorption sites",
    "NumHAcceptors": "H-bond acceptors -> polar adsorption sites",
    "NumRotatableBonds": "linker flexibility -> framework flexibility/interpenetration risk",
    "NumAromaticRings": "aromaticity -> rigidity, pi-stacking, thermal stability",
    "RingCount": "overall ring content of the linker",
    "FractionCSP3": "sp3 fraction -> flexibility vs. rigidity of linker backbone",
    "MolLogP": "hydrophobicity -> pore surface polarity trade-off",
    "NumHeteroatoms": "heteroatom content -> polar/coordination sites",
    "HeavyAtomCount": "linker size proxy",
    "NumCarboxylGroups": "carboxylate count -> typical MOF coordination groups",
    "NumPyridylN": "pyridyl-type N count -> alternative coordination chemistry",
    "metal_electronegativity_avg": "higher electronegativity -> different metal-linker polarization",
    "metal_covalent_radius_avg": "metal size -> node geometry, pore shape",
    "metal_atomic_weight_avg": "correlates with framework density",
    "metal_atomic_number_avg": "correlates with metal identity/row in periodic table",
    "num_metal_atoms": "cluster nuclearity (single-metal vs. multi-metal SBU)",
    "num_distinct_metals": "mixed-metal node indicator",
}


def compute_shap_for_application(app_name: str, master: pd.DataFrame):
    feat_df = pd.read_parquet(
        os.path.join(config.DATA_PROCESSED, "features_precursor_descriptor.parquet")
    )
    label_col = f"label_{app_name}"
    split_idx = splits_mod.random_split(master, label_col)

    data_prep.assert_no_leakage(feat_df.columns, app_name)

    X_train = feat_df.loc[split_idx["train"]]
    y_train = master.loc[split_idx["train"], label_col]
    X_test = feat_df.loc[split_idx["test"]]

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        random_state=config.SEED, eval_metric="logloss", n_jobs=-1,
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    return model, explainer, shap_values, X_test, split_idx


def build_trend_table(shap_values, X_test, top_n: int = None):
    top_n = top_n or config.SHAP_TOP_K
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]

    rows = []
    for i in order:
        feat_name = X_test.columns[i]
        # direction: correlation sign between feature value and SHAP value
        vals = X_test.iloc[:, i].values
        svals = shap_values.values[:, i]
        if np.std(vals) == 0 or np.std(svals) == 0:
            direction = "flat"
        else:
            corr = np.corrcoef(vals, svals)[0, 1]
            if abs(corr) < 0.15:
                direction = "nonlinear"
            elif corr > 0:
                direction = "positive"
            else:
                direction = "negative"
        rows.append(
            {
                "feature": feat_name,
                "mean_abs_shap": mean_abs[i],
                "shap_trend": direction,
                "materials_meaning": FEATURE_NOTES.get(
                    feat_name,
                    "fingerprint bit (specific linker substructure)"
                    if feat_name.startswith("fp_")
                    else "metal identity indicator"
                    if feat_name.startswith("metal_")
                    else "n/a",
                ),
            }
        )
    return pd.DataFrame(rows)


def build_shap_selected_feature_group(shap_values, X_test, all_feat_df, top_n: int = None):
    top_n = top_n or config.SHAP_TOP_K
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    top_features = X_test.columns[order].tolist()
    return all_feat_df[top_features], top_features


def main():
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    master = pd.read_csv(os.path.join(config.DATA_PROCESSED, "master_table.csv"))
    all_feat_df = pd.read_parquet(
        os.path.join(config.DATA_PROCESSED, "features_precursor_descriptor.parquet")
    )

    for app_name in config.APPLICATIONS:
        print(f"\n=== SHAP: {app_name} ===")
        model, explainer, shap_values, X_test, split_idx = compute_shap_for_application(
            app_name, master
        )

        trend_table = build_trend_table(shap_values, X_test)
        trend_table.to_csv(
            os.path.join(config.RESULTS_TABLES, f"shap_trend_{app_name}.csv"),
            index=False,
        )
        print(trend_table.to_string(index=False))

        shap_selected_df, top_features = build_shap_selected_feature_group(
            shap_values, X_test, all_feat_df
        )
        shap_selected_df.to_parquet(
            os.path.join(
                config.DATA_PROCESSED, f"features_shap_selected_{app_name}.parquet"
            )
        )
        print(f"Saved shap_selected feature group ({len(top_features)} features) "
              f"for {app_name}")

    print(f"\nSaved SHAP tables to {config.RESULTS_TABLES}/")


if __name__ == "__main__":
    main()
