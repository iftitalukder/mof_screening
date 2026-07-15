"""
run_pipeline.py
----------------
Runs the entire pipeline end-to-end, in order. This is the ONLY script
you need to run for a full result set.

    python run_pipeline.py

Produces (all under results/tables/):
    ablation_results.csv          - feature-group x model x application (#2)
    metal_holdout_results.csv     - leakage-aware generalization check (#3/#4)
    compute_comparison.csv        - low-compute comparison table (#7)
    shap_trend_<app>.csv          - global SHAP trend table (#5)
    confidence_screen_<app>.csv   - confidence-based screening table (#6)
    error_analysis_<app>.csv      - false positive/negative analysis (#8)

And under data/processed/: master_table.csv + all four feature-group
parquet files (including the shap_selected ones, built after SHAP runs).
"""
import time

import data_prep
import featurize
import splits
import models
import shap_analysis
import confidence_screen
import error_analysis


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    t_start = time.perf_counter()

    section("STEP 1/6: Data prep (merge CSVs, build application labels)")
    data_prep.main()

    section("STEP 2/6: Featurization (precursor-only, descriptor-only, combined)")
    featurize.main()

    section("STEP 3/6: Splits sanity check (random + metal-holdout)")
    splits.main()

    section("STEP 4/6: Model training (ablation + metal-holdout eval + compute table)")
    models.main()

    section("STEP 5/6: SHAP analysis (trend table + shap_selected feature group)")
    shap_analysis.main()

    section("STEP 6/6: Confidence screening + error analysis")
    confidence_screen.main()
    error_analysis.main()

    elapsed = time.perf_counter() - t_start
    section(f"DONE in {elapsed:.1f}s -- see results/tables/ for all output CSVs")


if __name__ == "__main__":
    main()
