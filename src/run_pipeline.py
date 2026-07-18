"""
run_pipeline.py (v2)
----------------------
Runs the ENTIRE corrected pipeline end-to-end.

    python run_pipeline.py

Expect this to take roughly 15-25 minutes on a typical laptop CPU (no
GPU used anywhere) -- this is substantially longer than the original
v1 pipeline (~1 minute) because this version adds: multi-seed
evaluation (3 seeds), small hyperparameter grids per model, a scaffold
split (Bemis-Murcko grouping), an element-level metal-holdout split
with a programmatic no-leakage check, a Tanimoto-kNN baseline, and
probability calibration. This is the necessary cost of fixing the 20
issues raised in the co-author's review -- see README.md for the full
list and how each one was addressed.

Produces (all under results/tables/):
    precursor_degeneracy_report.csv   -- fix #1 quantification
    ablation_results_full.csv          -- fixes #2,3,4,5,8,9,10,16,17,18,19
    confidence_screen_<app>.csv        -- fixes #14,#15 (full table, calibrated)
    calibration_summary.csv            -- Brier score before/after calibration
    error_analysis_<app>.csv           -- fix #6 (now run on the rigorous split)
    shap_trend_<app>.csv               -- fixes #11,#12,#13
    compute_comparison.csv             -- fix #19
"""
import time

import data_prep
import featurize
import splits
import models
import confidence_screen
import error_analysis
import run_shap_trend_report
import build_compute_table


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    t_start = time.perf_counter()

    section("STEP 1/7: Data prep (row-aligned merge -- fix #1, #3, #7)")
    data_prep.main()

    section("STEP 2/7: Featurization")
    featurize.main()

    section("STEP 3/7: Splits sanity check (random/scaffold/metal-element -- fix #4, #5)")
    splits.main()

    section("STEP 4/7: Full model training (this is the long step -- 15-20 min)")
    models.main()

    section("STEP 5/7: Compute comparison table (fix #19)")
    build_compute_table.main()

    section("STEP 6/7: SHAP trend report (fix #11, #12, #13)")
    run_shap_trend_report.main()

    section("STEP 7/7: Confidence screening (fix #14, #15) + error analysis")
    confidence_screen.main()
    error_analysis.main()

    elapsed = time.perf_counter() - t_start
    section(f"DONE in {elapsed/60:.1f} minutes -- see results/tables/ for all output CSVs")


if __name__ == "__main__":
    main()
