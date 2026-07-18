"""
build_compute_table.py
------------------------
Fix for #19: extracts a proper low-compute comparison (train time,
per-sample inference time, peak training memory, serialized model
size) from the already-computed ablation_results_full.csv, for the
scaffold split / precursor_descriptor feature group (the primary
evaluation condition), averaged across applications.
"""
import os
import pandas as pd
import config


def main():
    df = pd.read_csv(os.path.join(config.RESULTS_TABLES, "ablation_results_full.csv"))
    subset = df[(df.split_type == "scaffold") & (df.feature_group == "precursor_descriptor")]

    rows = []
    for model_name, g in subset.groupby("model"):
        rows.append({
            "model": model_name,
            "train_time_sec_mean": g["train_time_sec_mean"].mean(),
            "inference_time_sec_per_sample_mean": g["inference_time_sec_per_sample_mean"].mean(),
            "peak_train_memory_kb_mean": g["peak_train_memory_kb_mean"].mean(),
            "model_size_kb_mean": g["model_size_kb_mean"].mean(),
            "gpu_needed": "no",
        })
    out = pd.DataFrame(rows).sort_values("train_time_sec_mean")
    out.to_csv(os.path.join(config.RESULTS_TABLES, "compute_comparison.csv"), index=False)
    print(out.to_string(index=False))
    print(f"\nSaved -> {config.RESULTS_TABLES}/compute_comparison.csv")


if __name__ == "__main__":
    main()
