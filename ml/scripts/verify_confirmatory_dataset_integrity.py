"""Rigorous integrity audit of downloaded confirmatory dataset and execution manifest."""

import gzip
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONF_DIR = BASE_DIR / "results" / "matched_fidelity_study" / "confirmatory_run"
CSV_GZ = CONF_DIR / "full_distortion_fingerprints_and_margins.csv.gz"
PILOT_JSON = BASE_DIR / "results" / "matched_fidelity_study" / "pilot_exploratory_100_image_ids.json"


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_integrity():
    print("=" * 80)
    print("CONFIRMATORY DATASET INTEGRITY & PROVENANCE AUDIT")
    print("=" * 80)

    # 1. File existence & checksums
    artifacts = list(CONF_DIR.iterdir())
    checksums = {}
    print("\nDownloaded Artifacts:")
    for a in sorted(artifacts):
        csum = sha256_file(a)
        checksums[a.name] = {"bytes": a.stat().st_size, "sha256": csum}
        print(f"  {a.name:50s} {a.stat().st_size:>10,} bytes  SHA256: {csum}")

    # 2. Load dataset
    print(f"\nLoading {CSV_GZ.name}...")
    df = pd.read_csv(CSV_GZ)
    print(f"Total rows: {len(df):,}")

    # Expected values
    N_IMAGES = 3825
    N_CONDITIONS = 18
    N_MODELS = 4
    EXPECTED_TOTAL_ROWS = N_IMAGES * N_CONDITIONS * N_MODELS  # 275,400

    # Checks
    assert len(df) == EXPECTED_TOTAL_ROWS, f"Row mismatch: {len(df)} vs {EXPECTED_TOTAL_ROWS}"
    assert df["image_id"].nunique() == N_IMAGES, f"Unique images mismatch: {df['image_id'].nunique()} vs {N_IMAGES}"
    assert df["model"].nunique() == N_MODELS, f"Model count mismatch: {df['model'].nunique()} vs {N_MODELS}"
    assert set(df["model"].unique()) == {"resnet50", "convnext_tiny", "vit_base", "swin_tiny"}
    assert df["level"].nunique() == 3, f"Level count mismatch: {df['level'].nunique()} vs 3"
    assert set(df["level"].unique()) == {"Level_1_Mild", "Level_2_Moderate", "Level_3_SevereCommon"}
    assert df["order"].nunique() == 6, f"Order count mismatch: {df['order'].nunique()} vs 6"
    assert set(df["order"].unique()) == {"NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"}

    # Final operator mapping check
    expected_final_ops = {
        "NBJ": "JPEG", "BNJ": "JPEG",
        "NJB": "Blur", "JNB": "Blur",
        "BJN": "Noise", "JBN": "Noise",
    }
    for seq, expected_fop in expected_final_ops.items():
        sub = df[df["order"] == seq]
        actual_fops = sub["final_op"].unique()
        assert len(actual_fops) == 1 and actual_fops[0] == expected_fop, f"Final op mismatch for {seq}: {actual_fops}"

    # Missing values check
    missing_counts = df.isnull().sum()
    print(f"\nMissing value count across all columns: {missing_counts.sum()}")
    assert missing_counts.sum() == 0, f"Detected missing values: {missing_counts[missing_counts > 0]}"

    # Finite numeric checks
    num_cols = ["psnr", "ssim", "lpips", "fft_energy_low", "fft_energy_high", "jpeg_block_artifact", "delta_edge_energy", "std_local_ssim", "logit_margin", "delta_logit_margin", "confidence"]
    for col in num_cols:
        vals = df[col].to_numpy()
        assert np.all(np.isfinite(vals)), f"Non-finite values found in {col}"
        print(f"  Column '{col:22s}': min={vals.min():.4f}, mean={vals.mean():.4f}, max={vals.max():.4f} [ALL FINITE]")

    # Check duplicates
    dups = df.duplicated(subset=["image_id", "level", "order", "model"]).sum()
    assert dups == 0, f"Detected {dups} duplicate records!"
    print(f"Duplicate image-level-order-model tuples: {dups}")

    # Pilot leakage check
    with open(PILOT_JSON, "r") as fp:
        pilot_data = json.load(fp)
    pilot_row_indices = set(pilot_data["pilot_dataset_row_indices"])
    df_dataset_rows = set(df["dataset_row"].unique())
    leakage = df_dataset_rows.intersection(pilot_row_indices)
    assert len(leakage) == 0, f"PILOT LEAKAGE: {len(leakage)} pilot rows found in confirmatory dataset!"
    print(f"Exploratory pilot image leakage check: 0 / {len(pilot_row_indices)} pilot images present in confirmatory dataset [STRICT QUARANTINE VERIFIED].")

    # Image condition balance
    counts_per_image = df.groupby("image_id").size()
    assert (counts_per_image == (N_CONDITIONS * N_MODELS)).all(), "Imbalanced conditions per image detected"
    print(f"Condition balance per image: exactly {N_CONDITIONS * N_MODELS} records per image for all {N_IMAGES} images.")

    audit_summary = {
        "status": "PASS",
        "total_records": len(df),
        "unique_images": N_IMAGES,
        "conditions_per_image": N_CONDITIONS,
        "models_evaluated": sorted(list(df["model"].unique())),
        "missing_values": int(missing_counts.sum()),
        "duplicates": int(dups),
        "pilot_leakage_count": len(leakage),
        "checksums": checksums,
    }

    out_json = CONF_DIR / "confirmatory_dataset_integrity_audit.json"
    with open(out_json, "w") as fp:
        json.dump(audit_summary, fp, indent=2)
    print(f"\nSaved integrity report: {out_json}")
    return True


if __name__ == "__main__":
    verify_integrity()
