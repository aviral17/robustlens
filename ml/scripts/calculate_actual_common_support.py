"""Calculate empirical 6-way joint PSNR-SSIM common support and all 15 pairwise overlaps.

Pure NumPy/SciPy implementation (no external GIS dependencies):
1. 2D Bounding Boxes and 2D Bin Occupancy for each sequence in (PSNR, SSIM) space.
2. 6-Way Joint Empirical Overlap: S_6way = Intersection(S_1 .. S_6).
3. All 15 Pairwise Overlaps: S_ij = S_i Intersection S_j.
4. Overlap Area Coverage (Absolute area in dB x SSIM, and relative coverage ratio).
5. Closest attainable grid points for unachievable coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

BASE_DIR = Path(__file__).resolve().parent.parent.parent
GRID_CSV = BASE_DIR / "results" / "matched_fidelity_study" / "common_support_grid_calibration.csv"
OUTPUT_JSON = BASE_DIR / "results" / "matched_fidelity_study" / "empirical_common_support_overlaps.json"

SEQUENCES = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]
FINAL_OPS = {
    "NBJ": "JPEG",
    "BNJ": "JPEG",
    "NJB": "Blur",
    "JNB": "Blur",
    "BJN": "Noise",
    "JBN": "Noise",
}


def build_occupancy_grid(df_seq: pd.DataFrame, p_bins: np.ndarray, s_bins: np.ndarray) -> np.ndarray:
    """Builds a 2D boolean mask indicating empirical achievable bins via 2D Delaunay triangulation."""
    pts = df_seq[["mean_psnr", "mean_ssim"]].drop_duplicates().to_numpy()
    tri = Delaunay(pts)

    # Meshgrid of bin centers
    p_centers = 0.5 * (p_bins[:-1] + p_bins[1:])
    s_centers = 0.5 * (s_bins[:-1] + s_bins[1:])
    P, S = np.meshgrid(p_centers, s_centers)
    query_pts = np.column_stack([P.ravel(), S.ravel()])

    # Find points inside Delaunay simplex (convex hull interior)
    inside = (tri.find_simplex(query_pts) >= 0)
    return inside.reshape(P.shape)


def main():
    print("=" * 80)
    print("CALCULATING EMPIRICAL JOINT COMMON SUPPORT & 15 PAIRWISE OVERLAPS")
    print("=" * 80)

    df = pd.read_csv(GRID_CSV)
    print(f"Loaded {len(df)} grid records across {df['order'].nunique()} sequences.")

    p_min_all = 10.0
    p_max_all = 36.0
    s_min_all = 0.0
    s_max_all = 1.0

    n_p_bins = 130  # 0.20 dB per bin
    n_s_bins = 100  # 0.01 SSIM per bin

    p_bins = np.linspace(p_min_all, p_max_all, n_p_bins + 1)
    s_bins = np.linspace(s_min_all, s_max_all, n_s_bins + 1)
    bin_area = (p_bins[1] - p_bins[0]) * (s_bins[1] - s_bins[0])  # dB x SSIM units

    # 1. Sequence-specific occupancy masks & bounds
    seq_masks = {}
    seq_bounds = {}
    for seq in SEQUENCES:
        sub = df[df["order"] == seq]
        mask = build_occupancy_grid(sub, p_bins, s_bins)
        seq_masks[seq] = mask

        occupied_count = int(mask.sum())
        hull_area = occupied_count * bin_area

        seq_bounds[seq] = {
            "final_op": FINAL_OPS[seq],
            "psnr_min": round(float(sub["mean_psnr"].min()), 3),
            "psnr_max": round(float(sub["mean_psnr"].max()), 3),
            "psnr_range": round(float(sub["mean_psnr"].max() - sub["mean_psnr"].min()), 3),
            "ssim_min": round(float(sub["mean_ssim"].min()), 4),
            "ssim_max": round(float(sub["mean_ssim"].max()), 4),
            "ssim_range": round(float(sub["mean_ssim"].max() - sub["mean_ssim"].min()), 4),
            "achievable_area_units": round(float(hull_area), 4),
        }

    # Total Union Mask across all 6 sequences
    union_mask = np.zeros_like(seq_masks[SEQUENCES[0]])
    for seq in SEQUENCES:
        union_mask = np.logical_or(union_mask, seq_masks[seq])
    total_union_area = float(union_mask.sum() * bin_area)

    # 2. Six-Way Joint Empirical Overlap
    joint_6way_mask = np.ones_like(seq_masks[SEQUENCES[0]])
    for seq in SEQUENCES:
        joint_6way_mask = np.logical_and(joint_6way_mask, seq_masks[seq])

    joint_6way_area = float(joint_6way_mask.sum() * bin_area)
    joint_6way_coverage = float(joint_6way_area / (total_union_area + 1e-12))

    p_centers = 0.5 * (p_bins[:-1] + p_bins[1:])
    s_centers = 0.5 * (s_bins[:-1] + s_bins[1:])
    P, S = np.meshgrid(p_centers, s_centers)

    joint_6way_bounds = {}
    if joint_6way_mask.any():
        p_occ = P[joint_6way_mask]
        s_occ = S[joint_6way_mask]
        joint_6way_bounds = {
            "psnr_overlap_min": round(float(p_occ.min()), 3),
            "psnr_overlap_max": round(float(p_occ.max()), 3),
            "ssim_overlap_min": round(float(s_occ.min()), 4),
            "ssim_overlap_max": round(float(s_occ.max()), 4),
        }

    # 3. All 15 Pairwise Overlaps
    pairwise_results = {}
    for i in range(len(SEQUENCES)):
        for j in range(i + 1, len(SEQUENCES)):
            s1, s2 = SEQUENCES[i], SEQUENCES[j]
            pair_key = f"{s1}_vs_{s2}"
            inter_mask = np.logical_and(seq_masks[s1], seq_masks[s2])
            union_pair_mask = np.logical_or(seq_masks[s1], seq_masks[s2])

            inter_area = float(inter_mask.sum() * bin_area)
            union_pair_area = float(union_pair_mask.sum() * bin_area)
            jaccard = float(inter_area / (union_pair_area + 1e-12))

            bounds = {}
            if inter_mask.any():
                bounds = {
                    "psnr_min": round(float(P[inter_mask].min()), 3),
                    "psnr_max": round(float(P[inter_mask].max()), 3),
                    "ssim_min": round(float(S[inter_mask].min()), 4),
                    "ssim_max": round(float(S[inter_mask].max()), 4),
                }

            pairwise_results[pair_key] = {
                "sequence_1": s1,
                "final_op_1": FINAL_OPS[s1],
                "sequence_2": s2,
                "final_op_2": FINAL_OPS[s2],
                "same_final_op": bool(FINAL_OPS[s1] == FINAL_OPS[s2]),
                "overlap_area": round(inter_area, 4),
                "jaccard_overlap_index": round(jaccard, 4),
                "overlap_bounds": bounds,
            }

    # 4. Closest Attainable Points for Historical Targets
    historical_targets = {
        1: (28.0, 0.850, "Target 1 (Mild)"),
        2: (24.5, 0.740, "Target 2 (Mod-Mild)"),
        3: (21.5, 0.620, "Target 3 (Moderate)"),
        4: (18.5, 0.480, "Target 4 (Mod-Severe)"),
        5: (16.0, 0.360, "Target 5 (Severe)"),
    }

    closest_points_analysis = {}
    for t_idx, (t_p, t_s, t_name) in historical_targets.items():
        seq_attain = {}
        for seq in SEQUENCES:
            sub = df[df["order"] == seq].copy()
            norm_dist = np.sqrt(((sub["mean_psnr"] - t_p) / 0.75)**2 + ((sub["mean_ssim"] - t_s) / 0.035)**2)
            closest_idx = norm_dist.idxmin()
            closest_row = sub.loc[closest_idx]

            p_err = closest_row["mean_psnr"] - t_p
            s_err = closest_row["mean_ssim"] - t_s
            within_tol = bool(abs(p_err) <= 0.75 and abs(s_err) <= 0.035)

            seq_attain[seq] = {
                "final_op": FINAL_OPS[seq],
                "noise_std": float(closest_row["noise_std"]),
                "blur_sigma": float(closest_row["blur_sigma"]),
                "jpeg_quality": int(closest_row["jpeg_quality"]),
                "achieved_psnr": round(float(closest_row["mean_psnr"]), 3),
                "psnr_error": round(float(p_err), 3),
                "achieved_ssim": round(float(closest_row["mean_ssim"]), 4),
                "ssim_error": round(float(s_err), 4),
                "normalized_distance": round(float(norm_dist.min()), 3),
                "within_prespecified_tolerance": within_tol,
            }
        closest_points_analysis[f"target_{t_idx}"] = {
            "target_name": t_name,
            "target_psnr": t_p,
            "target_ssim": t_s,
            "attainable_sequences_count": sum(1 for v in seq_attain.values() if v["within_prespecified_tolerance"]),
            "sequences": seq_attain,
        }

    summary = {
        "analysis": "Empirical Grid-Sampled Achievable Region & Overlap Analysis",
        "total_union_area": round(total_union_area, 4),
        "joint_6way_overlap_area": round(joint_6way_area, 4),
        "joint_6way_coverage_fraction": round(joint_6way_coverage, 4),
        "joint_6way_bounds": joint_6way_bounds,
        "sequence_achievable_regions": seq_bounds,
        "pairwise_overlaps_15": pairwise_results,
        "closest_points_for_targets": closest_points_analysis,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal Union Area: {total_union_area:.2f} (dB x SSIM)")
    print(f"6-Way Joint Overlap Area: {joint_6way_area:.2f} (Coverage: {joint_6way_coverage*100:.1f}%)")
    if joint_6way_bounds:
        print(f"6-Way Common Support Bounds: PSNR [{joint_6way_bounds['psnr_overlap_min']}, {joint_6way_bounds['psnr_overlap_max']}] dB, SSIM [{joint_6way_bounds['ssim_overlap_min']}, {joint_6way_bounds['ssim_overlap_max']}]")


if __name__ == "__main__":
    main()
