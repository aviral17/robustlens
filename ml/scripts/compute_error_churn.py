"""Compute Order-Dependent Error Churn, directional gain/loss rates, and confidence dynamics

This script implements the research extension on Order-Dependent Error Churn:
1. Evaluates all 300 condition-level pairs (4 models x 5 severities x 15 sequence pairs).
2. Calculates Gain rate (G), Loss rate (L), Net Accuracy Difference (G - L), Error Churn (G + L), and Excess Churn (2 * min(G, L)).
3. Verifies mathematical identities numerically.
4. Computes 95% Clustered Percentile Bootstrap Confidence Intervals (B = 10,000).
5. Conducts asymptotic McNemar and Monte Carlo paired sign-flip permutation tests (R = 20,000) with Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR) corrections.
6. Computes confidence dynamics on switching vs. stable images.
7. Evaluates correlation between Net Difference and Churn, identifying decoupling cases.
8. Exports CSV and JSON artifacts under results/order_effect_study/full/outputs/.
"""

import gzip
import json
import os
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_FILE = os.path.join(REPO_ROOT, "results", "order_effect_study", "full", "outputs", "all_order_predictions.csv.gz")
OUTPUT_DIR = os.path.join(REPO_ROOT, "results", "order_effect_study", "full", "outputs")

SEED = 42
B_BOOT = 10000
R_PERM = 20000

MODELS = ["convnext_tiny", "swin_tiny", "vit_base", "resnet50"]
MODEL_NAMES = {
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_tiny": "Swin-Tiny",
    "vit_base": "ViT-Base/16",
    "resnet50": "ResNet-50",
}
SEVERITIES = [1, 2, 3, 4, 5]
ORDERS = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]

ORDER_DESCRIPTIONS = {
    "NBJ": "Noise -> Blur -> JPEG",
    "NJB": "Noise -> JPEG -> Blur",
    "BNJ": "Blur -> Noise -> JPEG",
    "BJN": "Blur -> JPEG -> Noise",
    "JNB": "JPEG -> Noise -> Blur",
    "JBN": "JPEG -> Blur -> Noise",
}


def run_mcnemar_test(n01: int, n10: int):
    """Asymptotic McNemar test with Edwards' continuity correction."""
    total_discordant = n01 + n10
    if total_discordant == 0:
        return 0.0, 1.0
    stat = ((abs(n01 - n10) - 1.0) ** 2) / total_discordant
    p_val = float(stats.chi2.sf(stat, df=1))
    return float(stat), p_val


def run_sign_flip_test(d: np.ndarray, R: int = 20000, rng: np.random.Generator = None):
    """Monte Carlo paired sign-flip permutation test with pseudocount correction."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    nonzero_d = d[d != 0]
    n_nonzero = len(nonzero_d)
    if n_nonzero == 0:
        return 0.0, 1.0, 0.0
    t_obs = float(np.sum(nonzero_d))
    signs = rng.choice([-1.0, 1.0], size=(R, n_nonzero))
    perm_t = np.dot(signs, nonzero_d)
    count = np.sum(np.abs(perm_t) >= np.abs(t_obs))
    p_val = float((1.0 + count) / (R + 1.0))
    se_null = np.sqrt(n_nonzero)
    z_score = float(t_obs / se_null) if se_null > 0 else 0.0
    return t_obs, p_val, z_score


def compute_multinomial_bootstrap_cis(n00, n01, n10, n11, N, B=10000, rng=None):
    """Fast multinomial bootstrap for cell proportions on N independent images."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    p = np.array([n00, n01, n10, n11], dtype=np.float64) / N
    counts = rng.multinomial(N, p, size=B)  # shape (B, 4)
    G_b = counts[:, 1] / N
    L_b = counts[:, 2] / N
    net_b = G_b - L_b
    churn_b = G_b + L_b
    excess_b = churn_b - np.abs(net_b)

    ci_G = np.percentile(G_b, [2.5, 97.5]).tolist()
    ci_L = np.percentile(L_b, [2.5, 97.5]).tolist()
    ci_net = np.percentile(net_b, [2.5, 97.5]).tolist()
    ci_churn = np.percentile(churn_b, [2.5, 97.5]).tolist()
    ci_excess = np.percentile(excess_b, [2.5, 97.5]).tolist()

    return ci_G, ci_L, ci_net, ci_churn, ci_excess


def apply_holm_bonferroni(p_vals):
    """Holm-Bonferroni FWER step-down adjustment."""
    m = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    adj_p = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        p_adj = p_vals[idx] * (m - rank)
        running_max = max(running_max, p_adj)
        adj_p[idx] = min(running_max, 1.0)
    return adj_p


def apply_benjamini_hochberg(p_vals):
    """Benjamini-Hochberg FDR step-up adjustment."""
    m = len(p_vals)
    sorted_indices = np.argsort(p_vals)[::-1]
    adj_p = np.zeros(m, dtype=np.float64)
    running_min = 1.0
    for rank_from_top, idx in enumerate(sorted_indices):
        orig_rank = m - rank_from_top
        p_adj = p_vals[idx] * m / orig_rank
        running_min = min(running_min, p_adj)
        adj_p[idx] = min(running_min, 1.0)
    return adj_p


def main():
    print(f"Loading validated full predictions from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df):,} records. Filtering compound conditions...")
    compound = df[df["order"] != "clean"].copy()
    n_images = compound["sample_idx"].nunique()
    print(f"Total compound records: {len(compound):,}, Physical images: {n_images}")

    rng = np.random.default_rng(SEED)

    records = []
    confidence_records = []
    identity_errors = 0

    print("Computing pairwise error churn across 4 models x 5 severities x 15 order pairs = 300 comparisons...")

    for m in MODELS:
        m_name = MODEL_NAMES[m]
        model_sub = compound[compound["model"] == m]

        for s in SEVERITIES:
            sub = model_sub[model_sub["severity"] == s]
            piv_correct = sub.pivot(index="sample_idx", columns="order", values="correct")
            piv_conf = sub.pivot(index="sample_idx", columns="order", values="confidence")

            for i in range(len(ORDERS)):
                for j in range(i + 1, len(ORDERS)):
                    oA = ORDERS[i]
                    oB = ORDERS[j]

                    cA = piv_correct[oA].values
                    cB = piv_correct[oB].values
                    confA = piv_conf[oA].values
                    confB = piv_conf[oB].values

                    N = len(cA)
                    n00 = int(np.sum((cA == 0) & (cB == 0)))
                    n01 = int(np.sum((cA == 0) & (cB == 1)))  # Gain: rescued by B
                    n10 = int(np.sum((cA == 1) & (cB == 0)))  # Loss: broken by B
                    n11 = int(np.sum((cA == 1) & (cB == 1)))

                    acc_a = (n10 + n11) / N
                    acc_b = (n01 + n11) / N
                    G = n01 / N
                    L = n10 / N
                    net_diff = G - L
                    churn = G + L
                    excess_churn = churn - abs(net_diff)
                    algebraic_excess = 2.0 * min(G, L)

                    # Identity check
                    diff_acc = acc_b - acc_a
                    if abs(diff_acc - net_diff) > 1e-12:
                        identity_errors += 1
                    if abs(excess_churn - algebraic_excess) > 1e-12:
                        identity_errors += 1

                    # Bootstrap CIs (B = 10,000)
                    ci_G, ci_L, ci_net, ci_churn, ci_excess = compute_multinomial_bootstrap_cis(
                        n00, n01, n10, n11, N, B=B_BOOT, rng=rng
                    )

                    # Statistical tests
                    mcnemar_stat, mcnemar_p = run_mcnemar_test(n01, n10)
                    t_obs, sign_flip_p, z_score = run_sign_flip_test(cB - cA, R=R_PERM, rng=rng)

                    # Confidence shifts on transitions
                    mask_gain = (cA == 0) & (cB == 1)
                    mask_loss = (cA == 1) & (cB == 0)
                    mask_c11 = (cA == 1) & (cB == 1)
                    mask_c00 = (cA == 0) & (cB == 0)

                    conf_gain_A = float(np.mean(confA[mask_gain])) if n01 > 0 else float("nan")
                    conf_gain_B = float(np.mean(confB[mask_gain])) if n01 > 0 else float("nan")
                    conf_loss_A = float(np.mean(confA[mask_loss])) if n10 > 0 else float("nan")
                    conf_loss_B = float(np.mean(confB[mask_loss])) if n10 > 0 else float("nan")
                    conf_c11_A = float(np.mean(confA[mask_c11])) if n11 > 0 else float("nan")
                    conf_c11_B = float(np.mean(confB[mask_c11])) if n11 > 0 else float("nan")
                    conf_c00_A = float(np.mean(confA[mask_c00])) if n00 > 0 else float("nan")
                    conf_c00_B = float(np.mean(confB[mask_c00])) if n00 > 0 else float("nan")

                    record = {
                        "model": m,
                        "model_name": m_name,
                        "severity": s,
                        "order_a": oA,
                        "order_b": oB,
                        "pair_name": f"{oA}_vs_{oB}",
                        "desc_a": ORDER_DESCRIPTIONS[oA],
                        "desc_b": ORDER_DESCRIPTIONS[oB],
                        "n_images": N,
                        "n00_both_wrong": n00,
                        "n01_gain_rescued": n01,
                        "n10_loss_broken": n10,
                        "n11_both_correct": n11,
                        "acc_a": round(acc_a, 6),
                        "acc_b": round(acc_b, 6),
                        "gain_rate_G": round(G, 6),
                        "gain_ci_lower": round(ci_G[0], 6),
                        "gain_ci_upper": round(ci_G[1], 6),
                        "loss_rate_L": round(L, 6),
                        "loss_ci_lower": round(ci_L[0], 6),
                        "loss_ci_upper": round(ci_L[1], 6),
                        "net_accuracy_diff": round(net_diff, 6),
                        "net_ci_lower": round(ci_net[0], 6),
                        "net_ci_upper": round(ci_net[1], 6),
                        "error_churn": round(churn, 6),
                        "churn_ci_lower": round(ci_churn[0], 6),
                        "churn_ci_upper": round(ci_churn[1], 6),
                        "excess_churn": round(excess_churn, 6),
                        "excess_ci_lower": round(ci_excess[0], 6),
                        "excess_ci_upper": round(ci_excess[1], 6),
                        "mcnemar_chi2": round(mcnemar_stat, 4),
                        "mcnemar_p_value": mcnemar_p,
                        "sign_flip_t_obs": t_obs,
                        "sign_flip_z": round(z_score, 4),
                        "sign_flip_p_value": sign_flip_p,
                    }
                    records.append(record)

                    confidence_records.append({
                        "model": m,
                        "model_name": m_name,
                        "severity": s,
                        "order_a": oA,
                        "order_b": oB,
                        "pair_name": f"{oA}_vs_{oB}",
                        "n_gain": n01,
                        "conf_gain_A": round(conf_gain_A, 4),
                        "conf_gain_B": round(conf_gain_B, 4),
                        "delta_conf_gain": round(conf_gain_B - conf_gain_A, 4) if not np.isnan(conf_gain_A) else float("nan"),
                        "n_loss": n10,
                        "conf_loss_A": round(conf_loss_A, 4),
                        "conf_loss_B": round(conf_loss_B, 4),
                        "delta_conf_loss": round(conf_loss_B - conf_loss_A, 4) if not np.isnan(conf_loss_A) else float("nan"),
                        "n_stable_correct": n11,
                        "conf_c11_A": round(conf_c11_A, 4),
                        "conf_c11_B": round(conf_c11_B, 4),
                        "n_stable_wrong": n00,
                        "conf_c00_A": round(conf_c00_A, 4),
                        "conf_c00_B": round(conf_c00_B, 4),
                    })

    df_churn = pd.DataFrame(records)
    df_conf = pd.DataFrame(confidence_records)

    print(f"Processed 300 comparisons. Identity check errors: {identity_errors}")

    # Multiple testing correction within each model family (75 tests per model)
    df_churn["holm_p_value_family"] = 1.0
    df_churn["bh_p_value_family"] = 1.0

    for m in MODELS:
        m_mask = df_churn["model"] == m
        p_vals = df_churn.loc[m_mask, "sign_flip_p_value"].values
        df_churn.loc[m_mask, "holm_p_value_family"] = apply_holm_bonferroni(p_vals)
        df_churn.loc[m_mask, "bh_p_value_family"] = apply_benjamini_hochberg(p_vals)

    # Global multiplicity correction across all 300 comparisons
    df_churn["holm_p_value_global"] = apply_holm_bonferroni(df_churn["sign_flip_p_value"].values)
    df_churn["bh_p_value_global"] = apply_benjamini_hochberg(df_churn["sign_flip_p_value"].values)

    # Save summary CSV
    out_csv = os.path.join(OUTPUT_DIR, "error_churn_summary.csv")
    df_churn.to_csv(out_csv, index=False)
    print(f"Saved complete error churn summary to: {out_csv}")

    # Save confidence shifts CSV
    out_conf_csv = os.path.join(OUTPUT_DIR, "error_churn_confidence_shifts.csv")
    df_conf.to_csv(out_conf_csv, index=False)
    print(f"Saved confidence shifts to: {out_conf_csv}")

    # Identify cases with small net differences but high churn
    # Criterion 1: |Net Diff| <= 0.02 (2.0%) AND Excess Churn >= 0.05 (5.0%)
    high_excess = df_churn[(df_churn["net_accuracy_diff"].abs() <= 0.02) & (df_churn["excess_churn"] >= 0.05)].copy()
    high_excess.sort_values(by="excess_churn", ascending=False, inplace=True)
    out_high_excess = os.path.join(OUTPUT_DIR, "error_churn_high_excess_cases.csv")
    high_excess.to_csv(out_high_excess, index=False)
    print(f"Found {len(high_excess)} comparisons with |Net Diff| <= 2% and Excess Churn >= 5%. Saved to: {out_high_excess}")

    # Compute correlation between Net Diff magnitude and Churn
    abs_net = df_churn["net_accuracy_diff"].abs().values
    churn_vals = df_churn["error_churn"].values
    excess_vals = df_churn["excess_churn"].values

    pearson_r, pearson_p = stats.pearsonr(abs_net, churn_vals)
    spearman_rho, spearman_p = stats.spearmanr(abs_net, churn_vals)

    # Severity-level and Model-level churn aggregation
    model_summary = {}
    for m in MODELS:
        m_df = df_churn[df_churn["model"] == m]
        m_excess = m_df[m_df["net_accuracy_diff"].abs() <= 0.02]
        model_summary[m] = {
            "model_name": MODEL_NAMES[m],
            "mean_churn": round(float(m_df["error_churn"].mean()), 4),
            "max_churn": round(float(m_df["error_churn"].max()), 4),
            "mean_excess_churn": round(float(m_df["excess_churn"].mean()), 4),
            "max_excess_churn": round(float(m_df["excess_churn"].max()), 4),
            "n_pairs_excess_gt_5pct": int(np.sum(m_df["excess_churn"] >= 0.05)),
            "n_pairs_small_net_high_excess": int(len(m_excess[m_excess["excess_churn"] >= 0.05])),
        }

    # Summary by severity
    severity_summary = {}
    for s in SEVERITIES:
        s_df = df_churn[df_churn["severity"] == s]
        severity_summary[s] = {
            "mean_churn": round(float(s_df["error_churn"].mean()), 4),
            "max_churn": round(float(s_df["error_churn"].max()), 4),
            "mean_excess_churn": round(float(s_df["excess_churn"].mean()), 4),
            "max_excess_churn": round(float(s_df["excess_churn"].max()), 4),
        }

    # Structured JSON report
    report_payload = {
        "benchmark_meta": {
            "dataset": "FastAI Imagenette-v2 (Validation Split, N=3,925)",
            "n_images": n_images,
            "n_models": len(MODELS),
            "n_severities": len(SEVERITIES),
            "n_order_pairs": 15,
            "total_pairwise_evaluations": len(df_churn),
            "b_bootstraps": B_BOOT,
            "r_permutations": R_PERM,
        },
        "statistical_correlations": {
            "pearson_r_abs_net_vs_churn": round(float(pearson_r), 4),
            "pearson_p_value": float(pearson_p),
            "spearman_rho_abs_net_vs_churn": round(float(spearman_rho), 4),
            "spearman_p_value": float(spearman_p),
        },
        "decoupling_summary": {
            "total_pairs": len(df_churn),
            "pairs_with_net_diff_le_2pct": int(np.sum(abs_net <= 0.02)),
            "pairs_with_excess_ge_5pct": int(np.sum(excess_vals >= 0.05)),
            "pairs_small_net_and_high_excess_5pct": int(len(high_excess)),
            "pairs_exact_zero_net_diff": int(np.sum(abs_net == 0.0)),
            "max_excess_churn_observed": round(float(np.max(excess_vals)), 4),
        },
        "model_summaries": model_summary,
        "severity_summaries": severity_summary,
        "top_decoupling_cases": high_excess.head(10)[
            ["model", "severity", "order_a", "order_b", "net_accuracy_diff", "error_churn", "excess_churn", "gain_rate_G", "loss_rate_L"]
        ].to_dict(orient="records"),
    }

    out_json = os.path.join(OUTPUT_DIR, "error_churn_report.json")
    with open(out_json, "w") as f:
        json.dump(report_payload, f, indent=2)
    print(f"Saved structured error churn report to: {out_json}")

    print("\n--- Key Research Insights on Error Churn ---")
    print(f"Pearson r(|Net Diff|, Churn) = {pearson_r:.4f} (Spearman rho = {spearman_rho:.4f})")
    print(f"Total pairs where |Net Diff| <= 2% but Excess Churn >= 5%: {len(high_excess)} / {len(df_churn)}")
    print("Top 5 Decoupled Pairs (Net Difference ~ 0, High Error Churn):")
    for _, row in high_excess.head(5).iterrows():
        print(f"  {row['model_name']} (Sev {row['severity']}) - {row['order_a']} vs {row['order_b']}: Net Diff = {row['net_accuracy_diff']*100:+.2f}%, Churn = {row['error_churn']*100:.2f}%, Excess = {row['excess_churn']*100:.2f}% (G={row['gain_rate_G']*100:.2f}%, L={row['loss_rate_L']*100:.2f}%)")


if __name__ == "__main__":
    main()
