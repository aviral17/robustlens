"""Unit tests for compound corruption order effects statistical engine."""

import numpy as np
import pandas as pd
import pytest

from robustlens.stats.order_analysis import (
    benjamini_hochberg_correction,
    clustered_paired_bootstrap_ci,
    compute_ece,
    compute_friedman_omnibus,
    distortion_controlled_ancova,
    holm_bonferroni_correction,
    paired_sign_flip_test,
)


def test_friedman_omnibus_identical_and_distinct():
    # 50 images, 6 orders identical
    mat_identical = np.ones((50, 6))
    res_ident = compute_friedman_omnibus(mat_identical)
    assert res_ident["chi2_stat"] == 0.0
    assert res_ident["p_value"] == 1.0
    assert res_ident["kendall_w"] == 0.0

    # Systematic difference across orders
    rng = np.random.default_rng(42)
    mat_distinct = np.zeros((100, 6))
    mat_distinct[:, 0] = rng.binomial(1, 0.9, size=100)
    mat_distinct[:, 1] = rng.binomial(1, 0.8, size=100)
    mat_distinct[:, 2] = rng.binomial(1, 0.7, size=100)
    mat_distinct[:, 3] = rng.binomial(1, 0.6, size=100)
    mat_distinct[:, 4] = rng.binomial(1, 0.5, size=100)
    mat_distinct[:, 5] = rng.binomial(1, 0.4, size=100)
    
    res_distinct = compute_friedman_omnibus(mat_distinct)
    assert res_distinct["chi2_stat"] > 50.0
    assert res_distinct["p_value"] < 1e-6
    assert res_distinct["kendall_w"] > 0.05


def test_paired_sign_flip_test():
    # Exactly zero difference
    diff_zero = np.zeros(50)
    assert paired_sign_flip_test(diff_zero) == 1.0

    # Massive shift: 40 positive differences, 0 negative differences
    diff_massive = np.array([1.0] * 40 + [0.0] * 10)
    p_massive = paired_sign_flip_test(diff_massive, n_permutations=5000, seed=123)
    assert p_massive < 0.001

    # Exact branch (<= 16 discordants)
    diff_small = np.array([1.0] * 10 + [-1.0] * 2 + [0.0] * 20)
    p_small = paired_sign_flip_test(diff_small)
    assert 0.0 < p_small < 0.1


def test_clustered_paired_bootstrap_ci():
    rng = np.random.default_rng(99)
    y_b = rng.binomial(1, 0.5, size=200).astype(float)
    y_a = np.clip(y_b + rng.binomial(1, 0.2, size=200), 0, 1)

    mean_diff, ci_l, ci_u = clustered_paired_bootstrap_ci(y_a, y_b, n_resamples=2000, seed=42)
    assert mean_diff > 0.0
    assert ci_l < mean_diff < ci_u


def test_corrections_monotonicity():
    pvals = [0.001, 0.02, 0.04, 0.15, 0.80]
    holm = holm_bonferroni_correction(pvals)
    bh = benjamini_hochberg_correction(pvals)

    assert len(holm) == len(pvals)
    assert len(bh) == len(pvals)
    # Holm is stricter than or equal to BH
    for h, b in zip(holm, bh):
        assert h >= b - 1e-9
        assert 0.0 <= h <= 1.0
        assert 0.0 <= b <= 1.0


def test_distortion_controlled_ancova():
    rng = np.random.default_rng(42)
    n = 300
    orders = rng.choice(["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"], size=n)
    psnr = rng.normal(25.0, 5.0, size=n)
    ssim = rng.uniform(0.3, 0.9, size=n)
    
    # Correct probability driven by PSNR + order effect
    latent = -3.0 + 0.1 * psnr + 1.5 * ssim + (orders == "JBN") * 1.5
    prob = 1.0 / (1.0 + np.exp(-latent))
    correct = rng.binomial(1, prob)

    df = pd.DataFrame({
        "order": orders,
        "psnr": psnr,
        "ssim": ssim,
        "correct": correct,
    })

    res = distortion_controlled_ancova(df, reference_order="NBJ")
    assert res["status"] == "success"
    assert "psnr" in res["coefficients"]
    assert "ssim" in res["coefficients"]
    assert res["order_joint_wald_stat"] > 0
    assert 0.0 <= res["order_joint_p_value"] <= 1.0


def test_compute_ece():
    # Perfectly calibrated toy
    conf = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1])
    correct = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    ece = compute_ece(conf, correct, n_bins=5)
    assert ece <= 0.1 + 1e-6


def test_error_churn_mathematical_identity():
    """Verify algebraic identities for Gain, Loss, Net Diff, Churn, and Excess."""
    rng = np.random.default_rng(2026)
    n = 1000
    c_a = rng.binomial(1, 0.75, size=n)
    c_b = rng.binomial(1, 0.73, size=n)

    n00 = int(np.sum((c_a == 0) & (c_b == 0)))
    n01 = int(np.sum((c_a == 0) & (c_b == 1)))
    n10 = int(np.sum((c_a == 1) & (c_b == 0)))
    n11 = int(np.sum((c_a == 1) & (c_b == 1)))

    assert n00 + n01 + n10 + n11 == n

    gain = n01 / n
    loss = n10 / n
    acc_a = np.mean(c_a)
    acc_b = np.mean(c_b)
    net_diff = acc_b - acc_a
    churn = np.mean(c_a != c_b)
    excess = churn - abs(net_diff)

    # Fundamental identities
    assert np.isclose(net_diff, gain - loss, atol=1e-14)
    assert np.isclose(churn, gain + loss, atol=1e-14)
    assert np.isclose(excess, 2 * min(gain, loss), atol=1e-14)
    assert excess >= 0.0


def test_error_churn_exact_zero_difference_high_churn():
    """Verify decoupling when net difference is exactly zero but churn is high."""
    # Synthetic: 200 rescued, 200 broken, 2600 both correct, 1000 both wrong (N=4000)
    c_a = np.array([0]*1200 + [1]*2800)
    # Order B: rescues 200 from the 1200 wrong, and breaks 200 from the 2800 correct
    c_b = np.array([0]*1000 + [1]*200 + [0]*200 + [1]*2600)

    acc_a = np.mean(c_a)
    acc_b = np.mean(c_b)
    assert acc_a == acc_b == 0.70

    n01 = np.sum((c_a == 0) & (c_b == 1))
    n10 = np.sum((c_a == 1) & (c_b == 0))
    assert n01 == 200
    assert n10 == 200

    gain = n01 / 4000.0
    loss = n10 / 4000.0
    net_diff = acc_b - acc_a
    churn = gain + loss
    excess = churn - abs(net_diff)

    assert net_diff == 0.0
    assert gain == 0.05
    assert loss == 0.05
    assert churn == 0.10
    assert excess == 0.10


def test_error_churn_full_artifact_integrity():
    """Verify that full evaluated error churn artifact satisfies all theoretical invariants."""
    import os
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    summary_path = os.path.join(repo_root, "results", "order_effect_study", "full", "outputs", "error_churn_summary.csv")
    if not os.path.exists(summary_path):
        pytest.skip("Full error churn artifact not found (running in isolated environment)")

    df = pd.read_csv(summary_path)
    assert len(df) == 300  # 4 models * 5 severities * 15 sequence pairs

    # Verify per-row consistency
    for _, r in df.iterrows():
        assert r["n_images"] == 3925
        assert r["n00_both_wrong"] + r["n01_gain_rescued"] + r["n10_loss_broken"] + r["n11_both_correct"] == 3925
        assert np.isclose(r["net_accuracy_diff"], r["gain_rate_G"] - r["loss_rate_L"], atol=1e-5)
        assert np.isclose(r["error_churn"], r["gain_rate_G"] + r["loss_rate_L"], atol=1e-5)
        assert np.isclose(r["excess_churn"], 2 * min(r["gain_rate_G"], r["loss_rate_L"]), atol=1e-5)
        assert r["excess_churn"] >= -1e-9

    # Verify decoupling count: 39 pairs with |net_diff| <= 2% and excess >= 5%
    decoupled = df[(df["net_accuracy_diff"].abs() <= 0.02) & (df["excess_churn"] >= 0.05)]
    assert len(decoupled) == 39

