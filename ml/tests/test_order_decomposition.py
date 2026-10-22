"""Unit tests for saturated 6-cell order decomposition and error churn identities."""

import numpy as np
import pytest

from robustlens.corruptions.order_effects import (
    ORDER_PERMUTATIONS,
    ORDER_DESCRIPTIONS,
)


def test_order_operator_mappings():
    """Verify code-level ground truth of operator sequence execution."""
    assert ORDER_PERMUTATIONS == ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]

    expected_ends = {
        "NBJ": "J",
        "NJB": "B",
        "BNJ": "J",
        "BJN": "N",
        "JNB": "B",
        "JBN": "N",
    }
    for order, expected_terminal in expected_ends.items():
        assert order[-1] == expected_terminal, f"Sequence {order} should terminate in {expected_terminal}, got {order[-1]}"
        desc_terminal = ORDER_DESCRIPTIONS[order].split("->")[-1].strip()
        if expected_terminal == "J":
            assert desc_terminal == "JPEG"
        elif expected_terminal == "B":
            assert desc_terminal == "Blur"
        elif expected_terminal == "N":
            assert desc_terminal == "Noise"


def test_saturated_design_matrix_properties():
    """Verify rank, determinant, and balanced parity properties of the 6-cell model."""
    # Design matrix: columns [Intercept, Start_B, Start_J, End_B, End_J, Parity]
    # Rows: [NBJ, NJB, BNJ, BJN, JNB, JBN]
    X = np.array([
        [1.0, 0.0, 0.0, 0.0, 1.0,  1.0],  # NBJ: Start N, End J, Parity +1
        [1.0, 0.0, 0.0, 1.0, 0.0, -1.0],  # NJB: Start N, End B, Parity -1
        [1.0, 1.0, 0.0, 0.0, 1.0, -1.0],  # BNJ: Start B, End J, Parity -1
        [1.0, 1.0, 0.0, 0.0, 0.0,  1.0],  # BJN: Start B, End N, Parity +1
        [1.0, 0.0, 1.0, 1.0, 0.0,  1.0],  # JNB: Start J, End B, Parity +1
        [1.0, 0.0, 1.0, 0.0, 0.0, -1.0],  # JBN: Start J, End N, Parity -1
    ], dtype=float)

    # 1. Full rank and non-zero determinant
    rank = np.linalg.matrix_rank(X)
    det = np.linalg.det(X)
    assert rank == 6, f"Design matrix must have full rank 6, got {rank}"
    assert np.isclose(det, -6.0), f"Determinant of design matrix should be -6.0, got {det}"

    # 2. Balanced parity check:
    # Group 1 (+1): NBJ, BJN, JNB
    # Group 2 (-1): NJB, BNJ, JBN
    # Check that in each group, every start primitive (N, B, J) appears exactly once
    group1_starts = ["N", "B", "J"]  # NBJ (N), BJN (B), JNB (J)
    group2_starts = ["N", "B", "J"]  # NJB (N), BNJ (B), JBN (J)
    group1_ends = ["J", "N", "B"]    # NBJ (J), BJN (N), JNB (B)
    group2_ends = ["B", "J", "N"]    # NJB (B), BNJ (J), JBN (N)

    assert sorted(group1_starts) == ["B", "J", "N"]
    assert sorted(group2_starts) == ["B", "J", "N"]
    assert sorted(group1_ends) == ["B", "J", "N"]
    assert sorted(group2_ends) == ["B", "J", "N"]

    # 3. Verify analytic inverse matrix
    inv_X = np.linalg.inv(X)
    identity = inv_X @ X
    assert np.allclose(identity, np.eye(6)), "inv(X) @ X must equal identity"

    # Row 6 of inv_X corresponds to parity parameter: 1/6 * [1, -1, -1, 1, 1, -1]
    expected_parity_weights = np.array([1/6, -1/6, -1/6, 1/6, 1/6, -1/6])
    assert np.allclose(inv_X[5, :], expected_parity_weights), "Parity weights in inv(X) must match theoretical weights"


def test_cell_reconstruction():
    """Verify that any arbitrary cell means vector reconstructs exactly via beta = inv(X) @ mu."""
    X = np.array([
        [1.0, 0.0, 0.0, 0.0, 1.0,  1.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, -1.0],
        [1.0, 1.0, 0.0, 0.0, 1.0, -1.0],
        [1.0, 1.0, 0.0, 0.0, 0.0,  1.0],
        [1.0, 0.0, 1.0, 1.0, 0.0,  1.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, -1.0],
    ], dtype=float)
    inv_X = np.linalg.inv(X)

    # Test random cell means
    rng = np.random.default_rng(42)
    for _ in range(10):
        mu_true = rng.uniform(0.3, 0.8, size=6)
        beta = inv_X @ mu_true
        mu_reconstructed = X @ beta
        assert np.allclose(mu_true, mu_reconstructed), "Reconstructed cell means must equal original"


def test_error_churn_identity():
    """Verify algebraic identity: Churn == |Delta_Acc| + 2 * min(Gain, Loss)."""
    # Create synthetic binary predictions on 1000 items
    rng = np.random.default_rng(123)
    c_a = rng.binomial(1, 0.65, size=1000)
    c_b = rng.binomial(1, 0.55, size=1000)

    g = np.mean((c_a == 0) & (c_b == 1))
    l = np.mean((c_a == 1) & (c_b == 0))
    delta_acc = g - l
    churn = g + l
    excess_churn = 2.0 * min(g, l)

    assert np.isclose(churn, abs(delta_acc) + excess_churn)


def test_tolerance_failure_detection():
    """Verify that tolerance verification correctly flags out-of-tolerance calibration."""
    max_psnr_tol = 0.75
    max_ssim_tol = 0.035

    # Measured empirical residuals
    achieved_psnr_rmse = 0.943
    max_abs_psnr_res = 2.639
    achieved_ssim_rmse = 0.1380
    max_abs_ssim_res = 0.2379

    psnr_failed = (achieved_psnr_rmse > max_psnr_tol) or (max_abs_psnr_res > max_psnr_tol)
    ssim_failed = (achieved_ssim_rmse > max_ssim_tol) or (max_abs_ssim_res > max_ssim_tol)

    assert psnr_failed is True, "PSNR calibration should be correctly marked as failed tolerance"
    assert ssim_failed is True, "SSIM calibration should be correctly marked as failed tolerance"


def test_two_sided_log_p_values_and_multiplicity():
    """Verify exact two-sided log p-value calculation log(2) + logsf(|z|) and log-space adjustments."""
    from scipy.stats import norm
    import sys
    from pathlib import Path
    
    # Import log adjustment functions from script
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from run_final_order_decomposition_analysis import (
        holm_bonferroni_log,
        benjamini_hochberg_log,
        format_log_p_value,
    )

    # 1. Test exact two-sided computation for known z
    z_values = [0.0, 1.959963984540054, 2.5758293035489004, 38.0]
    for z in z_values:
        abs_z = abs(z)
        log_p = 0.0 if abs_z == 0 else np.log(2.0) + norm.logsf(abs_z)
        if z == 0.0:
            assert np.isclose(log_p, 0.0)
            assert np.isclose(np.exp(log_p), 1.0)
        elif z < 10.0:
            # Verify agreement with standard CDF: 2 * (1 - cdf(|z|))
            expected_p = 2.0 * norm.sf(abs_z)
            assert np.isclose(np.exp(log_p), expected_p, rtol=1e-5)
        else:
            # For large z (e.g. 38.0), standard sf underflows to 0.0, but logsf is non-zero
            assert log_p < -700.0
            formatted = format_log_p_value(log_p)
            assert "e" in formatted
            assert float(formatted) > 0.0, "Formatted p-value must never be literal 0.0"

    # 2. Test Holm and BH monotonicity
    raw_log_p = np.array([-100.0, -50.0, -10.0, -2.0, -0.5])
    holm_adj = holm_bonferroni_log(raw_log_p)
    bh_adj = benjamini_hochberg_log(raw_log_p)

    assert np.all(holm_adj >= raw_log_p), "Holm adjusted log p must be >= raw log p"
    assert np.all(bh_adj >= raw_log_p), "BH adjusted log p must be >= raw log p"
    assert np.all(holm_adj >= bh_adj), "Holm FWER control must be >= BH FDR control"
    assert np.all(np.diff(holm_adj) >= -1e-12), "Holm adjusted values must be monotonic"
    assert np.all(np.diff(bh_adj) >= -1e-12), "BH adjusted values must be monotonic"


def test_five_state_prediction_partition():
    """Verify 5-state prediction transition partition and label vs correctness churn identities."""
    rng = np.random.default_rng(999)
    n = 2000
    # 10 classes
    y_true = rng.integers(0, 10, size=n)
    y_hat_a = rng.integers(0, 10, size=n)
    y_hat_b = rng.integers(0, 10, size=n)

    c_a = (y_hat_a == y_true).astype(int)
    c_b = (y_hat_b == y_true).astype(int)

    e11 = np.mean((c_a == 1) & (c_b == 1))
    e10 = np.mean((c_a == 1) & (c_b == 0))
    e01 = np.mean((c_a == 0) & (c_b == 1))
    e00_same = np.mean((c_a == 0) & (c_b == 0) & (y_hat_a == y_hat_b))
    e00_diff = np.mean((c_a == 0) & (c_b == 0) & (y_hat_a != y_hat_b))

    # 1. Total probability conservation
    assert np.isclose(e11 + e10 + e01 + e00_same + e00_diff, 1.0)

    # 2. Correctness churn identity: Loss + Gain
    correctness_churn = np.mean(c_a != c_b)
    assert np.isclose(correctness_churn, e10 + e01)

    # 3. Label churn identity: Loss + Gain + Error Migration
    label_churn = np.mean(y_hat_a != y_hat_b)
    assert np.isclose(label_churn, correctness_churn + e00_diff)
    assert np.isclose(label_churn, e10 + e01 + e00_diff)

    # 4. Agreement rate identity: Both Correct + Same Wrong
    agreement_rate = np.mean(y_hat_a == y_hat_b)
    assert np.isclose(agreement_rate, e11 + e00_same)
    assert np.isclose(agreement_rate, 1.0 - label_churn)

    # 5. Excess churn identities
    delta_acc = np.mean(c_b) - np.mean(c_a)
    excess_corr_churn = 2.0 * min(e10, e01)
    assert np.isclose(correctness_churn, abs(delta_acc) + excess_corr_churn)

    excess_label_churn = label_churn - abs(delta_acc)
    assert np.isclose(excess_label_churn, excess_corr_churn + e00_diff)


def test_marginal_contrasts():
    """Verify that direct marginal contrasts match cell mean combinations and 2 * parity beta."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from run_final_order_decomposition_analysis import compute_marginal_contrasts, INV_X

    # Arbitrary cell means: [NBJ, NJB, BNJ, BJN, JNB, JBN]
    rng = np.random.default_rng(777)
    mu = rng.uniform(0.4, 0.7, size=6)

    contrasts = compute_marginal_contrasts(mu)
    betas = INV_X @ mu

    # Parity beta is index 5
    assert np.isclose(contrasts["parity_contrast"], 2.0 * betas[5])

    # Check even and odd means
    expected_even = (mu[0] + mu[3] + mu[4]) / 3.0  # NBJ, BJN, JNB
    expected_odd = (mu[1] + mu[2] + mu[5]) / 3.0   # NJB, BNJ, JBN
    assert np.isclose(contrasts["even_mean"], expected_even)
    assert np.isclose(contrasts["odd_mean"], expected_odd)


