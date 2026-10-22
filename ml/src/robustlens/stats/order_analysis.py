"""Statistical analysis engine for compound corruption order effects.

Provides:
- Repeated-measures Friedman omnibus tests across 6 permutation orders.
- Image-level paired sign-flip permutation tests (R=20,000) for pairwise order comparisons.
- Clustered paired bootstrap confidence intervals (B=10,000) for order differences.
- Distortion-controlled regression (ANCOVA) testing whether order effects persist
  when conditioning on low-level objective distortion (PSNR and SSIM).
- Multiplicity corrections: Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def compute_friedman_omnibus(order_acc_matrix: np.ndarray) -> Dict[str, float]:
    """Run non-parametric Friedman repeated-measures test across orders.
    
    Args:
        order_acc_matrix: 2D array of shape (N_images, K_orders) where each element
                          is 1.0 (correct) or 0.0 (incorrect).
                          
    Returns:
        dict with chi2_stat, p_value, kendall_w (effect size), and df.
    """
    n_images, k_orders = order_acc_matrix.shape
    if k_orders < 2:
        raise ValueError("Need at least 2 orders for Friedman test")
    if n_images < 5:
        raise ValueError("Need at least 5 images for valid asymptotic test")

    # If all orders have identical values across all images, test is trivially null
    if np.all(order_acc_matrix == order_acc_matrix[:, [0]]):
        return {
            "chi2_stat": 0.0,
            "p_value": 1.0,
            "kendall_w": 0.0,
            "df": int(k_orders - 1),
            "n_samples": int(n_images),
        }
        
    cols = [order_acc_matrix[:, j] for j in range(k_orders)]
    stat, p_val = stats.friedmanchisquare(*cols)

    if math.isnan(stat):
        stat = 0.0
        p_val = 1.0
    
    # Kendall's W = chi2 / (N * (k - 1))
    kendall_w = float(stat / (n_images * (k_orders - 1))) if n_images > 0 else 0.0
    kendall_w = max(0.0, min(1.0, kendall_w))
    
    return {
        "chi2_stat": float(stat),
        "p_value": float(p_val),
        "kendall_w": float(kendall_w),
        "df": int(k_orders - 1),
        "n_samples": int(n_images),
    }


def paired_sign_flip_test(
    diffs: np.ndarray,
    n_permutations: int = 20000,
    seed: int = 42,
) -> float:
    """Exact/Monte-Carlo paired sign-flip permutation test on matched differences.
    
    H0: The symmetric distribution of differences is centered at 0.
    Under H0, each difference d_i has equal probability of being +d_i or -d_i.
    """
    nonzero_diffs = diffs[diffs != 0]
    n_discordant = len(nonzero_diffs)
    if n_discordant == 0:
        return 1.0
        
    observed_mean = float(np.mean(diffs))
    abs_obs = abs(observed_mean)
    
    # If number of discordant pairs is small, exact combinatorial enumeration:
    if n_discordant <= 16:
        # 2^16 = 65,536 combinations
        signs = np.array(list(np.ndindex(*(2,) * n_discordant)), dtype=float) * 2 - 1.0
        # signs shape: (2^M, M)
        sum_nonzero = np.sum(signs * nonzero_diffs, axis=1)
        mean_all = sum_nonzero / len(diffs)
        n_extreme = np.sum(np.abs(mean_all) >= abs_obs - 1e-12)
        return float((1 + n_extreme) / (1 + len(signs)))

    # Monte Carlo simulation
    rng = np.random.default_rng(seed)
    # Generate random +1/-1 signs
    batch_size = 5000
    n_batches = math.ceil(n_permutations / batch_size)
    extreme_count = 0
    total_samples = 0
    
    for _ in range(n_batches):
        cur_batch = min(batch_size, n_permutations - total_samples)
        if cur_batch <= 0:
            break
        random_signs = rng.choice([-1.0, 1.0], size=(cur_batch, n_discordant))
        perm_means = np.sum(random_signs * nonzero_diffs, axis=1) / len(diffs)
        extreme_count += int(np.sum(np.abs(perm_means) >= abs_obs - 1e-12))
        total_samples += cur_batch
        
    # With +1 pseudo-count
    p_val = (1.0 + extreme_count) / (1.0 + total_samples)
    return float(p_val)


def clustered_paired_bootstrap_ci(
    y_a: np.ndarray,
    y_b: np.ndarray,
    n_resamples: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute clustered paired percentile bootstrap confidence interval for mean(y_a - y_b).
    
    Returns (diff_mean, ci_lower, ci_upper).
    """
    diffs = y_a - y_b
    n = len(diffs)
    observed_mean = float(np.mean(diffs))
    
    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(0, n, size=(n_resamples, n))
    bootstrap_means = np.mean(diffs[resample_indices], axis=1)
    
    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(bootstrap_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_means, 100.0 * (1.0 - alpha / 2.0)))
    
    return observed_mean, ci_lower, ci_upper


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Step-down Holm-Bonferroni FWER adjustment."""
    m = len(p_values)
    if m == 0:
        return []
    sorted_indices = np.argsort(p_values)
    adjusted = [0.0] * m
    cum_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        p = p_values[idx]
        adj = p * (m - rank)
        cum_max = max(cum_max, adj)
        adjusted[idx] = min(1.0, cum_max)
    return adjusted


def benjamini_hochberg_correction(p_values: List[float]) -> List[float]:
    """Step-up Benjamini-Hochberg FDR adjustment."""
    m = len(p_values)
    if m == 0:
        return []
    sorted_indices = np.argsort(p_values)
    adjusted = [0.0] * m
    cum_min = 1.0
    for rank in reversed(range(m)):
        idx = sorted_indices[rank]
        p = p_values[idx]
        adj = (p * m) / (rank + 1)
        cum_min = min(cum_min, adj)
        adjusted[idx] = min(1.0, cum_min)
    return adjusted


def distortion_controlled_ancova(
    df: pd.DataFrame,
    correct_col: str = "correct",
    psnr_col: str = "psnr",
    ssim_col: str = "ssim",
    order_col: str = "order",
    reference_order: str = "NBJ",
) -> Dict[str, Any]:
    """Logistic regression testing order effect after controlling for PSNR and SSIM.
    
    Model: logit(P(Correct=1)) = beta_0 + beta_psnr*PSNR + beta_ssim*SSIM + sum(beta_k * I(Order=k))
    
    Returns:
        dict containing coefficients, standard errors, p-values, pseudo_r2,
        and an omnibus likelihood-ratio / Wald test for the order dummy coefficients.
    """
    # Create design matrix with dummy encoding
    sub_df = df[[correct_col, psnr_col, ssim_col, order_col]].dropna()
    
    # Reference category dummy encoding
    dummies = pd.get_dummies(sub_df[order_col], drop_first=False, dtype=float)
    if reference_order in dummies.columns:
        dummies = dummies.drop(columns=[reference_order])
    order_cols = list(dummies.columns)
    
    X = pd.concat([sub_df[[psnr_col, ssim_col]], dummies], axis=1)
    X = sm.add_constant(X)
    y = sub_df[correct_col].astype(float)
    
    # Fit Logit model
    try:
        logit_model = sm.Logit(y, X).fit(disp=False, maxiter=100)
        coefs = {col: float(logit_model.params[col]) for col in X.columns}
        pvals = {col: float(logit_model.pvalues[col]) for col in X.columns}
        std_errs = {col: float(logit_model.bse[col]) for col in X.columns}
        
        # Test joint significance of order dummy variables (Wald test)
        order_idx = [X.columns.get_loc(c) for c in order_cols]
        r_matrix = np.zeros((len(order_cols), len(X.columns)))
        for i, idx in enumerate(order_idx):
            r_matrix[i, idx] = 1.0
            
        wald_test = logit_model.wald_test(r_matrix)
        order_joint_p = float(wald_test.pvalue)
        order_joint_stat = float(wald_test.statistic)
        
        return {
            "status": "success",
            "pseudo_r2": float(logit_model.prsquared),
            "coefficients": coefs,
            "std_errors": std_errs,
            "p_values": pvals,
            "order_joint_wald_stat": order_joint_stat,
            "order_joint_p_value": order_joint_p,
            "reference_order": reference_order,
            "n_obs": len(y),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "n_obs": len(y),
        }


def compute_ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE) across confidence bins."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    if n == 0:
        return 0.0
        
    for i in range(n_bins):
        b_low, b_high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= b_low) & (confidences < b_high) if i < n_bins - 1 else (confidences >= b_low) & (confidences <= b_high)
        n_b = np.sum(mask)
        if n_b > 0:
            bin_acc = np.mean(correct[mask])
            bin_conf = np.mean(confidences[mask])
            ece += (n_b / n) * abs(bin_acc - bin_conf)
            
    return float(ece)
