import numpy as np
from robustlens.stats.methods import (
    percentile_bootstrap_ci, bca_bootstrap_ci, paired_bootstrap_comparison,
    mcnemar_test, holm_correction, benjamini_hochberg_correction
)

def test_percentile_bootstrap_ci():
    np.random.seed(42)
    data = np.random.normal(0, 1, 100)
    lower, upper = percentile_bootstrap_ci(data, np.mean, n_bootstraps=1000, random_state=42)
    assert lower < 0.2 and upper > -0.2

def test_bca_bootstrap_ci():
    np.random.seed(42)
    data = np.random.normal(0, 1, 100)
    lower, upper = bca_bootstrap_ci(data, np.mean, n_bootstraps=1000, random_state=42)
    assert lower < 0.2 and upper > -0.2

def test_paired_bootstrap_comparison():
    np.random.seed(42)
    data1 = np.random.normal(1, 1, 100)
    data2 = np.random.normal(0, 1, 100)
    lower, upper, p_value = paired_bootstrap_comparison(data1, data2, np.mean, random_state=42)
    assert p_value < 0.05
    assert lower > 0

def test_mcnemar_test():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred1 = np.array([0, 1, 0, 0, 0, 1]) # 1 error
    y_pred2 = np.array([1, 0, 1, 0, 1, 0]) # all wrong
    p_value = mcnemar_test(y_true, y_pred1, y_pred2)
    assert p_value < 1.0

def test_holm_correction():
    p_values = [0.01, 0.04, 0.03, 0.001]
    reject = holm_correction(p_values, alpha=0.05)
    assert reject[3] == True # 0.001 <= 0.05/4

def test_benjamini_hochberg_correction():
    p_values = [0.01, 0.04, 0.03, 0.001]
    reject = benjamini_hochberg_correction(p_values, alpha=0.05)
    assert reject[3] == True


def test_image_level_paired_bootstrap():
    """Verify cluster paired bootstrap resampling at the independent image level."""
    from robustlens.stats.methods import image_level_paired_bootstrap
    np.random.seed(42)

    # 50 images, each with 8 conditions (400 observations total)
    sample_ids = []
    corrects1 = []
    corrects2 = []
    for img_id in range(50):
        # Image base difficulty
        base_diff = np.random.uniform(0.3, 0.9)
        for cond_id in range(8):
            sample_ids.append(img_id)
            # Model 1 has slightly higher probability of correctness on this image
            c1 = 1.0 if np.random.rand() < min(1.0, base_diff + 0.15) else 0.0
            c2 = 1.0 if np.random.rand() < base_diff else 0.0
            corrects1.append(c1)
            corrects2.append(c2)

    lower, upper, p_val, diff = image_level_paired_bootstrap(
        sample_ids=sample_ids,
        corrects1=corrects1,
        corrects2=corrects2,
        n_bootstraps=500,
        alpha=0.05,
        random_state=42
    )

    assert lower <= diff <= upper
    assert diff > 0.0
    # Finite-sample continuity corrected empirical p-value cannot drop below 2 / (B + 1)
    assert p_val >= 2.0 / (500 + 1)
    assert p_val <= 1.0


def test_image_level_paired_bootstrap_details():
    """Verify details return: SE, Wald z, p_norm, and cluster integrity."""
    from robustlens.stats.methods import image_level_paired_bootstrap
    np.random.seed(42)

    sample_ids = []
    corrects1 = []
    corrects2 = []
    # 100 images, 15 conditions each
    for img_id in range(100):
        for cond_id in range(15):
            sample_ids.append(img_id)
            corrects1.append(1.0)
            corrects2.append(0.0)

    # Model 1 is strictly superior (1.0 vs 0.0)
    lower, upper, p_emp, diff, se_boot, z_stat, p_norm, img_diffs = image_level_paired_bootstrap(
        sample_ids=sample_ids,
        corrects1=corrects1,
        corrects2=corrects2,
        n_bootstraps=1000,
        alpha=0.05,
        random_state=42,
        return_details=True
    )

    assert diff == 1.0
    assert lower == 1.0 and upper == 1.0
    assert len(img_diffs) == 100
    assert p_emp == 2.0 / 1001.0
    assert se_boot == 0.0


def test_paired_sign_flip_permutation_test():
    """Verify paired sign-flip permutation test null hypothesis and bounds."""
    from robustlens.stats.methods import paired_sign_flip_permutation_test

    # When differences are strictly positive (Model 1 always wins)
    diffs = np.ones(100)
    p_perm, t_obs, z_perm, p_asymp, count_ext = paired_sign_flip_permutation_test(
        diffs=diffs,
        n_permutations=2000,
        random_state=42,
    )

    assert t_obs == 1.0
    assert count_ext == 0
    # Finite-sample exact pseudocount: (1 + 0) / (2000 + 1)
    assert p_perm == 1.0 / 2001.0
    assert z_perm == 10.0  # t_obs / (1 / sqrt(100)) = 1 / 0.1 = 10
    assert p_asymp < 1e-15

    # When differences are symmetric around 0 (null is true)
    np.random.seed(42)
    null_diffs = np.random.choice([-0.2, 0.2], size=200)
    p_perm_null, t_obs_null, z_null, p_asymp_null, _ = paired_sign_flip_permutation_test(
        diffs=null_diffs,
        n_permutations=2000,
        random_state=42,
    )
    # Under null, p-value should not reject at 0.01
    assert p_perm_null > 0.05


