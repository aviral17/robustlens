import numpy as np
from scipy import stats as scipy_stats

def percentile_bootstrap_ci(data, stat_func, n_bootstraps=1000, alpha=0.05, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n = len(data)
    boot_stats = []
    for _ in range(n_bootstraps):
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(stat_func(boot_sample))
    
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return lower, upper

def bca_bootstrap_ci(data, stat_func, n_bootstraps=1000, alpha=0.05, random_state=None):
    # Fallback to scipy's bootstrap which does BCa
    res = scipy_stats.bootstrap((data,), stat_func, n_resamples=n_bootstraps, 
                                confidence_level=1-alpha, method='BCa', random_state=random_state)
    return res.confidence_interval.low, res.confidence_interval.high

def paired_bootstrap_comparison(data1, data2, stat_func, n_bootstraps=1000, alpha=0.05, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    data1 = np.array(data1)
    data2 = np.array(data2)
    n = len(data1)
    
    diff_stats = []
    for _ in range(n_bootstraps):
        indices = np.random.choice(np.arange(n), size=n, replace=True)
        stat1 = stat_func(data1[indices])
        stat2 = stat_func(data2[indices])
        diff_stats.append(stat1 - stat2)
    
    lower = np.percentile(diff_stats, 100 * alpha / 2)
    upper = np.percentile(diff_stats, 100 * (1 - alpha / 2))
    p_value = np.mean(np.array(diff_stats) <= 0) * 2
    p_value = min(p_value, 1.0)
    return lower, upper, p_value

def paired_sign_flip_permutation_test(
    diffs,
    n_permutations=20000,
    random_state=None,
):
    """
    Exact / Monte Carlo Paired Sign-Flip Permutation Test.

    Tests the null hypothesis H0: The distribution of paired image differences d_i
    is symmetric about zero (exchangeability of model classifications for each image),
    against H1: E[d_i] != 0 (two-sided).

    Under H0, swapping model identities for image i corresponds to multiplying d_i
    by a random Rademacher sign s_i in {-1, +1} with probability 0.5.

    Computes:
    1. Observed mean difference: T_obs = mean(diffs).
    2. Monte Carlo sign-flip permutation p-value with finite-sample pseudocount correction:
       p_perm = (1 + sum(|T_perm| >= |T_obs|)) / (R + 1).
    3. Asymptotic permutation test statistic and p-value (Lindeberg-Feller CLT):
       sigma0 = sqrt(sum(diffs^2)) / N, z_perm = T_obs / sigma0,
       p_asymp = 2 * (1 - Phi(|z_perm|)).
    """
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random.RandomState()

    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    t_obs = float(np.mean(diffs))

    # Theoretical null variance under the Rademacher sign-flip distribution
    sigma0 = float(np.sqrt(np.sum(diffs**2)) / n)
    z_perm = float(t_obs / sigma0) if sigma0 > 0 else 0.0
    p_asymp = float(2.0 * scipy_stats.norm.sf(abs(z_perm)))

    # Vectorized Monte Carlo permutations in flat memory batches
    count_ext = 0
    batch_size = 5000
    abs_t_obs = abs(t_obs)
    for b in range(0, n_permutations, batch_size):
        cur_b = min(batch_size, n_permutations - b)
        signs = rng.choice([-1.0, 1.0], size=(cur_b, n))
        t_samples = np.abs(np.dot(signs, diffs) / n)
        count_ext += int(np.sum(t_samples >= abs_t_obs))

    p_perm = float((1 + count_ext) / (n_permutations + 1))
    return p_perm, t_obs, z_perm, p_asymp, count_ext


def image_level_paired_bootstrap(
    sample_ids,
    corrects1,
    corrects2,
    n_bootstraps=10000,
    alpha=0.05,
    random_state=None,
    return_details=False,
):
    """
    Cluster paired bootstrap resampling at the independent physical image level.
    Preserves all condition observations for each resampled image.

    Used for uncertainty estimation (calculating confidence intervals and standard errors).
    """
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random.RandomState()

    sample_ids = np.array(sample_ids)
    corrects1 = np.array(corrects1, dtype=float)
    corrects2 = np.array(corrects2, dtype=float)

    unique_samples = np.unique(sample_ids)
    n_images = len(unique_samples)

    import pandas as pd
    s1 = pd.Series(corrects1).groupby(sample_ids).mean()
    s2 = pd.Series(corrects2).groupby(sample_ids).mean()
    image_diffs = (s1 - s2).values
    observed_diff = float(np.mean(image_diffs))

    boot_diffs = np.empty(n_bootstraps)
    for b in range(n_bootstraps):
        resampled_idx = rng.randint(0, n_images, size=n_images)
        boot_diffs[b] = np.mean(image_diffs[resampled_idx])

    lower = float(np.percentile(boot_diffs, 100 * (alpha / 2.0)))
    upper = float(np.percentile(boot_diffs, 100 * (1.0 - alpha / 2.0)))

    # Empirical bootstrap tail proportion
    count_le = int(np.sum(boot_diffs <= 0))
    count_ge = int(np.sum(boot_diffs >= 0))
    min_tail = min(count_le, count_ge)
    p_emp = float(min(1.0, 2.0 * (min_tail + 1) / (n_bootstraps + 1)))

    se_boot = float(np.std(boot_diffs, ddof=1))
    z_stat = float(observed_diff / se_boot) if se_boot > 0 else 0.0
    p_norm = float(2.0 * scipy_stats.norm.sf(abs(z_stat)))

    if return_details:
        return lower, upper, p_emp, observed_diff, se_boot, z_stat, p_norm, image_diffs
    return lower, upper, p_emp, observed_diff

def mcnemar_test(y_true, y_pred1, y_pred2):

    correct1 = (y_true == y_pred1)
    correct2 = (y_true == y_pred2)
    
    n01 = np.sum(~correct1 & correct2)
    n10 = np.sum(correct1 & ~correct2)
    
    if n01 + n10 == 0:
        return 1.0
        
    statistic = (abs(n01 - n10) - 1)**2 / (n01 + n10)
    p_value = scipy_stats.chi2.sf(statistic, 1)
    return p_value

def holm_correction(p_values, alpha=0.05):
    p_values = np.array(p_values)
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    reject = np.zeros(n, dtype=bool)
    for i, p in enumerate(sorted_p):
        if p <= alpha / (n - i):
            reject[sorted_indices[i]] = True
        else:
            break
    return reject

def benjamini_hochberg_correction(p_values, alpha=0.05):
    p_values = np.array(p_values)
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    reject = np.zeros(n, dtype=bool)
    for i in range(n-1, -1, -1):
        if sorted_p[i] <= (i + 1) / n * alpha:
            reject[sorted_indices[:i+1]] = True
            break
    return reject
