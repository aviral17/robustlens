import numpy as np

def split_conformal_prediction(cal_probs, cal_labels, test_probs, alpha=0.1):
    n = len(cal_probs)
    cal_scores = 1 - cal_probs[np.arange(n), cal_labels]
    
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    qhat = np.quantile(cal_scores, q_level, method='higher')
    
    prediction_sets = test_probs >= (1 - qhat)
    return prediction_sets

def coverage_vs_set_size(prediction_sets, true_labels):
    n = len(prediction_sets)
    coverage = np.mean(prediction_sets[np.arange(n), true_labels])
    avg_size = np.mean(np.sum(prediction_sets, axis=1))
    return coverage, avg_size
