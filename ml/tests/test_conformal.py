import numpy as np
from robustlens.conformal.methods import split_conformal_prediction, coverage_vs_set_size

def test_split_conformal_prediction():
    cal_probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    cal_labels = np.array([1, 0, 1])
    test_probs = np.array([[0.2, 0.8], [0.9, 0.1]])
    
    prediction_sets = split_conformal_prediction(cal_probs, cal_labels, test_probs, alpha=0.1)
    assert prediction_sets.shape == (2, 2)
    
    test_labels = np.array([1, 0])
    cov, size = coverage_vs_set_size(prediction_sets, test_labels)
    assert cov <= 1.0
    assert size >= 1.0
