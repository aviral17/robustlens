import numpy as np
from robustlens.calibration.methods import expected_calibration_error, reliability_diagram_data, temperature_scaling

def test_expected_calibration_error():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2])
    # For binary case in our implementation
    ece = expected_calibration_error(y_true, y_prob, n_bins=2)
    assert ece >= 0

def test_reliability_diagram_data():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2])
    acc, conf = reliability_diagram_data(y_true, y_prob, n_bins=2)
    assert len(acc) == 2
    assert len(conf) == 2

def test_temperature_scaling():
    logits = np.array([[2.0, 1.0], [0.1, 2.5]])
    probs = temperature_scaling(logits, temperature=2.0)
    assert np.allclose(np.sum(probs, axis=-1), 1.0)
