import numpy as np
from robustlens.ensembles.methods import soft_voting_ensemble, weighted_ensemble, stacking

def test_soft_voting_ensemble():
    probs = [np.array([[0.1, 0.9]]), np.array([[0.3, 0.7]])]
    res = soft_voting_ensemble(probs)
    assert np.allclose(res, [[0.2, 0.8]])

def test_weighted_ensemble():
    probs = [np.array([[0.1, 0.9]]), np.array([[0.3, 0.7]])]
    res = weighted_ensemble(probs, [0.5, 0.5])
    assert np.allclose(res, [[0.2, 0.8]])

def test_stacking():
    class DummyMeta:
        def predict(self, X):
            return np.mean(X, axis=1)
    base_preds = np.array([[0.1, 0.3], [0.8, 0.9]])
    res = stacking(base_preds, DummyMeta())
    assert np.allclose(res, [0.2, 0.85])
