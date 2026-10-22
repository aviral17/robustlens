import numpy as np

def soft_voting_ensemble(probs_list):
    return np.mean(probs_list, axis=0)

def weighted_ensemble(probs_list, weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    return np.sum([p * w for p, w in zip(probs_list, weights)], axis=0)

def stacking(base_preds, meta_model):
    # base_preds shape: (n_samples, n_models)
    return meta_model.predict(base_preds)
