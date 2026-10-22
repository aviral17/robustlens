import numpy as np

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        if bin_lower == 0.0:
            in_bin = in_bin | (y_prob == 0.0)
            
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin] == np.argmax(y_prob[in_bin], axis=-1) if y_prob.ndim > 1 else y_true[in_bin] == (y_prob[in_bin] > 0.5))
            if y_prob.ndim > 1:
                conf_in_bin = np.mean(np.max(y_prob[in_bin], axis=-1))
            else:
                conf_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - conf_in_bin) * prop_in_bin
            
    return ece

def reliability_diagram_data(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        if bin_lower == 0.0:
            in_bin = in_bin | (y_prob == 0.0)
            
        if np.any(in_bin):
            accuracy = np.mean(y_true[in_bin] == np.argmax(y_prob[in_bin], axis=-1) if y_prob.ndim > 1 else y_true[in_bin] == (y_prob[in_bin] > 0.5))
            if y_prob.ndim > 1:
                confidence = np.mean(np.max(y_prob[in_bin], axis=-1))
            else:
                confidence = np.mean(y_prob[in_bin])
        else:
            accuracy = np.nan
            confidence = np.nan
            
        accuracies.append(accuracy)
        confidences.append(confidence)
        
    return np.array(accuracies), np.array(confidences)

def temperature_scaling(logits, temperature):
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
