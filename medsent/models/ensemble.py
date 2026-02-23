import numpy as np

def hard_voting_ensemble(predictions_list):
    preds = np.array(predictions_list)  # shape (n_models, n_samples)
    return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)

def soft_voting_ensemble(probabilities_list):
    avg_probs = np.mean(probabilities_list, axis=0)
    return np.argmax(avg_probs, axis=1)