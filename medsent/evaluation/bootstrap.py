import numpy as np
from sklearn.metrics import f1_score

def bootstrap_test(y_true, pred_model1, pred_model2, n_iterations=1000, metric='f1_macro'):
    """
    Paired bootstrap test for comparing two models.
    Returns p-value (two-tailed) that model1 is different from model2.
    """
    np.random.seed(42)
    n = len(y_true)
    scores1 = []
    scores2 = []
    for _ in range(n_iterations):
        idx = np.random.choice(n, n, replace=True)
        if metric == 'f1_macro':
            s1 = f1_score(y_true[idx], pred_model1[idx], average='macro')
            s2 = f1_score(y_true[idx], pred_model2[idx], average='macro')
        else:
            raise ValueError("Only f1_macro supported")
        scores1.append(s1)
        scores2.append(s2)
    diff = np.array(scores1) - np.array(scores2)
    # Two-tailed test: proportion of absolute differences >= observed absolute difference
    obs_diff = f1_score(y_true, pred_model1, average='macro') - f1_score(y_true, pred_model2, average='macro')
    p_value = np.mean(np.abs(diff) >= np.abs(obs_diff))
    return p_value