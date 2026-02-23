import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score)

def evaluate(y_true, y_pred, average='macro'):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average)
    cm = confusion_matrix(y_true, y_pred)
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_pred)
    else:
        auc = None
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'auc': auc, 'confusion_matrix': cm}