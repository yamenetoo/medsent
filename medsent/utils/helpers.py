import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_results(data, filename):
    """Save results (dict or DataFrame) to CSV/JSON."""
    if isinstance(data, pd.DataFrame):
        data.to_csv(filename, index=False)
    elif isinstance(data, dict):
        pd.DataFrame([data]).to_csv(filename, index=False)
    else:
        raise TypeError("data must be DataFrame or dict")

def load_results(filename):
    return pd.read_csv(filename)