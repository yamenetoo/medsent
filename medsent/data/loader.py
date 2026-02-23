import pandas as pd
from sklearn.model_selection import train_test_split
from . import labeling

def load_reviews(path):
    df = pd.read_csv(path)
    df.dropna(subset=['review', 'rating'], inplace=True)
    df['rating'] = df['rating'].astype(int)
    return df

def label_data(df, experiment):
    if experiment == 'binary':
        df = labeling.binary_label(df)
    elif experiment == 'ternary':
        df = labeling.ternary_label(df)
    elif experiment == '10class':
        df = labeling.tenclass_label(df)
    else:
        raise ValueError("experiment must be binary/ternary/10class")
    return df

def split_data(df, test_size=0.1, val_size=0.1, random_state=42):
    X = df['clean_text'].values
    y = df['label'].values
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test