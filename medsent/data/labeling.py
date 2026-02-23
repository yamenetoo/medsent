from sklearn.preprocessing import LabelEncoder

def binary_label(df):
    def label(rating):
        if rating <= 4:
            return 'negative'
        elif rating >= 7:
            return 'positive'
        else:
            return None
    df['label_str'] = df['rating'].apply(label)
    df = df.dropna(subset=['label_str']).copy()
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label_str'])
    return df

def ternary_label(df):
    def label(rating):
        if rating <= 4:
            return 'negative'
        elif 5 <= rating <= 6:
            return 'neutral'
        else:
            return 'positive'
    df['label_str'] = df['rating'].apply(label)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label_str'])
    return df

def tenclass_label(df):
    df['label_str'] = df['rating'].astype(str)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label_str'])
    return df