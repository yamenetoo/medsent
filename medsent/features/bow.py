from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_bow_features(train_texts, test_texts, max_features=10000, ngram_range=(1,1)):
    vec = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    return X_train, X_test, vec

def get_tfidf_features(train_texts, test_texts, max_features=10000, ngram_range=(1,1)):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    return X_train, X_test, vec