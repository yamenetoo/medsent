import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from ...config import OUTPUT_DIR, RANDOM_STATE, GLOVE_PATH, PUBMED_PATH
from ...data.loader import load_reviews, label_data, split_data
from ...data.preprocessing import preprocess_dataframe
from ...features.bow import get_bow_features, get_tfidf_features
from ...features.embeddings import (load_pretrained_embeddings,
                                     get_averaged_embeddings,
                                     prepare_sequence_data,
                                     build_embedding_matrix)
from ...models.ml_models import get_ml_models
from ...models.dl_models import build_rnn_model, train_deep_model
from ...evaluation.metrics import evaluate
from ...utils.helpers import set_seed

def run():
    set_seed(RANDOM_STATE)
    df = load_reviews('data/drugscom_reviews.csv')
    df = preprocess_dataframe(df)
    df = label_data(df, 'binary')
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    results = []

    # ---- BoW ----
    X_tr_bow, X_te_bow, _ = get_bow_features(X_train, X_test)
    models, _ = get_ml_models()
    for name, model in models.items():
        if name in ['kNN', 'LDA']: continue
        model.fit(X_tr_bow, y_train)
        pred = model.predict(X_te_bow)
        f1 = evaluate(y_test, pred)['f1']
        results.append({'Model': name, 'Features': 'BoW', 'F1': f1})
    nb = MultinomialNB()
    nb.fit(X_tr_bow, y_train)
    pred = nb.predict(X_te_bow)
    results.append({'Model': 'NaiveBayes', 'Features': 'BoW', 'F1': evaluate(y_test, pred)['f1']})

    # ---- TF-IDF ----
    X_tr_tf, X_te_tf, _ = get_tfidf_features(X_train, X_test)
    for name, model in models.items():
        if name in ['kNN', 'LDA']: continue
        model.fit(X_tr_tf, y_train)
        pred = model.predict(X_te_tf)
        f1 = evaluate(y_test, pred)['f1']
        results.append({'Model': name, 'Features': 'TF-IDF', 'F1': f1})
    nb = MultinomialNB()
    nb.fit(X_tr_tf, y_train)
    pred = nb.predict(X_te_tf)
    results.append({'Model': 'NaiveBayes', 'Features': 'TF-IDF', 'F1': evaluate(y_test, pred)['f1']})

    # ---- Averaged GloVe ----
    if os.path.exists(GLOVE_PATH):
        glove = load_pretrained_embeddings(GLOVE_PATH, binary=False)
        X_tr_glove = get_averaged_embeddings(X_train, glove, 300)
        X_te_glove = get_averaged_embeddings(X_test, glove, 300)
        scaler = StandardScaler()
        X_tr_glove_scaled = scaler.fit_transform(X_tr_glove)
        X_te_glove_scaled = scaler.transform(X_te_glove)
        for name, model in models.items():
            if name in ['SVM_linear', 'SVM_rbf', 'LogisticRegression', 'kNN', 'LDA']:
                model.fit(X_tr_glove_scaled, y_train)
                pred = model.predict(X_te_glove_scaled)
            else:
                model.fit(X_tr_glove, y_train)
                pred = model.predict(X_te_glove)
            f1 = evaluate(y_test, pred)['f1']
            results.append({'Model': name, 'Features': 'GloVe avg', 'F1': f1})
        gnb = GaussianNB()
        gnb.fit(X_tr_glove_scaled, y_train)
        pred = gnb.predict(X_te_glove_scaled)
        results.append({'Model': 'GaussianNB', 'Features': 'GloVe avg', 'F1': evaluate(y_test, pred)['f1']})

    # ---- Averaged PubMed ----
    if os.path.exists(PUBMED_PATH):
        pubmed = load_pretrained_embeddings(PUBMED_PATH, binary=True)
        X_tr_pub = get_averaged_embeddings(X_train, pubmed, 300)
        X_te_pub = get_averaged_embeddings(X_test, pubmed, 300)
        scaler = StandardScaler()
        X_tr_pub_scaled = scaler.fit_transform(X_tr_pub)
        X_te_pub_scaled = scaler.transform(X_te_pub)
        for name, model in models.items():
            if name in ['SVM_linear', 'SVM_rbf', 'LogisticRegression', 'kNN', 'LDA']:
                model.fit(X_tr_pub_scaled, y_train)
                pred = model.predict(X_te_pub_scaled)
            else:
                model.fit(X_tr_pub, y_train)
                pred = model.predict(X_te_pub)
            f1 = evaluate(y_test, pred)['f1']
            results.append({'Model': name, 'Features': 'PubMed avg', 'F1': f1})
        gnb = GaussianNB()
        gnb.fit(X_tr_pub_scaled, y_train)
        pred = gnb.predict(X_te_pub_scaled)
        results.append({'Model': 'GaussianNB', 'Features': 'PubMed avg', 'F1': evaluate(y_test, pred)['f1']})

    # ---- Averaged Concat (GloVe+PubMed) ----
    if os.path.exists(GLOVE_PATH) and os.path.exists(PUBMED_PATH):
        X_tr_concat = np.concatenate([X_tr_glove, X_tr_pub], axis=1)
        X_te_concat = np.concatenate([X_te_glove, X_te_pub], axis=1)
        scaler = StandardScaler()
        X_tr_concat_scaled = scaler.fit_transform(X_tr_concat)
        X_te_concat_scaled = scaler.transform(X_te_concat)
        for name, model in models.items():
            if name in ['SVM_linear', 'SVM_rbf', 'LogisticRegression', 'kNN', 'LDA']:
                model.fit(X_tr_concat_scaled, y_train)
                pred = model.predict(X_te_concat_scaled)
            else:
                model.fit(X_tr_concat, y_train)
                pred = model.predict(X_te_concat)
            f1 = evaluate(y_test, pred)['f1']
            results.append({'Model': name, 'Features': 'Concat avg', 'F1': f1})
        gnb = GaussianNB()
        gnb.fit(X_tr_concat_scaled, y_train)
        pred = gnb.predict(X_te_concat_scaled)
        results.append({'Model': 'GaussianNB', 'Features': 'Concat avg', 'F1': evaluate(y_test, pred)['f1']})

    # ---- Sequence embeddings (Deep) ----
    tokenizer, X_tr_seq, X_val_seq, X_te_seq = prepare_sequence_data(
        X_train, X_val, X_test, max_words=10000, max_len=200
    )
    y_train_cat = tf.keras.utils.to_categorical(y_train)
    y_val_cat = tf.keras.utils.to_categorical(y_val)

    # GloVe sequence
    if os.path.exists(GLOVE_PATH):
        glove = load_pretrained_embeddings(GLOVE_PATH, binary=False)
        emb_mat = build_embedding_matrix(tokenizer, glove, 300, 10000)
        for rnn_type in ['lstm', 'gru', 'rnn']:
            model = build_rnn_model(rnn_type, 10000, 200, emb_mat, num_classes=len(np.unique(y_train)))
            model, _ = train_deep_model(model, X_tr_seq, y_train_cat, X_val_seq, y_val_cat)
            pred_probs = model.predict(X_te_seq)
            pred = np.argmax(pred_probs, axis=1)
            f1 = evaluate(y_test, pred)['f1']
            results.append({'Model': f'Bi-{rnn_type.upper()}', 'Features': 'GloVe seq', 'F1': f1})

    # PubMed sequence
    if os.path.exists(PUBMED_PATH):
        pubmed = load_pretrained_embeddings(PUBMED_PATH, binary=True)
        emb_mat = build_embedding_matrix(tokenizer, pubmed, 300, 10000)
        for rnn_type in ['lstm', 'gru', 'rnn']:
            model = build_rnn_model(rnn_type, 10000, 200, emb_mat, num_classes=len(np.unique(y_train)))
            model, _ = train_deep_model(model, X_tr_seq, y_train_cat, X_val_seq, y_val_cat)
            pred_probs = model.predict(X_te_seq)
            pred = np.argmax(pred_probs, axis=1)
            f1 = evaluate(y_test, pred)['f1']
            results.append({'Model': f'Bi-{rnn_type.upper()}', 'Features': 'PubMed seq', 'F1': f1})

    # Concat sequence (GloVe+PubMed)
    if os.path.exists(GLOVE_PATH) and os.path.exists(PUBMED_PATH):
        emb_mat_glove = build_embedding_matrix(tokenizer, glove, 300, 10000)
        emb_mat_pub = build_embedding_matrix(tokenizer, pubmed, 300, 10000)
        emb_mat_concat = np.concatenate([emb_mat_glove, emb_mat_pub], axis=1)
        for rnn_type in ['lstm', 'gru', 'rnn']:
            model = build_rnn_model(rnn_type, 10000, 200, emb_mat_concat, num_classes=len(np.unique(y_train)))
            model, _ = train_deep_model(model, X_tr_seq, y_train_cat, X_val_seq, y_val_cat)
            pred_probs = model.predict(X_te_seq)
            pred = np.argmax(pred_probs, axis=1)
            f1 = evaluate(y_test, pred)['f1']
            results.append({'Model': f'Bi-{rnn_type.upper()}', 'Features': 'Concat seq', 'F1': f1})

    # Save results
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(OUTPUT_DIR, 'table_4_2_binary_all.csv'), index=False)
    return df_res