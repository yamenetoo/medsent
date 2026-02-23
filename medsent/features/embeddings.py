import numpy as np
import gensim
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_pretrained_embeddings(path, binary=False):
    return gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary)

def get_averaged_embeddings(texts, wv_model, vector_size):
    X = []
    for text in texts:
        words = text.split()
        vecs = [wv_model[word] for word in words if word in wv_model]
        if vecs:
            X.append(np.mean(vecs, axis=0))
        else:
            X.append(np.zeros(vector_size))
    return np.array(X)

def prepare_sequence_data(train_texts, val_texts, test_texts, max_words=10000, max_len=200):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train_texts)
    X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=max_len)
    X_val = pad_sequences(tokenizer.texts_to_sequences(val_texts), maxlen=max_len)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=max_len)
    return tokenizer, X_train, X_val, X_test

def build_embedding_matrix(tokenizer, wv_model, vector_size, max_words):
    word_index = tokenizer.word_index
    emb_matrix = np.zeros((max_words, vector_size))
    for word, i in word_index.items():
        if i >= max_words:
            continue
        if word in wv_model:
            emb_matrix[i] = wv_model[word]
    return emb_matrix