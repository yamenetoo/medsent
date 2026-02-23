from gensim.models import Word2Vec

def train_word2vec(texts, vector_size=300, window=5, min_count=5, sg=1):
    tokenized = [text.split() for text in texts]
    model = Word2Vec(sentences=tokenized, vector_size=vector_size,
                     window=window, min_count=min_count, sg=sg, workers=4)
    return model