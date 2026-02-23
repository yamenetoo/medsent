import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english')).union({'drug', 'medication', 'pill', 'dose', 'mg'})
stemmer = PorterStemmer()

def clean_text(text, stem=True, remove_stopwords=True):
    # Remove URLs, HTML tags, dosages, non‑letters
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+\s?mg', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words]
    if stem:
        tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

def preprocess_dataframe(df, text_column='review', stem=True, remove_stopwords=True):
    df['clean_text'] = df[text_column].apply(lambda x: clean_text(x, stem, remove_stopwords))
    return df