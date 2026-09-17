"""TF-IDF features for SVM (M1) and Logistic Regression (M2)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

def build_tfidf(word_cfg: dict, char_cfg: dict):
    word_vec = TfidfVectorizer(
        analyzer=word_cfg["analyzer_word"],
        ngram_range=tuple(word_cfg["word_ngram"]),
        max_features=word_cfg["max_features"],
        sublinear_tf=word_cfg["sublinear_tf"],
        min_df=word_cfg["min_df"],
        max_df=word_cfg["max_df"],
        lowercase=False,        # already cleaned
    )
    char_vec = TfidfVectorizer(
        analyzer=char_cfg["analyzer_char"],
        ngram_range=tuple(char_cfg["char_ngram"]),
        max_features=char_cfg["max_features"],
        sublinear_tf=char_cfg["sublinear_tf"],
        min_df=char_cfg["min_df"],
        max_df=char_cfg["max_df"],
        lowercase=False,
    )
    return word_vec, char_vec

def fit_transform_pair(word_vec, char_vec, X_train):
    W = word_vec.fit_transform(X_train)
    C = char_vec.fit_transform(X_train)
    return hstack([W, C]).tocsr()

def transform_pair(word_vec, char_vec, X_other):
    W = word_vec.transform(X_other)
    C = char_vec.transform(X_other)
    return hstack([W, C]).tocsr()
