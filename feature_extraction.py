from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_features(corpus, max_features=8000, ngram_range=(1, 2), min_df=3):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )

    features = vectorizer.fit_transform(corpus)

    return vectorizer, features
