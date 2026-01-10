from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV

def make_pipeline(vectorizer):
    return Pipeline([
        ("vect", vectorizer),
        ("clf", LogisticRegression(
            solver="liblinear",
            penalty="l2",
            max_iter=2000,
            random_state=42
        ))
    ])
