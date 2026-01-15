from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import make_scorer, f1_score

from utils.config import RANDOM_STATE, CV_FOLDS_BINARY, CV_FOLDS_MULTILABEL


# -------------------------
# Binary (Subtask_1)
# -------------------------

def run_logreg_gridsearch(X_train, y_train, vectorizer, param_grid):

    pipeline = Pipeline(
        [
            ("vectorizer", vectorizer),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=CV_FOLDS_BINARY,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2,
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_score_, grid.best_params_


# -------------------------
# Multilabel (Subtask_2 & Subtask_3)
# -------------------------

def run_logreg_gridsearch_multilabel(X_train, y_train, vectorizer, param_grid):
    
    pipeline = Pipeline(
        [
            ("vectorizer", vectorizer),
            (
                "classifier",
                MultiOutputClassifier(
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    )
                ),
            ),
        ]
    )

    scorer = make_scorer(f1_score, average="macro", zero_division=0)

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=CV_FOLDS_MULTILABEL,
        scoring=scorer,
        n_jobs=-1,
        verbose=2,
        error_score=0.0,
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_score_, grid.best_params_