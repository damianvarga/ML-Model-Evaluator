from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    X = X.copy()
    X.columns = X.columns.str.strip()

    cat_features = [c for c in X.columns if X[c].dtype == 'object']
    num_features = [c for c in X.columns if c not in cat_features]

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ],
        remainder="drop",
    )
    return preprocessor


def train_logistic(X_train, y_train):
    preprocessor = _build_preprocessor(X_train)
    clf = LogisticRegression(max_iter=1000)

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", clf),
    ])
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    preprocessor = _build_preprocessor(X_train)
    clf = RandomForestClassifier(n_estimators=300, random_state=42)

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", clf),
    ])
    model.fit(X_train, y_train)
    return model
