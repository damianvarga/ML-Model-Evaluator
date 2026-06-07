from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
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
    
    param_grid = {
        'model__C': [0.1, 1, 10, 100],
        'model__l1_ratio': [0],
        'model__solver': ['lbfgs', 'liblinear'],
        'model__max_iter': [1000, 2000],
    }
    
    base_model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(random_state=42)),
    ])
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}\n")
    
    return grid_search


def train_random_forest(X_train, y_train):    
    preprocessor = _build_preprocessor(X_train)
    
    param_grid = {
        'model__n_estimators': [50, 100],
        'model__max_depth': [5, 10, None],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2],
    }
    
    base_model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(random_state=42)),
    ])
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}\n")
    return grid_search


def train_xgboost(X_train, y_train):
    import xgboost as xgb

    preprocessor = _build_preprocessor(X_train)

    param_grid = {
        'model__n_estimators': [50, 100],
        'model__max_depth': [3, 6, 10],
        'model__learning_rate': [0.01, 0.1, 0.3],
        'model__subsample': [0.8, 1.0],
    }

    base_model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", xgb.XGBClassifier(random_state=42)),
    ])

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}\n")
    return grid_search
