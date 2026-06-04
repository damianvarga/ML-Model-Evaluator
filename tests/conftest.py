import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "survived": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "age":       [22, 38, 26, 35, 28, 40, 18, 45, 32, 29],
        "fare":      [7.25, 71.28, 8.05, 53.10, 12.50, 30.0, 15.0, 80.0, 20.0, 45.0],
        "sex":       ["male", "female", "male", "female", "male",
                      "female", "male", "female", "male", "female"],
        "class":     ["third", "first", "third", "first", "third",
                      "second", "third", "first", "second", "second"],
    })


@pytest.fixture
def mock_model():
    class DummyModel:
        def fit(self, X, y=None):
            return self
        def predict(self, X):
            return [0] * len(X)
        def predict_proba(self, X):
            import numpy as np
            return np.array([[0.8, 0.2]] * len(X))
    return DummyModel()
