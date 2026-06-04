import numpy as np
import pandas as pd
from src.tracker import log_experiment


def test_log_experiment_creates_csv(monkeypatch, sample_df, mock_model):
    written_data = {}
    monkeypatch.setattr(
        "src.tracker.cross_val_score",
        lambda estimator, X, y, cv=None, scoring=None: np.array([0.8, 0.82, 0.79, 0.81, 0.83]),
    )

    def mock_read_csv(path):
        raise FileNotFoundError()

    def mock_to_csv(self, path, **kwargs):
        written_data["path"] = path
        written_data["data"] = self.copy()

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)

    X = sample_df.drop(columns=["survived"])
    y = sample_df["survived"]
    cv_mean, cv_std = log_experiment("TestModel", 0.85, mock_model, X, y)

    assert written_data["data"].iloc[0]["model"] == "TestModel"
    assert float(written_data["data"].iloc[0]["accuracy"]) == 0.85
    assert cv_mean is not None
    assert cv_std is not None


def test_log_experiment_appends_to_existing(monkeypatch):
    written_data = {}

    def mock_read_csv(path):
        return pd.DataFrame([{"model": "Existing", "accuracy": 0.9}])

    def mock_to_csv(self, path, **kwargs):
        written_data["path"] = path
        written_data["data"] = self.copy()

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)

    log_experiment("NewModel", 0.8)

    assert len(written_data["data"]) == 2
    assert written_data["data"].iloc[-1]["model"] == "NewModel"


def test_log_experiment_no_cv_when_no_model(monkeypatch):
    written_data = {}

    def mock_read_csv(path):
        raise FileNotFoundError()

    def mock_to_csv(self, path, **kwargs):
        written_data["data"] = self.copy()

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)

    cv_mean, cv_std = log_experiment("NoCV", 0.75)

    assert cv_mean is None
    assert cv_std is None
    assert written_data["data"].iloc[0]["model"] == "NoCV"
