from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.train import _build_preprocessor, train_logistic, train_random_forest, train_xgboost


class TestBuildPreprocessor:
    def test_returns_column_transformer(self, sample_df):
        preproc = _build_preprocessor(sample_df.drop(columns=["survived"]))
        assert isinstance(preproc, ColumnTransformer)

    def test_transforms_data(self, sample_df):
        X = sample_df.drop(columns=["survived"])
        preproc = _build_preprocessor(X)
        result = preproc.fit_transform(X)
        assert result.shape[0] == len(X)
        assert result.shape[1] > 0


class TestTrainLogistic:
    def test_returns_fitted_gridsearch(self, sample_df):
        X = sample_df.drop(columns=["survived"])
        y = sample_df["survived"]
        gs = train_logistic(X, y)
        assert hasattr(gs, "best_estimator_")
        assert hasattr(gs, "best_params_")
        assert gs.best_score_ > 0


class TestTrainRandomForest:
    def test_returns_fitted_gridsearch(self, sample_df):
        X = sample_df.drop(columns=["survived"])
        y = sample_df["survived"]
        gs = train_random_forest(X, y)
        assert hasattr(gs, "best_estimator_")
        assert hasattr(gs, "best_params_")
        assert gs.best_score_ > 0


class TestTrainXGBoost:
    def test_returns_fitted_gridsearch(self, sample_df):
        X = sample_df.drop(columns=["survived"])
        y = sample_df["survived"]
        gs = train_xgboost(X, y)
        assert hasattr(gs, "best_estimator_")
        assert hasattr(gs, "best_params_")
        assert gs.best_score_ > 0
