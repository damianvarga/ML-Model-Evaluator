import pandas as pd
import pytest
from src.preprocess import split_data


class TestSplitData:
    def test_returns_four_elements(self, sample_df):
        result = split_data(sample_df, "survived")
        assert len(result) == 4

    def test_target_removed_from_X(self, sample_df):
        X_train, X_test, y_train, y_test = split_data(sample_df, "survived")
        assert "survived" not in X_train.columns
        assert "survived" not in X_test.columns

    def test_y_values_preserved(self, sample_df):
        X_train, X_test, y_train, y_test = split_data(sample_df, "survived")
        all_y = list(y_train) + list(y_test)
        assert sorted(all_y) == sorted(sample_df["survived"].tolist())

    def test_train_test_split_ratio(self, sample_df):
        X_train, X_test, y_train, y_test = split_data(sample_df, "survived")
        total = len(sample_df)
        assert len(X_train) == pytest.approx(total * 0.8, rel=0.1)
        assert len(X_test) == pytest.approx(total * 0.2, rel=0.1)

    def test_raises_on_missing_target(self, sample_df):
        with pytest.raises(ValueError, match="not found"):
            split_data(sample_df, "nonexistent")

    def test_strips_whitespace_from_target(self, sample_df):
        X_train, X_test, y_train, y_test = split_data(sample_df, "  survived  ")
        assert "survived" not in X_train.columns

    def test_original_df_unchanged(self, sample_df):
        original_cols = list(sample_df.columns)
        split_data(sample_df, "survived")
        assert list(sample_df.columns) == original_cols
