import pandas as pd
from src.load_data import load_data


def test_load_data_returns_dataframe(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    df = load_data(str(csv))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]


def test_load_data_content(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text("x,y,z\n10,20,30\n")
    df = load_data(str(csv))
    assert df.iloc[0].to_dict() == {"x": 10, "y": 20, "z": 30}
