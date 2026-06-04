from src.evaluate import evaluate


def test_evaluate_returns_accuracy(mock_model):
    X = [[1], [2], [3], [4]]
    y = [0, 0, 0, 0]
    acc = evaluate(mock_model, X, y)
    assert acc == 1.0


def test_evaluate_with_partial_accuracy(mock_model):
    X = [[1], [2], [3], [4]]
    y = [0, 1, 0, 1]
    acc = evaluate(mock_model, X, y)
    assert acc == 0.5
