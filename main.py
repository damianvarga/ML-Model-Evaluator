import pandas as pd
from src.load_data import load_data
from src.preprocess import split_data
from src.train import train_logistic, train_random_forest
from src.evaluate import evaluate
from src.tracker import log_experiment


def main():
    df = load_data("data/dataset.csv")

    # Split using validated target
    X_train, X_test, y_train, y_test = split_data(df, "survived")

    # Train both models
    models = {
        "LogisticRegression": train_logistic(X_train, y_train),
        "RandomForest": train_random_forest(X_train, y_train),
    }

    # Evaluate and log
    scores = {}
    for name, model in models.items():
        acc = evaluate(model, X_test, y_test)
        scores[name] = acc
        log_experiment(name, acc)
        print(f"{name} accuracy: {acc:.4f}")

    # Select best model
    best_name = max(scores, key=scores.get)
    best_model = models[best_name]
    print(f"Best model: {best_name} (accuracy={scores[best_name]:.4f})")

    # Demonstrate predictions for sample passengers
    samples = pd.DataFrame([
        {"age": 25, "sex": "female", "fare": 80.0, "class": "first"},
        {"age": 40, "sex": "male", "fare": 15.0, "class": "third"},
        {"age": 6,  "sex": "female", "fare": 30.0, "class": "second"},
    ])
    preds = best_model.predict(samples)
    if hasattr(best_model, "predict_proba"):
        probs = best_model.predict_proba(samples)[:, 1]
    else:
        probs = [None] * len(preds)

    print("\nSample predictions (1=survived, 0=did not survive):")
    for i, row in samples.iterrows():
        print(f"{dict(row)} => pred={preds[i]} prob={probs[i]:.3f}" if probs[i] is not None else f"{dict(row)} => pred={preds[i]}")


if __name__ == "__main__":
    main()
