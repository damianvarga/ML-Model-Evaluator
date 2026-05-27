import pandas as pd
from datetime import datetime
from sklearn.model_selection import cross_val_score
import numpy as np


def log_experiment(model_name, accuracy, model=None, X_train=None, y_train=None):

    cv_mean = None
    cv_std = None

    # Run cross-validation only if data + model are provided
    if model is not None and X_train is not None and y_train is not None:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

    data = {
        "model": model_name,
        "accuracy": accuracy,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "timestamp": datetime.now()
    }

    df = pd.DataFrame([data])

    try:
        old = pd.read_csv("experiments.csv")
        df = pd.concat([old, df], ignore_index=True)
    except FileNotFoundError:
        pass

    df.to_csv("experiments.csv", index=False)

    print(f"Logged {model_name} | acc={accuracy:.4f} | CV={cv_mean:.4f} ± {cv_std:.4f}")