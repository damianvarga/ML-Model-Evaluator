import pandas as pd
from datetime import datetime
from sklearn.model_selection import cross_val_score
import numpy as np
from os import path, makedirs
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def log_experiment(model_name, accuracy, model=None, X_train=None, y_train=None):

    cv_mean = None
    cv_std = None

    # Run cross-validation only if model + training data exist
    if model is not None and X_train is not None and y_train is not None:
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy"
        )

        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

    data = {
        "model": model_name,
        "accuracy": accuracy,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "timestamp": datetime.now()
    }

    BASE_DIR = path.dirname(path.abspath(__file__))
    CSV_PATH = path.join(BASE_DIR, "..", "experiments.csv")
    df = pd.DataFrame([data])

    try:
        old = pd.read_csv(CSV_PATH)
        df = pd.concat([old, df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_csv(CSV_PATH, index=False)

    if cv_mean is not None:
        print(f"Logged {model_name} | acc={accuracy:.4f} | CV={cv_mean:.4f} ± {cv_std:.4f}")
    else:
        print(f"Logged {model_name} | acc={accuracy:.4f}")
    
    return cv_mean, cv_std

def _get_or_create_experiment(name="ml-model-comparison"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        artifact_path = path.abspath(path.join("mlruns", name))
        makedirs(artifact_path, exist_ok=True)
        artifact_uri = "file:///" + artifact_path.replace("\\", "/")
        experiment_id = client.create_experiment(
            name=name,
            artifact_location=artifact_uri,
        )
        experiment = client.get_experiment(experiment_id)
    return experiment

def log_mlflow(model, model_name, accuracy, cv_mean=None, cv_std=None, params=None):

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment = _get_or_create_experiment()
    mlflow.set_experiment(experiment.name)

    with mlflow.start_run(run_name=model_name):

        mlflow.log_metric("accuracy", accuracy)

        if cv_mean is not None:
            mlflow.log_metric("cv_mean", cv_mean)

        if cv_std is not None:
            mlflow.log_metric("cv_std", cv_std)

        if params:
            mlflow.log_params(params)

        mlflow.sklearn.log_model(sk_model=model, name="model")

        print(f"MLflow logged: {model_name}")