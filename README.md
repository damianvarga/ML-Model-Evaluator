# ML Model Comparison Pipeline

A lightweight machine learning framework for training, comparing, and tracking classification models with a REST API for serving predictions.

Demonstrates a complete ML workflow: preprocessing, model training, evaluation, cross-validation, experiment tracking via MLflow, and API deployment with FastAPI.

---

## Overview

- Load and preprocess structured tabular data
- Train multiple classification models (Logistic Regression, Random Forest, XGBoost)
- Evaluate with accuracy, precision, recall, F1-score
- K-Fold cross-validation for robust evaluation
- Track experiments with MLflow (runs in Docker alongside the app)
- Serve predictions via FastAPI REST API

---

## Features

- End-to-end pipeline: data → preprocessing → training → evaluation → API
- Automatic OneHot encoding + StandardScaler via `ColumnTransformer`
- Hyperparameter tuning with `GridSearchCV`
- Experiment logging: CSV fallback + MLflow tracking server
- Best model persistence with joblib
- REST API:
  - `GET /` — health check
  - `POST /predict` — predict survival
- Docker Compose: app + MLflow tracking server

---

## Project Structure

```text
data/
├── dataset.csv                 # Synthetic dataset

models/
└── best_model.pkl              # Trained best model (saved by joblib)

src/
├── api.py                      # FastAPI app (GET /, POST /predict)
├── create_dataset.py           # Dataset generator
├── evaluate.py                 # Evaluation metrics
├── load_data.py                # Data loading utilities
├── preprocess.py               # Train/test split + validation
├── tracker.py                  # Experiment logging (MLflow + CSV)
└── train.py                    # Model pipelines and GridSearchCV

tests/
├── conftest.py                 # Shared fixtures
├── test_evaluate.py
├── test_load_data.py
├── test_preprocess.py
├── test_tracker.py
└── test_train.py

docker-entrypoint.py            # Entrypoint: trains model (unless cached), starts API
docker-compose.yml              # App + MLflow tracking server
Dockerfile                      # App image build
main.py                         # Pipeline entry point
experiments.csv                 # CSV experiment log
requirements.txt                # Dependencies
README.md
```

---

## Usage

### 1. Run with Docker Compose (recommended)

```bash
docker compose up --build
``` 
<!-- mam otestovane vsetky prikazy s existujucim souborom best_model.pkl - funguje, ale nieco mi nedetegovalo best_model.pkl -->

Starts two services:
- **app** — trains model on first startup, serves API on port 8000
- **mlflow** — MLflow tracking server on port 5000

> **Note:** Set `FORCE_TRAIN=true` to retrain even if `best_model.pkl` exists:  
> `$env:FORCE_TRAIN = "true"; docker compose up --build`
> If you want to change it back, use `$env:FORCE_TRAIN = "false"; docker compose up --build`
>   
> **Note 2:** Alternatively, you can delete mlflow.db and models/best_model.pkl to force retraining.


### 2. Run locally

```bash
pip install -r requirements.txt
uvicorn src.api:app --port 8000
```

This command starts only the API on port 8000.

The model is auto-trained on the first API request if `models/best_model.pkl` is missing.

### 3. Manual training (optional)

```bash
python main.py
```

### 4. Tests

```bash
python -m pytest tests/ -v
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow tracking URI. For local runs, this can point directly to a SQLite database. In Docker, it is set to `http://mlflow:5000` so the application connects to the MLflow tracking server. |
| `MLFLOW_ARTIFACT_ROOT` | `mlruns/` (local) | Artifact storage path (set to `/mlflow/artifacts` for the MLflow server in Docker) |
| `FORCE_TRAIN` | *(unset)* | Set to `true` to retrain even if `best_model.pkl` exists |

Example — force retraining in Docker:
```bash
# docker compose run --rm -e FORCE_TRAIN=true app
$env:FORCE_TRAIN = "true"; docker compose up --build
```

---

## API Endpoints

### `GET /`
```json
{"message": "ML Model API running"}
```

### `POST /predict`

Request:
```json
{
  "age": 25,
  "sex": "female",
  "fare": 80.0,
  "class_name": "first"
}
```

Response:
```json
{
  "prediction": 1,
  "survival_probability": 0.7647
}
```

--- 
## Future Improvements 

- Preserve experiment history when using Docker Compose
- Additional models (SVM, KNN) 
- Feature importance visualization 
- Enhanced cross-validation reporting (per-fold logging) 
- API authentication and rate limiting 
- CI/CD integration 

--- 
## Author 
This project was independently developed as a machine learning experimentation and model comparison framework. AI tools (ChatGPT, GitHub Copilot, ...) were used for debugging assistance and implementation support. 

--- 
## Notes 
The dataset is synthetic but structured to mimic realistic classification problems. This project is intended for learning, experimentation, and demonstrating end-to-end machine learning workflows.