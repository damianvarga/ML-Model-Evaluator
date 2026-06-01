# ML Model Comparison Pipeline

A lightweight machine learning framework for training, comparing, and tracking classification models with a REST API for serving predictions.

This project demonstrates a complete ML workflow including preprocessing, model training, evaluation, cross-validation, experiment tracking, and API deployment with FastAPI.

---

## Overview

The goal of this project is to build a reusable and modular machine learning pipeline that can:

- Load and preprocess structured tabular data
- Train multiple classification models
- Evaluate model performance consistently
- Perform cross-validation for robust evaluation
- Track experiments for comparison
- Serve predictions via a FastAPI REST API

The dataset used is a synthetic Titanic-style dataset designed to simulate realistic classification features.

---

## Features

- End-to-end ML pipeline (data → preprocessing → training → evaluation → API)
- Automatic handling of categorical and numerical features
- Feature preprocessing:
  - OneHot encoding for categorical variables
  - Standard scaling for numerical features
- Model comparison:
  - Logistic Regression
  - Random Forest Classifier
- Classification report generation:
  - Precision
  - Recall
  - F1-score
  - Accuracy
- Cross-validation support (K-Fold CV)
- Hyperparameter tuning with GridSearchCV
- Experiment tracking system (logs results to CSV + MLflow)
- Best model persistence with joblib
- REST API built with FastAPI:
  - `GET /` — health check
  - `POST /predict` — predict survival for a passenger
- Reproducible synthetic dataset generation
- Docker support for reproducible environments

---

## Models Used

- Logistic Regression
- Random Forest Classifier

---

## Evaluation Metrics

- Accuracy (test set)
- Cross-validation mean accuracy (cv_mean)
- Cross-validation standard deviation (cv_std)
- Precision
- Recall
- F1-score

---

## Cross-Validation

This project uses **K-Fold Cross-Validation (default: 5 folds)** to improve evaluation reliability.

Instead of relying on a single train/test split, models are evaluated across multiple folds to measure:

- General performance (cv_mean)
- Stability across datasets (cv_std)

---

## Project Structure

```text
data/
└── dataset.csv                 # Synthetic dataset

models/
└── best_model.pkl              # Trained best model (saved by joblib)

src/
├── api.py                      # FastAPI app (GET /, POST /predict)
├── create_dataset.py           # Dataset generator (optional)
├── evaluate.py                 # Evaluation metrics
├── load_data.py                # Data loading utilities
├── preprocess.py               # Train/test split + validation
├── tracker.py                  # Experiment logging system (CSV + MLflow)
└── train.py                    # Model pipelines and training

docker-entrypoint.py            # Docker entrypoint (trains model if missing, then starts API)
main.py                         # Pipeline entry point (train, evaluate, select best, save model)
experiments.csv                 # Logged experiment results
Dockerfile                      # Docker setup
requirements.txt                # Project dependencies
README.md                       # Documentation
```

---

## Installation & Usage

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd ml-model-comparison-pipeline
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train models and save the best one

```bash
python main.py
```

This trains both models, evaluates them, selects the best, and saves it to `models/best_model.pkl`.

### 4. Run the API

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

### `GET /`

Returns a health check message.

**Response:**
```json
{"message": "ML Model API running"}
```

### `POST /predict`

Predicts survival for a passenger based on their features.

**Request body:**
```json
{
  "age": 25,
  "sex": "female",
  "fare": 80.0,
  "class_name": "first"
}
```

**Response:**
```json
{
  "prediction": 1,
  "survival_probability": 0.7647
}
```

`prediction` is `1` (survived) or `0` (did not survive). `survival_probability` is the model's probability estimate for the positive class.

---

## Running with Docker

The Docker container automatically trains the model (if not already saved) and starts the API on port 8000.

### Build the Docker image

```bash
docker build -t ml-model-pipeline .
```

### Run the container

```bash
docker run -p 8000:8000 --rm ml-model-pipeline
```

The API will be available at `http://localhost:8000`.

---

## Example Output

```text
Fitting 5 folds for each of 16 candidates, totalling 80 fits
Best parameters: {'model__C': 0.1, 'model__l1_ratio': 0, 'model__max_iter': 1000, 'model__solver': 'lbfgs'}
Best CV score: 0.6650

Fitting 5 folds for each of 24 candidates, totalling 120 fits
Best parameters: {'model__max_depth': 5, 'model__min_samples_leaf': 2, 'model__min_samples_split': 5, 'model__n_estimators': 100}
Best CV score: 0.6550

              precision    recall  f1-score   support

           0       0.73      0.59      0.65        46
           1       0.70      0.81      0.75        54

    accuracy                           0.71       100
   macro avg       0.71      0.70      0.70       100
weighted avg       0.71      0.71      0.71       100

Logged LogisticRegression | acc=0.7100 | CV=0.6650 ± 0.0429
LogisticRegression accuracy: 0.7100
----------------------------------------------------------------------
              precision    recall  f1-score   support

           0       0.59      0.59      0.59        46
           1       0.65      0.65      0.65        54

    accuracy                           0.62       100
   macro avg       0.62      0.62      0.62       100
weighted avg       0.62      0.62      0.62       100

Logged RandomForest | acc=0.6200 | CV=0.6550 ± 0.0430
RandomForest accuracy: 0.6200
----------------------------------------------------------------------
Best model: LogisticRegression (accuracy=0.7100)

Sample predictions (1=survived, 0=did not survive):
{'age': 25, 'sex': 'female', 'fare': 80.0, 'class': 'first'} => pred=1 prob=0.764
{'age': 40, 'sex': 'male', 'fare': 15.0, 'class': 'third'} => pred=0 prob=0.290
{'age': 6, 'sex': 'female', 'fare': 30.0, 'class': 'second'} => pred=1 prob=0.631
```

---

## Future Improvements

- Additional models (XGBoost, SVM, KNN)
- Feature importance visualization
- Unit testing with pytest
- Enhanced cross-validation reporting (per-fold logging)
- API authentication and rate limiting
- CI/CD integration

---

## Author

This project was independently developed as a machine learning experimentation and model comparison framework.

AI tools (e.g., ChatGPT and GitHub Copilot) were used for debugging assistance and implementation support.

---

## Notes

The dataset is synthetic but structured to mimic realistic classification problems.

This project is intended for learning, experimentation, and demonstrating end-to-end machine learning workflows.