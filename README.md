# ML Model Comparison Pipeline

A lightweight machine learning framework for training, comparing, and tracking classification models using structured pipelines.

This project demonstrates a complete ML workflow including preprocessing, model training, evaluation, and experiment tracking.

---

## Overview

The goal of this project is to build a reusable and modular machine learning pipeline that can:

- Load and preprocess structured tabular data
- Train multiple classification models
- Evaluate model performance consistently
- Track experiments for comparison

The dataset used is a synthetic Titanic-style dataset designed to simulate realistic classification features.

---

## Features

- End-to-end ML pipeline (data → preprocessing → training → evaluation)
- Automatic handling of categorical and numerical features
- Feature preprocessing:
  - OneHot encoding for categorical variables
  - Standard scaling for numerical features
- Model comparison:
  - Logistic Regression
  - Random Forest Classifier
- Experiment tracking system (logs results to CSV)
- Reproducible synthetic dataset generation

---

## Models Used

- Logistic Regression
- Random Forest Classifier

### Evaluation Metric

- Accuracy

---

## Project Structure

data/ \
dataset.csv # Synthetic dataset

src/ \
load_data.py # Data loading utilities \
preprocess.py # Train/test split + validation \
train.py # Model pipelines and training \
evaluate.py # Evaluation metrics \
tracker.py # Experiment logging system \
create_dataset.py # Dataset generator (optional) 

main.py # Pipeline entry point \
experiments.csv # Logged experiment results

---

## Installation & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python main.py
```

Or using Docker

```
docker build -t ml-model-pipeline .
docker run --rm ml-model-pipeline
```

## Example output

```
              precision    recall  f1-score   support

           0       0.67      0.61      0.64        46
           1       0.69      0.74      0.71        54

    accuracy                           0.68       100
   macro avg       0.68      0.67      0.68       100
weighted avg       0.68      0.68      0.68       100

LogisticRegression accuracy: 0.6800
----------------------------------------------------------------------
              precision    recall  f1-score   support

           0       0.48      0.57      0.52        46
           1       0.57      0.48      0.52        54

    accuracy                           0.52       100
   macro avg       0.52      0.52      0.52       100
weighted avg       0.53      0.52      0.52       100

RandomForest accuracy: 0.5200
----------------------------------------------------------------------
Best model: LogisticRegression (accuracy=0.6800)

Sample predictions (1=survived, 0=did not survive):
{'age': 25, 'sex': 'female', 'fare': 80.0, 'class': 'first'} => pred=1 prob=0.812
{'age': 40, 'sex': 'male', 'fare': 15.0, 'class': 'third'} => pred=0 prob=0.224
{'age': 6, 'sex': 'female', 'fare': 30.0, 'class': 'second'} => pred=1 prob=0.675
```

## Author

This project was independently developed as a machine learning learning and experimentation framework.

AI tools (e.g., ChatGPT, GitHub Copilot) were used as coding assistance for debugging and implementation support.

## Notes

Dataset is synthetic but structured to mimic real-world classification problems. 
The project is designed for learning, experimentation, and model comparison workflows.