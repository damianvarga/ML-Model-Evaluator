# Titanic Survival Predictor

## Overview
- This project trains simple ML models to predict Titanic survival using a synthetic dataset shaped like the classic Titanic features: age, sex, fare, class, survived.

## Project structure
- data/dataset.csv: Input CSV.
- src/
  - load_data.py: CSV loader.
  - preprocess.py: Train/test split with target validation and stratification.
  - train.py: Pipelines with preprocessing (OneHot + StandardScaler) and two models: LogisticRegression and RandomForest.
  - evaluate.py: Accuracy metric.
  - tracker.py: Appends experiment results to experiments.csv.
- main.py: Orchestrates training/evaluation, prints sample predictions.

## Setup
1) Python 3.10+ recommended.
2) Install dependencies
   pip install -r requirements.txt

## Run
   python main.py

## Notes
- The dataset here is synthetic but uses plausible rules (see src/create_dataset.py if you want to regenerate). Columns required: age (int), sex (male/female), fare (float), class (first/second/third), survived (0/1).
- Pipelines handle categorical encoding and numeric scaling. The app logs results to experiments.csv and prints example predictions.

## About
This project was developed by me, with significant assistance from AI tools such as GitHub Copilot, Qodo and ChatGPT.
