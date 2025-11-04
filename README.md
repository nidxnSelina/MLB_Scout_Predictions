# MLDS 400 HW3: Titanic Disaster Survival Prediction

This project builds a binary classification model to predict passenger survival on the Titanic using the Kaggle Titanic dataset ([Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/code)). Logistic regression model is used. The scripts perform data cleaning, train a logistic regression model, report training accuracy, and save predictions on the test set to a CSV file. This repository supports running both locally and on Docker using either Python or R.

## Prerequisites
- Download data from Kaggle: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/code)
- Load the following data files to src/data/: train.csv, test.csv, and gender_submission.csv.
- Install Docker (optional — only needed if running via Docker)

## Run locally in Python
### Install dependencies
```bash
pip install -r requirements.txt
```

### Run model training and prediction
```bash
python src/python/run.py
```

Output:
- training accuracy
- test accuracy (comparing predictions with gender_submission.csv)
- predictions saved to src/data/survival_predictions_python.csv

## Run locally in R
### Install dependencies
```bash
Rscript src/R/install_packages.R
```

### Run model training and prediction
```bash
Rscript src/R/run.R
```

Output:
- training accuracy
- test accuracy (comparing predictions with gender_submission.csv)
- predictions saved to src/data/survival_predictions_r.csv

## Run via Docker in Python
### Build Docker image
```bash
docker build -t titanic-python .
```

### Run container
```bash
docker run --rm -v "$PWD":/app titanic-python
```

Output:
- training accuracy
- test accuracy (comparing predictions with gender_submission.csv)
- predictions saved to src/data/survival_predictions_python.csv


## Run via Docker in R
### Build Docker image
```bash
docker build -t titanic-r -f src/R/Dockerfile .
```

### Run container
```bash
docker run --rm -v "$PWD/src/data":/app/data titanic-r
```

Output:
- training accuracy
- test accuracy (comparing predictions with gender_submission.csv)
- predictions saved to src/data/survival_predictions_r.csv

