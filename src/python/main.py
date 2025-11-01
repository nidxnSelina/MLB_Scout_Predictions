import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def load_training_data():
    """
    Load train set

    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(this_dir, "..", "data", "train.csv")
    train_path = os.path.abspath(train_path)
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"[ERROR] could not find file at {train_path}")

    df = pd.read_csv(train_path)
    print(f"[INFO] training data loaded. shape={df.shape}")

    return df


def load_test_data():
    """
    Load test set
    
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(this_dir, "..", "data", "test.csv")
    test_path = os.path.abspath(test_path)
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"[ERROR] could not find file at {test_path}")

    df = pd.read_csv(test_path)
    print(f"[INFO] test data loaded. shape={df.shape}")
    print(f"[INFO] columns: {list(df.columns)}")

    return df


def preprocess_titanic_train(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    """
    Preprocess the train set
    
    """
    # keep id out of features but return it
    passenger_ids = df["PassengerId"].copy() if "PassengerId" in df.columns else None

    # detect train vs test
    is_train = "Survived" in df.columns
    y = df["Survived"].copy() if is_train else None

    # start from a copy so we don't mutate caller's df
    X = df.copy()

    # drop columns we don't want the model to see
    drop_cols = [c for c in ["PassengerId", "Name", "Ticket", "Cabin", "Survived"] if c in X.columns]
    X = X.drop(columns=drop_cols)

    # ---------------- imputations ----------------
    # Age
    if "Age" in X.columns and X["Age"].isna().any():
        X["Age"] = X["Age"].fillna(X["Age"].median())

    # Fare (often missing in test)
    if "Fare" in X.columns and X["Fare"].isna().any():
        X["Fare"] = X["Fare"].fillna(X["Fare"].median())

    # Embarked
    if "Embarked" in X.columns and X["Embarked"].isna().any():
        X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

    # ---------------- encoding ----------------
    # Sex -> binary
    if "Sex" in X.columns:
        X["Sex"] = X["Sex"].map({"male": 0, "female": 1}).astype(int)

    # engineered features
    if "SibSp" in X.columns and "Parch" in X.columns:
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)

    # one-hot Embarked
    if "Embarked" in X.columns:
        X = pd.get_dummies(X, columns=["Embarked"], drop_first=True, dtype=int)

    # final safety net for numerics
    num_cols = X.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # final safety net for non-numerics (should be none, but just in case)
    non_num_cols = X.select_dtypes(exclude=["number"]).columns
    for col in non_num_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].mode()[0])

    return X, y, passenger_ids


def preprocess_titanic_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the test set
    
    """
    # keep PassengerId for later submission
    passenger_ids = df["PassengerId"].copy() if "PassengerId" in df.columns else None

    # make a working copy
    X = df.copy()

    # drop non-feature columns
    drop_cols = [c for c in ["PassengerId", "Name", "Ticket", "Cabin"] if c in X.columns]
    X = X.drop(columns=drop_cols)

    # ---------------- imputations ----------------
    if "Age" in X.columns and X["Age"].isna().any():
        X["Age"] = X["Age"].fillna(X["Age"].median())

    if "Fare" in X.columns and X["Fare"].isna().any():
        X["Fare"] = X["Fare"].fillna(X["Fare"].median())

    if "Embarked" in X.columns and X["Embarked"].isna().any():
        X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

    # ---------------- encoding ----------------
    if "Sex" in X.columns:
        X["Sex"] = X["Sex"].map({"male": 0, "female": 1}).astype(int)

    if "SibSp" in X.columns and "Parch" in X.columns:
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)

    if "Embarked" in X.columns:
        X = pd.get_dummies(X, columns=["Embarked"], drop_first=True, dtype=int)

    # safety net: fill remaining numeric NaNs
    num_cols = X.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    non_num_cols = X.select_dtypes(exclude=["number"]).columns
    for col in non_num_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].mode()[0])

    return X, passenger_ids


def train_titanic_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Train a logistic regression model on cleaned Titanic data.
    
    """
    # scaling + logistic regression
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(X_train, y_train)

    # measure training accuracy
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    print(f"[INFO] Training Accuracy: {train_acc:.4f}")

    return model


def predict_titanic_survival(model: Pipeline, X_test: pd.DataFrame, test_ids: pd.Series) -> pd.DataFrame:
    """
    Predict Titanic survival on the test set and return submission DataFrame.

    """
    # predict
    preds = model.predict(X_test)
    submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Survived": preds.astype(int)
    })

    return submission


def evaluate_test_accuracy(prediction_df: pd.DataFrame) -> float:
    """
    Evaluate the model accuracy on test set by comparing with gender_submission.csv (ground truth)

    """
    # load correct answers
    answer_path = os.path.join(os.path.dirname(__file__), "..", "data", "gender_submission.csv")
    answer_path = os.path.abspath(answer_path)
    answers = pd.read_csv(answer_path)

    # merge on PassengerId
    merged = prediction_df.merge(answers, on="PassengerId", suffixes=("_pred", "_true"))

    return accuracy_score(merged["Survived_true"], merged["Survived_pred"])


if __name__ == "__main__":

    # Load data
    print("-----------------------------Loading and Preprocessing Data-----------------------------")
    train_df = load_training_data()
    test_df = load_test_data() 
    
    # Preprocess data
    X_train, y_train, train_ids = preprocess_titanic_train(train_df)
    X_test, test_ids = preprocess_titanic_test(test_df)
    print("Finished preprocessing data")

    # Train model
    print("-----------------------------Training Model----------------------------")
    model = train_titanic_model(X_train, y_train)
    print("Model: ", model)

    # Predict on test set
    print("---------------------------Predicting on Test Set----------------------")
    prediction_df = predict_titanic_survival(model, X_test, test_ids)
    test_accuracy = evaluate_test_accuracy(prediction_df)
    print(f"[INFO] Test Accuracy: {test_accuracy:.4f}")

    # Save predictions to CSV
    this_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(this_dir, "..", "data", "survival_predictions_python.csv")
    output_path = os.path.abspath(output_path)    
    prediction_df.to_csv(output_path, index=False)
    print(f"[INFO] Predictions saved to: {output_path}")
    print("--------------------------------------------------------\n")



