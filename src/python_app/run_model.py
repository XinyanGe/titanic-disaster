"""Titanic survival prediction with logistic regression (Python)."""
import os
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_DIR = os.environ.get(
    "DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
)
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
LABELS_PATH = os.path.join(DATA_DIR, "gender_submission.csv")
OUT_PATH = os.path.join(DATA_DIR, "predictions_python.csv")

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
            "Title", "FamilySize", "IsAlone"]


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def extract_title(names):
    """Pull the honorific (Mr, Mrs, Miss, ...) out of the Name column."""
    title = names.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")


def fit_fill_values(df):
    """Imputation values learned on the TRAINING set only (avoids leakage)."""
    titles = extract_title(df["Name"])
    return {
        "age_by_title": df["Age"].groupby(titles).median().to_dict(),
        "age_overall": df["Age"].median(),
        "fare": df["Fare"].median(),
        "embarked": df["Embarked"].mode()[0],
    }


def preprocess(df, fills, train_columns=None):
    """Apply every cleaning / feature step, printing each change."""
    df = df.copy()

    df["Title"] = extract_title(df["Name"])
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    print("[change] Added 'Title' (from Name), 'FamilySize' (SibSp+Parch+1), 'IsAlone'")
    print("         Title counts:", df["Title"].value_counts().to_dict())

    n_age = int(df["Age"].isna().sum())
    df["Age"] = df["Age"].fillna(df["Title"].map(fills["age_by_title"]))
    df["Age"] = df["Age"].fillna(fills["age_overall"])
    print(f"[change] Filled {n_age} missing Age values with the training median Age per Title")

    n_fare = int(df["Fare"].isna().sum())
    df["Fare"] = df["Fare"].fillna(fills["fare"])
    print(f"[change] Filled {n_fare} missing Fare values with training median {fills['fare']:.2f}")

    n_emb = int(df["Embarked"].isna().sum())
    df["Embarked"] = df["Embarked"].fillna(fills["embarked"])
    print(f"[change] Filled {n_emb} missing Embarked values with training mode '{fills['embarked']}'")

    df["Sex"] = (df["Sex"] == "female").astype(int)
    print("[change] Encoded Sex: female=1, male=0")

    print("[change] Dropped unused columns: PassengerId, Name, Ticket, Cabin")
    X = pd.get_dummies(df[FEATURES], columns=["Pclass", "Embarked", "Title"],
                       drop_first=True, dtype=int)
    print("[change] One-hot encoded Pclass, Embarked, Title (drop_first=True)")

    if train_columns is not None:
        X = X.reindex(columns=train_columns, fill_value=0)
        print("[change] Aligned test columns to match the training feature set")
    return X


def main():
    banner("STEP 1: Load training data")
    if not os.path.exists(TRAIN_PATH):
        sys.exit(f"ERROR: {TRAIN_PATH} not found. See README for how to download the data.")
    train = pd.read_csv(TRAIN_PATH)
    print(f"Loaded train.csv with shape {train.shape}")
    print(train.head())

    banner("STEP 2: Explore the training data")
    print("Missing values per column:\n" + train.isna().sum().to_string())
    print(f"\nOverall survival rate: {train['Survived'].mean():.3f}")
    print("\nSurvival rate by Sex:\n" + train.groupby("Sex")["Survived"].mean().round(3).to_string())
    print("\nSurvival rate by Pclass:\n" + train.groupby("Pclass")["Survived"].mean().round(3).to_string())

    banner("STEP 3: Clean and engineer features (training set)")
    fills = fit_fill_values(train)
    X = preprocess(train, fills)
    y = train["Survived"]
    print(f"\nFinal training feature matrix shape: {X.shape}")
    print("Features used:", list(X.columns))

    banner("STEP 4: Fit logistic regression")
    model = LogisticRegression(max_iter=5000)
    model.fit(X, y)
    print("Model fitted. Coefficients (sorted):")
    print(pd.Series(model.coef_[0], index=X.columns).sort_values().round(3).to_string())

    banner("STEP 5: Accuracy on the training set")
    train_acc = accuracy_score(y, model.predict(X))
    print(f"Training accuracy: {train_acc:.4f}")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    val_model = LogisticRegression(max_iter=5000).fit(X_tr, y_tr)
    val_acc = accuracy_score(y_val, val_model.predict(X_val))
    print(f"Hold-out validation accuracy (80/20 split of train.csv): {val_acc:.4f}")

    banner("STEP 6: Load test data and predict")
    if not os.path.exists(TEST_PATH):
        sys.exit(f"ERROR: {TEST_PATH} not found. See README for how to download the data.")
    test = pd.read_csv(TEST_PATH)
    print(f"Loaded test.csv with shape {test.shape}")
    X_test = preprocess(test, fills, train_columns=X.columns)
    test_pred = model.predict(X_test)
    print(f"Predicted survival rate on test set: {test_pred.mean():.3f}")
    out = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred})
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved predictions to {OUT_PATH}")

    banner("STEP 7: Accuracy on the test set")
    if os.path.exists(LABELS_PATH):
        ref = pd.read_csv(LABELS_PATH)
        merged = out.merge(ref, on="PassengerId", suffixes=("_pred", "_ref"))
        acc = accuracy_score(merged["Survived_ref"], merged["Survived_pred"])
        print(f"Accuracy vs gender_submission.csv: {acc:.4f}")
        print("NOTE: Kaggle does not publish the true labels for test.csv. "
              "gender_submission.csv is Kaggle's sample baseline (every female survives), "
              "so this number is agreement with that baseline, not true accuracy. "
              "Upload predictions_python.csv to Kaggle to get the true test score.")
    else:
        print("gender_submission.csv not found, so no test labels are available. "
              "Upload predictions_python.csv to Kaggle to get the true test score.")

    banner("DONE")


if __name__ == "__main__":
    main()
