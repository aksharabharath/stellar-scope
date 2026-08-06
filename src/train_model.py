import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


FEATURE_PATH = "data/features/labeled_features.csv"
MODEL_DIR = "models"
RESULT_DIR = "results"


def load_data():
    df = pd.read_csv(FEATURE_PATH)

    print(f"Loaded {len(df)} stars")

    X = df.drop(columns=["tic_id", "classification"])
    y = df["classification"]

    return X, y


def train_models(X_train, X_test, y_train, y_test):
    models = {
        "random_forest": Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight="balanced"
                    )
                )
            ]
        ),
        "logistic_regression": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced"
                    )
                )
            ]
        )
    }

    results = {}

    for name, model in models.items():
        print()
        print(f"Training {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        report = classification_report(
            y_test,
            predictions
        )

        matrix = confusion_matrix(
            y_test,
            predictions
        )

        print(f"Accuracy: {accuracy:.3f}")
        print(report)

        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "report": report,
            "matrix": matrix
        }

    return results


def save_results(results):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    best_name = max(
        results,
        key=lambda x: results[x]["accuracy"]
    )

    best_model = results[best_name]["model"]

    model_path = os.path.join(
        MODEL_DIR,
        "stellar_classifier.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print()
    print(f"Best model: {best_name}")
    print(f"Saved model: {model_path}")

    with open(
        os.path.join(
            RESULT_DIR,
            "classification_report.txt"
        ),
        "w"
    ) as f:
        for name, result in results.items():
            f.write(f"{name}\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n\n")
            f.write(result["report"])
            f.write("\n\n")

    pd.DataFrame(
        results[best_name]["matrix"]
    ).to_csv(
        os.path.join(
            RESULT_DIR,
            "confusion_matrix.csv"
        ),
        index=False
    )


def main():
    X, y = load_data()

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    joblib.dump(
        encoder,
        os.path.join(
            MODEL_DIR,
            "label_encoder.pkl"
        )
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    results = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    save_results(results)


if __name__ == "__main__":
    main()