import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FEATURE_PATH = "data/features/labeled_features.csv"
MODEL_PATH = "models/stellar_classifier.pkl"

RESULT_DIR = "results"


def load_data():
    df = pd.read_csv(FEATURE_PATH)

    X = df.drop(
        columns=[
            "tic_id",
            "classification"
        ]
    )

    y = df["classification"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return X, y


def analyze_feature_importance(model, features):
    classifier = model.named_steps["classifier"]

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": classifier.feature_importances_
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print()
    print("Feature importance")
    print(importance)

    importance.to_csv(
        os.path.join(
            RESULT_DIR,
            "feature_importance.csv"
        ),
        index=False
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance["feature"],
        importance["importance"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel(
        "Importance"
    )

    plt.title(
        "Random Forest Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "feature_importance.png"
        ),
        dpi=300
    )

    plt.close()


def create_confusion_matrix(model, X_test, y_test):
    predictions = model.predict(X_test)

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions
    )

    display.figure_.savefig(
        os.path.join(
            RESULT_DIR,
            "confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close()


def correlation_analysis(df):
    features = df.drop(
        columns=[
            "tic_id",
            "classification"
        ]
    )

    correlation = features.corr()

    correlation.to_csv(
        os.path.join(
            RESULT_DIR,
            "feature_correlation.csv"
        )
    )


def main():

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    X, y = load_data()

    model = joblib.load(
        MODEL_PATH
    )

    analyze_feature_importance(
        model,
        X.columns
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    create_confusion_matrix(
        model,
        X_test,
        y_test
    )

    df = pd.read_csv(
        FEATURE_PATH
    )

    correlation_analysis(
        df
    )

    print()
    print("Analysis complete")


if __name__ == "__main__":
    main()