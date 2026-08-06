import os
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder


MODEL_PATH = "models/stellar_classifier.pkl"
FEATURE_PATH = "data/features/features.csv"
PROCESSED_DIR = "data/processed"

RESULT_DIR = "results/predictions"


def load_model():
    model = joblib.load(
        MODEL_PATH
    )

    return model


def load_features(tic_id):
    df = pd.read_csv(
        FEATURE_PATH
    )

    row = df[
        df["tic_id"] == int(tic_id)
    ]

    if len(row) == 0:
        raise ValueError(
            "No features found for this TIC ID"
        )

    X = row.drop(
        columns=[
            "tic_id"
        ]
    )

    return X


def load_light_curve(tic_id):
    path = os.path.join(
        PROCESSED_DIR,
        f"TIC_{tic_id}.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No light curve found: {path}"
        )

    return pd.read_csv(
        path
    )


def decode_prediction(value):
    labels = {
        0: "flare_variable",
        1: "quiet_star",
        2: "variable_star"
    }

    return labels[value]


def plot_light_curve(df, tic_id, prediction):
    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 4)
    )

    plt.plot(
        df["time"],
        df["flux"],
        linewidth=0.5
    )

    plt.xlabel(
        "Time (BTJD)"
    )

    plt.ylabel(
        "Normalized Flux"
    )

    plt.title(
        f"TIC {tic_id}: {prediction}"
    )

    plt.tight_layout()

    output = os.path.join(
        RESULT_DIR,
        f"TIC_{tic_id}.png"
    )

    plt.savefig(
        output,
        dpi=300
    )

    plt.close()

    print(
        f"Saved plot: {output}"
    )


def main():

    if len(sys.argv) < 2:
        print(
            "Usage: python src/predict_star.py TIC_ID"
        )
        return

    tic_id = sys.argv[1]

    model = load_model()

    features = load_features(
        tic_id
    )

    prediction = model.predict(
        features
    )[0]

    probabilities = model.predict_proba(
        features
    )[0]

    label = decode_prediction(
        prediction
    )

    confidence = probabilities[prediction]

    print()
    print(
        f"TIC {tic_id}"
    )

    print(
        f"Prediction: {label}"
    )

    print(
        f"Confidence: {confidence:.3f}"
    )

    print()
    print(
        "Feature values:"
    )

    print(
        features.T
    )

    light_curve = load_light_curve(
        tic_id
    )

    plot_light_curve(
        light_curve,
        tic_id,
        label
    )


if __name__ == "__main__":
    main()