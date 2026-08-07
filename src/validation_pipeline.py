import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from astroquery.mast import Catalogs

MODEL_PATH = "models/stellar_classifier.pkl"
FEATURE_PATH = "data/features/features.csv"

OUTPUT_DATA = "data/validation/validation_results.csv"
OUTPUT_DIR = "results/validation"

FEATURE_COLUMNS = [
    "flux_std",
    "flux_range",
    "flux_mad",
    "flux_skew",
    "flux_kurtosis",
    "dominant_period",
    "period_power",
    "num_periods",
    "flare_count",
    "largest_flare",
    "dip_count",
    "largest_dip_depth"
]


VAR_TYPE_MAP = {
    "FLARE": "flare_variable",
    "BY": "flare_variable",
    "BYDRA": "flare_variable",
    "ROT": "variable_star",
    "ROTATION": "variable_star",
    "ECL": "variable_star",
    "EB": "variable_star",
    "EA": "variable_star",
    "EW": "variable_star",
    "RR": "variable_star",
    "RRLYR": "variable_star",
    "CEP": "variable_star"
}


def load_model_data():
    saved = joblib.load(MODEL_PATH)

    if isinstance(saved, dict):
        model = saved["model"]
        encoder = saved["encoder"]
    else:
        model = saved
        encoder = joblib.load(
            "models/label_encoder.pkl"
        )

    df = pd.read_csv(FEATURE_PATH)

    X = df[FEATURE_COLUMNS]

    return model, encoder, df, X


def get_catalog_classification(tic_id):
    try:
        result = Catalogs.query_criteria(
            ID=int(tic_id),
            catalog="Tic"
        )

        if len(result) == 0:
            return "unknown"

        row = result[0]

        if "varType" not in row.colnames:
            return "unknown"

        var_type = str(row["varType"])

        if var_type == "nan":
            return "quiet_star"

        var_type = var_type.upper()

        for key, value in VAR_TYPE_MAP.items():
            if key in var_type:
                return value

        return "quiet_star"

    except Exception:
        return "unknown"


def create_predictions(model, encoder, df, X):
    predictions = model.predict(X)

    predictions = encoder.inverse_transform(
        predictions
    )

    output = df[["tic_id"]].copy()

    output["prediction"] = predictions

    return output


def add_catalog_labels(df):
    print("Querying TIC variability classifications...")

    labels = []

    for i, tic in enumerate(df["tic_id"]):
        print(
            f"[{i + 1}/{len(df)}] TIC {tic}"
        )

        labels.append(
            get_catalog_classification(tic)
        )

    df["catalog_classification"] = labels

    return df


def evaluate(df):
    valid = df[
        df["catalog_classification"] != "unknown"
    ]

    if len(valid) == 0:
        print("No catalog variability labels found")
        return

    report = classification_report(
        valid["catalog_classification"],
        valid["prediction"],
        zero_division=0
    )

    print(report)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    with open(
        f"{OUTPUT_DIR}/metrics.txt",
        "w"
    ) as f:
        f.write(report)

    cm = confusion_matrix(
        valid["catalog_classification"],
        valid["prediction"]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    display.plot()

    plt.savefig(
        f"{OUTPUT_DIR}/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Validated {len(valid)} stars"
    )


def main():
    model, encoder, df, X = load_model_data()

    results = create_predictions(
        model,
        encoder,
        df,
        X
    )

    results = add_catalog_labels(
        results
    )

    os.makedirs(
        "data/validation",
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_DATA,
        index=False
    )

    print(
        f"Saved: {OUTPUT_DATA}"
    )

    evaluate(results)


if __name__ == "__main__":
    main()