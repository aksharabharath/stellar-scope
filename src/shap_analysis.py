import os
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

MODEL_PATH = "models/stellar_classifier.pkl"
DATA_PATH = "data/features/labeled_features.csv"
OUTPUT_DIR = "results/shap"


def load_data():
    df = pd.read_csv(DATA_PATH)

    feature_columns = [
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

    X = df[feature_columns]

    return X, feature_columns


def create_shap_plots(model, X, feature_names):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Preparing model for SHAP...")

    classifier = model.named_steps["classifier"]

    print("Calculating SHAP values...")

    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_for_plot = shap_values[1]
    else:
        if len(shap_values.shape) == 3:
            shap_for_plot = shap_values[:, :, 1]
        else:
            shap_for_plot = shap_values
    print("SHAP shape:", shap_for_plot.shape)

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": abs(shap_for_plot).mean(axis=0)
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    print("\nSHAP Feature Importance")
    print(importance)

    importance.to_csv(
        f"{OUTPUT_DIR}/shap_feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_for_plot,
        X,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/summary_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {OUTPUT_DIR}/summary_plot.png")


def main():
    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Loading features...")
    X, feature_names = load_data()

    create_shap_plots(
        model,
        X,
        feature_names
    )

    print("SHAP analysis complete")


if __name__ == "__main__":
    main()