import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


FEATURE_PATH = "data/features/features.csv"

VSX_PATH = "data/validation/vsx_validation_results.csv"

OUTPUT_DIR = "results/validation"

OUTPUT_CSV = (
    "results/validation/"
    "ranked_variable_candidates.csv"
)


FEATURE_WEIGHTS = {

    # variability amplitude
    "flux_range": 0.20,
    "relative_flux_range": 0.15,
    "flux_std": 0.10,

    # periodic behavior
    "period_power": 0.15,
    "period_confidence": 0.10,
    "num_periods": 0.05,

    # flare behavior
    "flare_count": 0.10,
    "flare_strength": 0.05,

    # eclipse/dip behavior
    "dip_count": 0.05,
    "largest_dip_depth": 0.05
}



def load_data():

    print("Loading data...")

    features = pd.read_csv(
        FEATURE_PATH
    )

    vsx = pd.read_csv(
        VSX_PATH
    )


    print(
        f"Features: {len(features)}"
    )

    print(
        f"Candidates: {len(vsx)}"
    )


    return features, vsx



def normalize_features(df):

    scaler = MinMaxScaler()


    cols = [
        c for c in FEATURE_WEIGHTS
        if c in df.columns
    ]


    scaled = scaler.fit_transform(
        df[cols].fillna(0)
    )


    scaled = pd.DataFrame(
        scaled,
        columns=cols
    )


    return scaled



def calculate_variability_score(df):

    scaled = normalize_features(
        df
    )


    score = 0


    for feature, weight in FEATURE_WEIGHTS.items():

        if feature in scaled:

            score += (
                scaled[feature]
                *
                weight
            )


    return score



def model_score(row):

    score = 0


    confidence = row["confidence"]


    score += confidence * 5


    prediction = row["prediction"]


    if prediction == "flare_variable":

        score += 3


    elif prediction == "periodic_variable":

        score += 3


    elif prediction == "transient_variable":

        score += 2


    elif prediction != "non_variable":

        score += 1


    return score



def rank(df):


    df["variability_score"] = (
        calculate_variability_score(df)
    )


    df["model_score"] = (
        df.apply(
            model_score,
            axis=1
        )
    )


    df["total_score"] = (
        df["variability_score"] * 10
        +
        df["model_score"]
    )


    return df.sort_values(
        "total_score",
        ascending=False
    )



def save_results(df):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    df.to_csv(
        OUTPUT_CSV,
        index=False
    )


    summary = (
        OUTPUT_DIR
        +
        "/variable_candidate_summary.txt"
    )


    with open(summary, "w") as f:

        f.write(
            "Top Variable Candidates\n"
        )

        f.write(
            "=======================\n\n"
        )


        for _, row in df.head(20).iterrows():

            f.write(
                f"TIC {row.tic_id}\n"
            )

            f.write(
                f"Prediction: {row.prediction}\n"
            )

            f.write(
                f"Confidence: {row.confidence}\n"
            )

            f.write(
                f"Score: {row.total_score:.3f}\n\n"
            )


    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print(
        f"Saved: {summary}"
    )



def main():

    features, vsx = load_data()


    df = vsx.merge(
        features,
        on="tic_id",
        how="left"
    )


    ranked = rank(
        df
    )


    save_results(
        ranked
    )


    print()

    print(
        "Top candidates:"
    )


    print(
        ranked[
            [
                "tic_id",
                "prediction",
                "confidence",
                "variability_score",
                "total_score"
            ]
        ]
        .head(10)
    )



if __name__ == "__main__":

    main()