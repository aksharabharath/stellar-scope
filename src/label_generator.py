import os
import pandas as pd

INPUT_PATH = "data/features/features.csv"
OUTPUT_PATH = "data/features/labeled_features.csv"


def classify_star(row):

    flux_std = row["flux_std"]
    relative_flux_range = row["relative_flux_range"]
    coefficient_variation = row["coefficient_of_variation"]

    period_power = row["period_power"]
    period_confidence = row["period_confidence"]

    flare_count = row["flare_count"]
    flare_frequency = row["flare_frequency"]
    flare_strength = row["flare_strength"]
    largest_flare = row["largest_flare"]

    dip_count = row["dip_count"]
    dip_depth = row["largest_dip_depth"]


    if (
        flare_count >= 10
        and (
            flare_frequency > 0.001
            or flare_strength > 80
            or largest_flare > 10
        )
        and row["flux_skew"] > 5
    ):
        return "flare_variable"


    if (
    period_power > 0.01
    and period_confidence > 0.0001
):
        return "periodic_variable"


    if (
        (
            relative_flux_range > 50
            and coefficient_variation > 0.5
        )
        or (
            flux_std > 1
            and row["flux_kurtosis"] > 1000
        )
    ):
        return "transient_variable"


    return "non_variable"


def generate_labels():

    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded {len(df)} stars")


    df["classification"] = df.apply(
        classify_star,
        axis=1
    )


    os.makedirs(
        "data/features",
        exist_ok=True
    )


    df.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print()
    print("Saved:", OUTPUT_PATH)
    print()

    print(
        df["classification"].value_counts()
    )


if __name__ == "__main__":
    generate_labels()