import os
import pandas as pd


INPUT_PATH = "data/features/features.csv"
OUTPUT_PATH = "data/features/labeled_features.csv"


def classify_star(row):
    flare_count = row["flare_count"]
    largest_flare = row["largest_flare"]
    period_power = row["period_power"]
    dip_count = row["dip_count"]
    largest_dip_depth = row["largest_dip_depth"]

    if period_power >= 0.3:
        return "variable_star"

    if (
        flare_count >= 20
        and largest_flare > 0.01
    ):
        return "flare_variable"

    if (
        dip_count >= 10
        and largest_dip_depth > 0.005
    ):
        return "variable_star"

    return "quiet_star"


def generate_labels():
    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded {len(df)} stars")

    df["classification"] = df.apply(classify_star, axis=1)

    os.makedirs("data/features", exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print()
    print("Saved:", OUTPUT_PATH)
    print()
    print(df["classification"].value_counts())


if __name__ == "__main__":
    generate_labels()