"""
StellarScope Feature Extraction Pipeline

Converts processed TESS light curves into
astronomy-inspired numerical features.

Input:
    data/processed/TIC_<id>.csv

Output:
    data/features/features.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle


PROCESSED_DIR = Path("data/processed")
FEATURE_DIR = Path("data/features")


def extract_variability_features(flux):
    """
    Basic brightness variability statistics.
    """

    return {
        "flux_std": np.std(flux),
        "flux_range": np.max(flux) - np.min(flux),
        "flux_skew": skew(flux),
        "flux_kurtosis": kurtosis(flux),
    }


def extract_period_features(time, flux):
    """
    Detect dominant periodic behavior using
    Lomb-Scargle periodogram.
    """

    frequency, power = LombScargle(
        time,
        flux
    ).autopower()

    best_frequency = frequency[np.argmax(power)]

    dominant_period = 1 / best_frequency

    period_power = np.max(power)

    return {
        "dominant_period": dominant_period,
        "period_power": period_power,
    }


def count_events(mask, min_points=5):
    """
    Count groups of consecutive True values.

    Requires minimum duration to avoid
    counting random noise fluctuations.
    """

    events = 0
    current_length = 0

    for value in mask:

        if value:
            current_length += 1

        else:
            if current_length >= min_points:
                events += 1

            current_length = 0


    # Catch event at end of array
    if current_length >= min_points:
        events += 1


    return events

def extract_event_features(flux):
    """
    Detect unusual brightness events.

    Groups consecutive points into events.
    """

    median = np.median(flux)
    std = np.std(flux)

    deviation = (flux - median) / std


    flare_mask = deviation > 3
    dip_mask = deviation < -3


    return {
        "flare_count": count_events(flare_mask),
        "dip_count": count_events(dip_mask),
    }

def extract_features(filepath):
    """
    Extract all features from one light curve.
    """

    df = pd.read_csv(filepath)

    time = df["time"].values
    flux = df["flux"].values


    features = {}

    features.update(
        extract_variability_features(flux)
    )

    features.update(
        extract_period_features(
            time,
            flux
        )
    )

    features.update(
        extract_event_features(flux)
    )


    features["tic_id"] = (
        filepath.stem.replace(
            "TIC_",
            ""
        )
    )


    return features



def main():

    FEATURE_DIR.mkdir(
        exist_ok=True
    )

    all_features = []


    files = list(
        PROCESSED_DIR.glob(
            "TIC_*.csv"
        )
    )


    print(
        f"Processing {len(files)} light curves..."
    )


    for file in files:

        print(
            f"Extracting: {file.name}"
        )

        features = extract_features(
            file
        )

        all_features.append(
            features
        )


    feature_df = pd.DataFrame(
        all_features
    )


    output = FEATURE_DIR / "features.csv"


    feature_df.to_csv(
        output,
        index=False
    )


    print(
        f"\nSaved features: {output}"
    )

    print(
        feature_df
    )


if __name__ == "__main__":
    main()