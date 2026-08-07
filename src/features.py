import pandas as pd
import numpy as np

from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle


INPUT_DIR = Path("data/processed")
OUTPUT = Path("data/features/features.csv")


def extract_features(file):

    df = pd.read_csv(file)

    time = df["time"].values
    flux = df["flux"].values

    median_flux = np.median(flux)

    flux = flux / median_flux

    features = {}

    features["flux_std"] = np.std(flux)

    features["flux_range"] = (
        np.max(flux) -
        np.min(flux)
    )

    features["relative_flux_range"] = (
        np.max(flux) -
        np.min(flux)
    ) / np.abs(np.mean(flux))


    features["coefficient_of_variation"] = (
        np.std(flux) /
        np.abs(np.mean(flux))
    )


    features["flux_mad"] = np.median(
        np.abs(
            flux -
            np.median(flux)
        )
    )

    features["flux_skew"] = skew(flux)

    features["flux_kurtosis"] = kurtosis(flux)


    try:

        frequency, power = LombScargle(
            time,
            flux
        ).autopower()


        best_index = np.argmax(power)


        features["dominant_period"] = (
            1 /
            frequency[best_index]
        )


        features["period_power"] = (
            power[best_index]
        )


        total_power = np.sum(power)


        if total_power > 0:

            features["period_confidence"] = (
                power[best_index] /
                total_power
            )

        else:

            features["period_confidence"] = 0


        peaks, _ = find_peaks(
            power,
            height=np.percentile(
                power,
                90
            )
        )

        features["num_periods"] = len(peaks)


    except Exception:

        features["dominant_period"] = np.nan

        features["period_power"] = 0

        features["period_confidence"] = 0

        features["num_periods"] = 0



    flare_threshold = (
        np.mean(flux)
        +
        3 *
        np.std(flux)
    )


    flare_peaks, _ = find_peaks(
        flux,
        height=flare_threshold
    )


    features["flare_count"] = (
        len(flare_peaks)
    )


    features["flare_frequency"] = (
        len(flare_peaks) /
        len(flux)
    )


    if len(flare_peaks) > 0:

        features["largest_flare"] = (
            np.max(flux[flare_peaks])
            -
            np.median(flux)
        )

        features["flare_strength"] = (
            features["largest_flare"] /
            np.std(flux)
        )

    else:

        features["largest_flare"] = 0

        features["flare_strength"] = 0



    dip_threshold = (
        np.mean(flux)
        -
        3 *
        np.std(flux)
    )


    dip_peaks, _ = find_peaks(
        -flux,
        height=-dip_threshold
    )


    features["dip_count"] = (
        len(dip_peaks)
    )


    if len(dip_peaks) > 0:

        features["largest_dip_depth"] = (
            np.median(flux)
            -
            np.min(flux[dip_peaks])
        )

    else:

        features["largest_dip_depth"] = 0



    features["tic_id"] = int(
        file.stem.replace(
            "TIC_",
            ""
        )
    )


    return features



def main():

    files = sorted(
        INPUT_DIR.glob(
            "TIC_*.csv"
        )
    )


    print(
        f"Processing {len(files)} stars..."
    )


    results = []


    for index, file in enumerate(files):

        print(
            f"[{index + 1}/{len(files)}] Extracting {file.name}"
        )


        try:

            results.append(
                extract_features(file)
            )

        except Exception as error:

            print(
                f"Failed: {file.name} | {error}"
            )


    feature_df = pd.DataFrame(
        results
    )


    OUTPUT.parent.mkdir(
        exist_ok=True
    )


    feature_df.to_csv(
        OUTPUT,
        index=False
    )


    print(
        f"Saved features: {OUTPUT}"
    )


    print(
        feature_df.head()
    )


if __name__ == "__main__":
    main()