import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightkurve as lk

from astropy.timeseries import LombScargle


CANDIDATE_FILE = (
    "results/validation/"
    "ranked_variable_candidates.csv"
)

OUTPUT_DIR = (
    "results/validation"
)

PERIODOGRAM_DIR = (
    "results/validation/"
    "periodograms"
)

ANALYSIS_FILE = (
    "results/validation/"
    "candidate_analysis.csv"
)

SUMMARY_FILE = (
    "results/validation/"
    "candidate_analysis_summary.txt"
)

TOP_N = 10
MAX_SECTORS = 3

MIN_POINTS = 100


os.makedirs(
    PERIODOGRAM_DIR,
    exist_ok=True
)


def load_candidates():

    print(
        "Loading candidates..."
    )

    df = pd.read_csv(
        CANDIDATE_FILE
    )

    return df.head(
        TOP_N
    )


def extract_flux(lc):

    flux = None


    if hasattr(
        lc,
        "pdcsap_flux"
    ):

        flux = lc.pdcsap_flux


    elif hasattr(
        lc,
        "sap_flux"
    ):

        flux = lc.sap_flux


    elif hasattr(
        lc,
        "flux"
    ):

        flux = lc.flux


    if flux is None:

        raise ValueError(
            "No flux available"
        )


    try:

        flux = flux.value

    except AttributeError:

        pass


    time = lc.time


    try:

        time = time.value

    except AttributeError:

        pass


    flux = np.asarray(
        flux,
        dtype=float
    )

    time = np.asarray(
        time,
        dtype=float
    )


    valid = (
        np.isfinite(time)
        &
        np.isfinite(flux)
    )


    time = time[valid]
    flux = flux[valid]


    if len(flux) < MIN_POINTS:

        raise ValueError(
            "Not enough points"
        )


    median = np.median(
        flux
    )


    flux = (
        flux /
        median
    )


    return time, flux



def download_lightcurve(tic):

    print(
        f"TIC {tic}: searching..."
    )


    search = lk.search_lightcurve(
        f"TIC {tic}",
        mission="TESS"
    )


    if len(search) == 0:

        raise ValueError(
            "No TESS observations"
        )


    search = search[
        :MAX_SECTORS
    ]


    all_time = []
    all_flux = []


    for i, item in enumerate(search):

        print(
            f"TIC {tic}: sector "
            f"{i+1}/{len(search)}"
        )


        lc = item.download()


        if lc is None:

            continue


        try:

            time, flux = (
                extract_flux(
                    lc
                )
            )

            all_time.extend(
                time
            )

            all_flux.extend(
                flux
            )


        except Exception as e:

            print(
                f"Sector skipped: {e}"
            )


    if len(all_flux) < MIN_POINTS:

        raise ValueError(
            "No usable lightcurve"
        )


    order = np.argsort(
        all_time
    )


    return (
        np.asarray(all_time)[order],
        np.asarray(all_flux)[order]
    )



def calculate_features(
    time,
    flux
):

    median = np.median(
        flux
    )


    amplitude = (
        np.max(flux)
        -
        np.min(flux)
    )


    rms = np.sqrt(
        np.mean(
            (
                flux - median
            ) ** 2
        )
    )


    mad = np.median(
        np.abs(
            flux - median
        )
    )


    sigma = (
        1.4826 *
        mad
    )


    if sigma == 0:

        sigma = np.std(
            flux
        )


    if sigma == 0:

        sigma = 1e-8


    z_scores = (
        np.abs(
            flux - median
        )
        /
        sigma
    )


    max_sigma = np.max(
        z_scores
    )


    outliers = np.sum(
        z_scores > 5
    )


    return {
        "amplitude": float(
            amplitude
        ),
        "rms": float(
            rms
        ),
        "mad": float(
            mad
        ),
        "max_sigma": float(
            max_sigma
        ),
        "num_5sigma_outliers": int(
            outliers
        )
    }



def period_analysis(
    tic,
    time,
    flux
):

    frequency, power = (
        LombScargle(
            time,
            flux
        ).autopower(
            minimum_frequency=0.01,
            maximum_frequency=10
        )
    )


    best_frequency = (
        frequency[
            np.argmax(power)
        ]
    )


    best_period = (
        1 /
        best_frequency
    )


    false_alarm = (
        LombScargle(
            time,
            flux
        ).false_alarm_probability(
            np.max(power)
        )
    )


    plt.figure(
        figsize=(8,4)
    )


    plt.plot(
        1 / frequency,
        power,
        linewidth=0.8
    )


    plt.xscale(
        "log"
    )


    plt.xlabel(
        "Period (days)"
    )


    plt.ylabel(
        "Power"
    )


    plt.title(
        f"TIC {tic} Periodogram"
    )


    plt.grid(
        alpha=0.3
    )


    path = os.path.join(
        PERIODOGRAM_DIR,
        f"TIC_{tic}_periodogram.png"
    )


    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200
    )


    plt.close()


    return {
        "best_period_days": float(
            best_period
        ),
        "false_alarm_probability": float(
            false_alarm
        )
    }



def main():

    candidates = load_candidates()


    results = []


    print(
        f"Analyzing {len(candidates)} candidates..."
    )


    for _, row in candidates.iterrows():

        tic = int(
            row["tic_id"]
        )


        print(
            f"\nTIC {tic}"
        )


        try:

            time, flux = (
                download_lightcurve(
                    tic
                )
            )


            features = calculate_features(
                time,
                flux
            )


            periods = period_analysis(
                tic,
                time,
                flux
            )


            result = {
                "tic_id": tic,
                "prediction": row["prediction"],
                "confidence": row["confidence"],
                "total_score": row["total_score"],
                **features,
                **periods
            }


            results.append(
                result
            )


            print(
                "Success"
            )


        except Exception as e:

            print(
                f"Failed: {e}"
            )


            results.append(
                {
                    "tic_id": tic,
                    "prediction": row["prediction"],
                    "confidence": row["confidence"],
                    "total_score": row["total_score"],
                    "error": str(e)
                }
            )


    df = pd.DataFrame(
        results
    )


    df.to_csv(
        ANALYSIS_FILE,
        index=False
    )


    with open(
        SUMMARY_FILE,
        "w"
    ) as f:

        f.write(
            "Candidate Lightcurve Analysis\n"
        )

        f.write(
            "============================\n\n"
        )


        for _, row in df.iterrows():

            f.write(
                f"TIC {row['tic_id']}\n"
            )

            f.write(
                f"Prediction: {row['prediction']}\n"
            )

            if "amplitude" in row:

                f.write(
                    f"Amplitude: "
                    f"{row['amplitude']:.5f}\n"
                )

                f.write(
                    f"RMS: "
                    f"{row['rms']:.5f}\n"
                )

                f.write(
                    f"Period: "
                    f"{row['best_period_days']:.5f} days\n"
                )


            f.write(
                "\n"
            )


    print(
        "\nFinished."
    )

    print(
        f"Saved: {ANALYSIS_FILE}"
    )

    print(
        f"Saved: {SUMMARY_FILE}"
    )


if __name__ == "__main__":

    main()