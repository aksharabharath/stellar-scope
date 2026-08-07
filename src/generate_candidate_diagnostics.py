import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightkurve as lk
from astropy.timeseries import LombScargle


INPUT_FILE = (
    "results/validation/"
    "candidate_morphology_classification.csv"
)
OUTPUT_DIR = (
    "results/validation/"
    "diagnostics"
)

MAX_SECTORS = 3
MIN_POINTS = 50


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def extract_flux(lc):

    flux = None

    candidates = [
        "pdcsap_flux",
        "sap_flux",
        "flux"
    ]

    for name in candidates:

        if hasattr(lc, name):

            try:

                flux = getattr(
                    lc,
                    name
                )

                if flux is not None:
                    break

            except Exception:
                continue


    if flux is None:

        raise ValueError(
            "No usable flux"
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


    n = min(
        len(time),
        len(flux)
    )

    time = time[:n]
    flux = flux[:n]


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


    median = np.nanmedian(
        flux
    )


    flux = (
        flux /
        median
    )


    return (
        time,
        flux
    )


def download_lightcurve(tic):

    print(
        f"TIC {tic}: downloading..."
    )


    search = lk.search_lightcurve(
        f"TIC {tic}",
        mission="TESS"
    )


    if len(search) == 0:

        raise ValueError(
            "No TESS data"
        )


    curves = []


    for item in search[:MAX_SECTORS]:

        try:

            lc = item.download()

            if lc is None:

                continue


            time, flux = extract_flux(
                lc
            )


            sector = getattr(
                lc,
                "sector",
                len(curves) + 1
            )


            curves.append(
                {
                    "time": time,
                    "flux": flux,
                    "sector": sector
                }
            )


        except Exception as e:

            print(
                f"Sector failed: {e}"
            )


    if len(curves) == 0:

        raise ValueError(
            "No usable sectors"
        )


    return curves



def combine_curves(curves):

    time = []
    flux = []


    for curve in curves:

        time.extend(
            curve["time"]
        )

        flux.extend(
            curve["flux"]
        )


    time = np.array(
        time
    )

    flux = np.array(
        flux
    )


    order = np.argsort(
        time
    )


    return (
        time[order],
        flux[order]
    )



def save_lightcurve_plot(
    tic,
    time,
    flux,
    folder,
    morphology
):

    path = os.path.join(
        folder,
        "full_lightcurve.png"
    )


    plt.figure(
        figsize=(12, 5)
    )


    plt.plot(
        time,
        flux,
        linewidth=0.5
    )


    plt.axhline(
        1,
        linestyle="--",
        linewidth=0.8
    )


    plt.title(
        f"TIC {tic} | {morphology}"
    )


    plt.xlabel(
        "Time (BTJD)"
    )

    plt.ylabel(
        "Normalized Flux"
    )


    plt.grid(
        alpha=0.2
    )


    plt.tight_layout()


    plt.savefig(
        path,
        dpi=200
    )


    plt.close()



def save_periodogram(
    tic,
    time,
    flux,
    folder
):

    path = os.path.join(
        folder,
        "periodogram.png"
    )


    try:

        frequency, power = (
            LombScargle(
                time,
                flux
            )
            .autopower(
                minimum_frequency=0.01,
                maximum_frequency=10
            )
        )


        period = (
            1 /
            frequency
        )


        plt.figure(
            figsize=(10, 4)
        )


        plt.plot(
            period,
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
            alpha=0.2
        )


        plt.tight_layout()


        plt.savefig(
            path,
            dpi=200
        )


        plt.close()


    except Exception as e:

        print(
            f"Periodogram failed: {e}"
        )



def save_sigma_distribution(
    tic,
    flux,
    folder
):

    path = os.path.join(
        folder,
        "sigma_distribution.png"
    )


    median = np.median(
        flux
    )


    sigma = np.std(
        flux
    )


    if sigma == 0:

        return


    values = (
        (flux - median)
        /
        sigma
    )


    plt.figure(
        figsize=(8, 4)
    )


    plt.hist(
        values,
        bins=100
    )


    plt.axvline(
        5,
        linestyle="--"
    )


    plt.axvline(
        -5,
        linestyle="--"
    )


    plt.xlabel(
        "Sigma deviation"
    )


    plt.ylabel(
        "Count"
    )


    plt.title(
        f"TIC {tic} Sigma Distribution"
    )


    plt.grid(
        alpha=0.2
    )


    plt.tight_layout()


    plt.savefig(
        path,
        dpi=200
    )


    plt.close()



def save_phase_fold(
    tic,
    time,
    flux,
    period,
    folder
):

    if not np.isfinite(
        period
    ):

        return


    if period <= 0:

        return


    path = os.path.join(
        folder,
        "phase_folded.png"
    )


    phase = (
        (time % period)
        /
        period
    )


    plt.figure(
        figsize=(8, 4)
    )


    plt.scatter(
        phase,
        flux,
        s=2
    )


    plt.xlabel(
        "Phase"
    )


    plt.ylabel(
        "Normalized Flux"
    )


    plt.title(
        f"TIC {tic} Phase Folded"
    )


    plt.grid(
        alpha=0.2
    )


    plt.tight_layout()


    plt.savefig(
        path,
        dpi=200
    )


    plt.close()



def analyze_candidate(row):

    tic = int(
        row["tic_id"]
    )


    morphology = row.get(
        "morphology",
        "unknown"
    )


    folder = os.path.join(
        OUTPUT_DIR,
        f"TIC_{tic}"
    )


    os.makedirs(
        folder,
        exist_ok=True
    )


    curves = download_lightcurve(
        tic
    )


    time, flux = combine_curves(
        curves
    )


    save_lightcurve_plot(
        tic,
        time,
        flux,
        folder,
        morphology
    )


    save_periodogram(
        tic,
        time,
        flux,
        folder
    )


    save_sigma_distribution(
        tic,
        flux,
        folder
    )


    if np.isfinite(
        row.get(
            "best_period_days",
            np.nan
        )
    ):

        save_phase_fold(
            tic,
            time,
            flux,
            row["best_period_days"],
            folder
        )


    print(
        f"TIC {tic}: diagnostics complete"
    )



def main():

    print(
        "Loading classified candidates..."
    )


    df = pd.read_csv(
        INPUT_FILE
    )


    print(
        f"Generating diagnostics for {len(df)} candidates..."
    )


    for _, row in df.iterrows():

        tic = int(
            row["tic_id"]
        )


        try:

            analyze_candidate(
                row
            )


        except Exception as e:

            print(
                f"TIC {tic}: failed - {e}"
            )


    print(
        "\nFinished."
    )


    print(
        f"Saved diagnostics to: {OUTPUT_DIR}"
    )



if __name__ == "__main__":

    main()