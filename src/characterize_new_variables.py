import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightkurve as lk
from astropy.timeseries import LombScargle
from scipy.stats import skew, kurtosis


INPUT_FILE = (
    "results/validation/"
    "new_variable_candidate_analysis.csv"
)

OUTPUT_DIR = (
    "results/validation/"
    "variable_characterization"
)

CSV_OUTPUT = (
    "results/validation/"
    "variable_characterization.csv"
)

REPORT_OUTPUT = (
    "results/validation/"
    "variable_characterization_report.txt"
)


MAX_SECTORS = 3
MIN_POINTS = 100


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def extract_lightcurve(lc):

    flux = None

    for name in [
        "pdcsap_flux",
        "sap_flux",
        "flux"
    ]:

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
            "No flux"
        )


    try:
        flux = flux.value
    except AttributeError:
        pass


    time = lc.time.value


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
            "Too few points"
        )


    median = np.median(
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



def download_candidate(tic):

    print(
        f"TIC {tic}: downloading..."
    )


    search = lk.search_lightcurve(
        f"TIC {tic}",
        mission="TESS"
    )


    if len(search) == 0:
        raise ValueError(
            "No observations"
        )


    curves = []


    for item in search[:MAX_SECTORS]:

        try:

            lc = item.download()

            if lc is None:
                continue


            time, flux = extract_lightcurve(
                lc
            )


            curves.append(
                (
                    time,
                    flux
                )
            )


        except Exception:

            continue


    if len(curves) == 0:

        raise ValueError(
            "No usable curves"
        )


    time = np.concatenate(
        [
            x[0]
            for x in curves
        ]
    )


    flux = np.concatenate(
        [
            x[1]
            for x in curves
        ]
    )


    order = np.argsort(
        time
    )


    return (
        time[order],
        flux[order]
    )



def refine_period(
    time,
    flux
):

    frequency, power = (
        LombScargle(
            time,
            flux
        )
        .autopower(
            minimum_frequency=1/100,
            maximum_frequency=10,
            samples_per_peak=20
        )
    )


    best_frequency = (
        frequency[
            np.argmax(power)
        ]
    )


    period = (
        1 /
        best_frequency
    )


    return float(period)



def phase_fold(
    time,
    flux,
    period
):

    phase = (
        time %
        period
    ) / period


    order = np.argsort(
        phase
    )


    return (
        phase[order],
        flux[order]
    )



def calculate_metrics(
    flux
):

    amplitude = (
        np.percentile(
            flux,
            99
        )
        -
        np.percentile(
            flux,
            1
        )
    )


    return {

        "amplitude":
            float(amplitude),

        "std":
            float(
                np.std(flux)
            ),

        "skewness":
            float(
                skew(flux)
            ),

        "kurtosis":
            float(
                kurtosis(flux)
            )
    }



def plot_phase_curve(
    tic,
    phase,
    flux,
    period
):

    path = os.path.join(
        OUTPUT_DIR,
        f"TIC_{tic}_phase_curve.png"
    )


    plt.figure(
        figsize=(8,4)
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
        f"TIC {tic} "
        f"Phase Folded\n"
        f"Period={period:.5f} days"
    )


    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(
        path,
        dpi=200
    )


    plt.close()


    return path



def characterize_candidate(tic):

    time, flux = download_candidate(
        tic
    )


    period = refine_period(
        time,
        flux
    )


    phase, folded_flux = phase_fold(
        time,
        flux,
        period
    )


    plot = plot_phase_curve(
        tic,
        phase,
        folded_flux,
        period
    )


    metrics = calculate_metrics(
        flux
    )


    result = {

        "tic_id":
            tic,

        "period_days":
            period,

        "time_span_days":
            float(
                np.max(time)
                -
                np.min(time)
            ),

        "num_points":
            len(flux),

        "phase_plot":
            plot

    }


    result.update(
        metrics
    )


    return result



def main():

    print(
        "Loading candidates..."
    )


    df = pd.read_csv(
        INPUT_FILE
    )


    print(
        f"Loaded {len(df)} candidates"
    )


    results = []


    for _, row in df.iterrows():

        tic = int(
            row["tic_id"]
        )


        print(
            f"TIC {tic}"
        )


        try:

            result = characterize_candidate(
                tic
            )


            result.update(
                {
                    "confidence":
                        row["confidence"],

                    "classification":
                        row["classification"],

                    "priority":
                        row["priority"]
                }
            )


            results.append(
                result
            )


            print(
                "  complete"
            )


        except Exception as e:

            print(
                f"  failed: {e}"
            )


            results.append(
                {
                    "tic_id": tic,
                    "error": str(e)
                }
            )


    output = pd.DataFrame(
        results
    )


    output.to_csv(
        CSV_OUTPUT,
        index=False
    )


    with open(
        REPORT_OUTPUT,
        "w"
    ) as f:


        f.write(
            "VARIABLE CHARACTERIZATION REPORT\n"
        )

        f.write(
            "================================\n\n"
        )


        for _, row in output.iterrows():

            f.write(
                f"TIC {row['tic_id']}\n"
            )

            f.write(
                f"Period: {row.get('period_days','')}\n"
            )

            f.write(
                f"Amplitude: {row.get('amplitude','')}\n"
            )

            f.write(
                f"Skewness: {row.get('skewness','')}\n"
            )

            f.write(
                "\n"
            )


    print(
        "\nFinished."
    )


    print(
        f"Saved CSV: {CSV_OUTPUT}"
    )

    print(
        f"Saved report: {REPORT_OUTPUT}"
    )



if __name__ == "__main__":

    main()