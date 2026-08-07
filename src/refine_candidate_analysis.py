import numpy as np
import pandas as pd
import lightkurve as lk
from astropy.timeseries import LombScargle


INPUT_FILE = (
    "results/validation/"
    "candidate_lightcurves_status.csv"
)

OUTPUT_FILE = (
    "results/validation/"
    "candidate_analysis_refined.csv"
)

MAX_SECTORS = 3
MIN_POINTS = 50


def robust_sigma(flux):
    """
    Robust scatter estimate using MAD.
    """

    median = np.nanmedian(flux)

    mad = np.nanmedian(
        np.abs(
            flux - median
        )
    )

    sigma = (
        1.4826 *
        mad
    )

    return sigma, mad


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
            "No usable flux column"
        )


    try:

        flux = flux.value

    except AttributeError:

        pass


    flux = np.asarray(
        flux,
        dtype=float
    )


    time = np.asarray(
        lc.time.value,
        dtype=float
    )


    # Lightkurve objects can have
    # mismatched masked arrays
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
            "Not enough valid points"
        )


    median = np.nanmedian(
        flux
    )


    if not np.isfinite(median):

        raise ValueError(
            "Invalid median"
        )


    normalized_flux = (
        flux /
        median
    )


    return (
        time,
        normalized_flux
    )


def download_candidate(tic):

    print(
        f"TIC {tic}: searching MAST..."
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


    print(
        f"TIC {tic}: downloading "
        f"{min(len(search), MAX_SECTORS)} sectors"
    )


    for i, item in enumerate(
        search[:MAX_SECTORS]
    ):

        print(
            f"TIC {tic}: sector "
            f"{i+1}/{min(len(search), MAX_SECTORS)}"
        )


        try:

            lc = item.download()


            if lc is None:

                continue


            time, flux = extract_flux(
                lc
            )


            curves.append(
                (
                    time,
                    flux
                )
            )


        except Exception as e:

            print(
                f"Sector failed: {e}"
            )


    if len(curves) == 0:

        raise ValueError(
            "No usable light curves"
        )


    return curves


def combine_curves(curves):

    times = []
    fluxes = []


    for time, flux in curves:

        times.extend(
            time
        )

        fluxes.extend(
            flux
        )


    order = np.argsort(
        times
    )


    return (
        np.asarray(times)[order],
        np.asarray(fluxes)[order]
    )


def estimate_period(
    time,
    flux
):

    try:

        frequency, power = (
            LombScargle(
                time,
                flux
            )
            .autopower(
                minimum_frequency=1 / 100,
                maximum_frequency=10
            )
        )


        best_frequency = frequency[
            np.argmax(power)
        ]


        period = (
            1 /
            best_frequency
        )


        return float(
            period
        )


    except Exception:

        return np.nan


def analyze_curve(tic):

    curves = download_candidate(
        tic
    )


    time, flux = combine_curves(
        curves
    )


    sigma, mad = robust_sigma(
        flux
    )


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


    rms = np.std(
        flux
    )


    median = np.median(
        flux
    )


    if sigma > 0:

        sigma_values = (
            np.abs(
                flux - median
            )
            /
            sigma
        )

    else:

        sigma_values = np.zeros(
            len(flux)
        )


    max_sigma = np.max(
        sigma_values
    )


    outliers = np.sum(
        sigma_values > 5
    )


    period = estimate_period(
        time,
        flux
    )


    return {

        "tic_id": tic,

        "num_points": len(flux),

        "num_sectors": len(curves),

        "amplitude": amplitude,

        "rms": rms,

        "mad": mad,

        "robust_sigma": sigma,

        "max_sigma": max_sigma,

        "num_5sigma_outliers": int(
            outliers
        ),

        "best_period_days": period,

        "time_span_days": float(
            np.max(time)
            -
            np.min(time)
        )
    }


def main():

    print(
        "Loading candidates..."
    )


    df = pd.read_csv(
        INPUT_FILE
    )


    results = []


    print(
        f"Analyzing {len(df)} candidates..."
    )


    for _, row in df.iterrows():

        tic = int(
            row["tic_id"]
        )


        print(
            f"\nTIC {tic}"
        )


        try:

            analysis = analyze_curve(
                tic
            )


            analysis.update(
                {
                    "prediction":
                        row.get(
                            "prediction",
                            ""
                        ),

                    "confidence":
                        row.get(
                            "confidence",
                            np.nan
                        ),

                    "total_score":
                        row.get(
                            "total_score",
                            np.nan
                        )
                }
            )


            results.append(
                analysis
            )


            print(
                "success"
            )


        except Exception as e:

            print(
                f"failed: {e}"
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
        OUTPUT_FILE,
        index=False
    )


    print()
    print(
        "Finished."
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":

    main()