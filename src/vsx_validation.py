import os
import joblib
import pandas as pd

from astroquery.mast import Catalogs
from astroquery.vizier import Vizier

from astropy.coordinates import SkyCoord
import astropy.units as u

from sklearn.metrics import classification_report


MODEL_PATH = "models/stellar_classifier.pkl"

FEATURE_PATH = "data/features/features.csv"

OUTPUT_DATA = "data/validation/vsx_validation_results.csv"

OUTPUT_DIR = "results/validation"

CACHE_FILE = "data/validation/tic_coordinates_cache.csv"

CANDIDATE_FILE = "results/validation/variable_candidates.csv"


FEATURE_COLUMNS = [
    "flux_std",
    "flux_range",
    "relative_flux_range",
    "coefficient_of_variation",
    "flux_mad",
    "flux_skew",
    "flux_kurtosis",
    "dominant_period",
    "period_power",
    "period_confidence",
    "num_periods",
    "flare_count",
    "flare_frequency",
    "largest_flare",
    "flare_strength",
    "dip_count",
    "largest_dip_depth",
]


Vizier.ROW_LIMIT = 1



def load_model():

    saved = joblib.load(
        MODEL_PATH
    )

    if isinstance(saved, dict):

        return (
            saved.get("model"),
            saved.get("encoder")
        )

    return saved, None



def load_features():

    df = pd.read_csv(
        FEATURE_PATH
    )

    X = df[
        FEATURE_COLUMNS
    ]

    return df, X



def predict(model, encoder, X):

    predictions = model.predict(
        X
    )

    if encoder is not None:

        predictions = encoder.inverse_transform(
            predictions
        )

    return predictions



def predict_confidence(model, X):

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            X
        )

        return probabilities.max(
            axis=1
        )

    return [
        None
        for _ in range(len(X))
    ]



def load_coordinate_cache():

    if os.path.exists(
        CACHE_FILE
    ):

        return pd.read_csv(
            CACHE_FILE
        )

    return pd.DataFrame(
        columns=[
            "tic_id",
            "ra",
            "dec"
        ]
    )



def save_coordinate_cache(cache):

    os.makedirs(
        os.path.dirname(CACHE_FILE),
        exist_ok=True
    )

    cache.to_csv(
        CACHE_FILE,
        index=False
    )



def get_coordinates(tic, cache):

    cached = cache[
        cache["tic_id"] == tic
    ]


    if len(cached) > 0:

        return (
            float(cached.iloc[0]["ra"]),
            float(cached.iloc[0]["dec"]),
            cache
        )


    try:

        result = Catalogs.query_criteria(
            ID=int(tic),
            catalog="Tic"
        )


        if len(result) == 0:

            return None, None, cache


        ra = float(
            result[0]["ra"]
        )

        dec = float(
            result[0]["dec"]
        )


        cache.loc[len(cache)] = [
            tic,
            ra,
            dec
        ]


        save_coordinate_cache(
            cache
        )


        return (
            ra,
            dec,
            cache
        )


    except Exception:

        return None, None, cache



def map_vsx_type(vsx_type):

    if vsx_type is None:

        return None, False


    value = str(
        vsx_type
    ).upper()



    flare_types = [
        "UV",
        "UV CET",
        "FL"
    ]


    if any(
        item in value
        for item in flare_types
    ):

        return (
            "flare_variable",
            True
        )



    periodic_types = [

        "EA",
        "EB",
        "EW",
        "ELL",
        "ROT",
        "DSCT",
        "GDOR",
        "RR",
        "CEP",
        "MIRA",
        "SR",
        "RS",
        "ESD",
        "ED"

    ]


    if any(
        item in value
        for item in periodic_types
    ):

        return (
            "periodic_variable",
            True
        )



    transient_types = [

        "CV",
        "NOVA",
        "SN",
        "YSO"

    ]


    if any(
        item in value
        for item in transient_types
    ):

        return (
            "transient_variable",
            True
        )



    # Generic variable label
    # usable only for binary validation

    if "VAR" in value:

        return (
            "variable",
            False
        )


    return (
        None,
        False
    )



def query_vsx(ra, dec):

    try:

        coord = SkyCoord(
            ra,
            dec,
            unit="deg"
        )


        tables = Vizier.query_region(
            coord,
            radius=5 * u.arcsec,
            catalog="B/vsx"
        )


        if len(tables) == 0:

            return (
                None,
                None,
                False,
                False
            )


        table = tables[0]


        if len(table) == 0:

            return (
                None,
                None,
                False,
                False
            )


        vsx_type = str(
            table[0]["Type"]
        )


        classification, subtype = map_vsx_type(
            vsx_type
        )


        return (
            classification,
            vsx_type,
            subtype,
            True
        )


    except Exception:

        return (
            None,
            None,
            False,
            False
        )



def validate(df, predictions, confidence):

    print(
        "Querying VSX catalog..."
    )


    cache = load_coordinate_cache()


    rows = []


    for i, tic in enumerate(
        df["tic_id"]
    ):

        print(
            f"[{i+1}/{len(df)}] TIC {tic}"
        )


        ra, dec, cache = get_coordinates(
            tic,
            cache
        )


        if ra is None:

            classification = None
            vsx_type = None
            subtype = False
            matched = False


        else:

            (
                classification,
                vsx_type,
                subtype,
                matched
            ) = query_vsx(
                ra,
                dec
            )


        rows.append(
            {
                "tic_id": tic,
                "prediction": predictions[i],
                "confidence": confidence[i],
                "vsx_classification": classification,
                "vsx_type": vsx_type,
                "subtype_confirmed": subtype,
                "matched_vsx": matched
            }
        )


    return pd.DataFrame(rows)



def evaluate_subtypes(results):

    valid = results[
        results["subtype_confirmed"]
    ]


    print()

    print(
        f"Subtype validation samples: {len(valid)}"
    )


    if len(valid) == 0:

        return


    print(
        pd.crosstab(
            valid["vsx_classification"],
            valid["prediction"]
        )
    )


    report = classification_report(
        valid["vsx_classification"],
        valid["prediction"],
        zero_division=0
    )


    print(report)


    with open(
        f"{OUTPUT_DIR}/vsx_subtype_report.txt",
        "w"
    ) as f:

        f.write(report)



def evaluate_binary(results):

    valid = results[
        results["matched_vsx"]
    ].copy()


    print()

    print(
        "Binary variable detection"
    )


    if len(valid) == 0:

        return


    valid["vsx_binary"] = "variable"


    valid["prediction_binary"] = valid[
        "prediction"
    ].apply(
        lambda x:
        "variable"
        if x != "non_variable"
        else "non_variable"
    )


    report = classification_report(
        valid["vsx_binary"],
        valid["prediction_binary"],
        zero_division=0
    )


    print(report)


    with open(
        f"{OUTPUT_DIR}/vsx_binary_report.txt",
        "w"
    ) as f:

        f.write(report)



def save_errors(results):

    valid = results[
        results["matched_vsx"]
    ]


    errors = valid[
        valid["prediction"]
        !=
        valid["vsx_classification"]
    ]


    errors.to_csv(
        f"{OUTPUT_DIR}/vsx_errors.csv",
        index=False
    )


    print(
        f"Saved VSX errors: {len(errors)}"
    )



def save_candidates(results):

    candidates = results[
        (~results["matched_vsx"])
        &
        (results["prediction"] != "non_variable")
    ].copy()


    candidates = candidates.sort_values(
        "confidence",
        ascending=False
    )


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    candidates.to_csv(
        CANDIDATE_FILE,
        index=False
    )


    print(
        f"Saved variable candidates: {len(candidates)}"
    )



def main():

    model, encoder = load_model()


    df, X = load_features()


    predictions = predict(
        model,
        encoder,
        X
    )


    confidence = predict_confidence(
        model,
        X
    )


    results = validate(
        df,
        predictions,
        confidence
    )


    os.makedirs(
        "data/validation",
        exist_ok=True
    )


    results.to_csv(
        OUTPUT_DATA,
        index=False
    )


    print()

    print(
        f"Saved: {OUTPUT_DATA}"
    )


    print()

    print(
        f"Total VSX matches: {results['matched_vsx'].sum()}"
    )


    evaluate_subtypes(
        results
    )


    evaluate_binary(
        results
    )


    save_errors(
        results
    )


    save_candidates(
        results
    )



if __name__ == "__main__":

    main()