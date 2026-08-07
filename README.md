# Stellar Scope

## Machine Learning Assisted Discovery and Validation of Variable Stars from TESS Light Curves

Stellar Scope is an astronomical data analysis pipeline designed to identify, validate, and characterize potential variable stars using data from NASA's Transiting Exoplanet Survey Satellite (TESS).

The project combines light curve analysis, statistical variability detection, period analysis, morphology classification, external catalog crossmatching, and stellar characterization to investigate unusual stellar brightness variations.

The goal is to build a reproducible workflow for finding candidate variable stars and evaluating whether they represent known objects or potential new discoveries.

## Research Question

Can automated analysis of TESS photometric time-series data identify previously unclassified variable star candidates?

## Overview

The pipeline follows several stages:

1. Candidate generation
2. Light curve validation
3. Variability analysis
4. Morphological classification
5. Diagnostic visualization
6. Catalog crossmatching
7. Stellar characterization

The final output is a ranked list of variable star candidates with supporting evidence from photometric behavior and external astronomical databases.

## Data Sources

### TESS

Data source:

* NASA Transiting Exoplanet Survey Satellite (TESS)
* Accessed through the Mikulski Archive for Space Telescopes (MAST)

TESS provides high-precision stellar brightness measurements collected over multiple observing sectors.

### Gaia

Gaia DR3 is used for stellar characterization, including:

* Effective temperature
* Radius
* Luminosity
* Color index
* Parallax
* Stellar identifiers

### VSX

The International Variable Star Index (VSX) is used to determine whether candidates are already classified variable stars.

Candidates without VSX matches are investigated as possible new variable star candidates.

## Pipeline Structure

```
stellar-scope/

├── data/
│
├── models/
│
├── notebooks/
│
├── results/
│   └── validation/
│       ├── candidate_lightcurves/
│       ├── diagnostics/
│       ├── candidate_validation_summary.csv
│       ├── vsx_coordinate_crossmatch.csv
│       ├── variable_characterization.csv
│       └── gaia_characterization.csv
│
├── src/
│   ├── plot_candidate_lightcurves.py
│   ├── refine_candidate_analysis.py
│   ├── classify_variable_morphology.py
│   ├── generate_candidate_diagnostics.py
│   ├── summarize_candidate_validation.py
│   ├── crossmatch_vsx_candidates.py
│   ├── crossmatch_vsx_coordinates.py
│   ├── analyze_new_variable_candidates.py
│   ├── characterize_new_variables.py
│   └── gaia_characterize_candidates.py
│
└── README.md
```

## Pipeline Stages

## 1. Candidate Validation

Initial candidates are loaded from ranked variability results.

Each candidate is evaluated using:

* Flux amplitude
* RMS variability
* Median absolute deviation (MAD)
* Robust sigma deviation
* Number of significant outliers
* Time baseline
* Number of available TESS sectors

Example metrics:

* Variability strength
* Long-term consistency
* Statistical significance

## 2. TESS Light Curve Retrieval

For each candidate:

* TESS observations are downloaded from MAST
* Multiple sectors are combined
* Flux values are normalized
* Diagnostic plots are generated

Outputs:

```
results/validation/candidate_lightcurves/
```

## 3. Variability Characterization

Each candidate is analyzed for:

### Statistical Variability

Measured using:

* Flux amplitude
* RMS scatter
* MAD
* Robust sigma

### Periodicity

Period searches are performed using Lomb-Scargle analysis.

Detected periodic behavior can indicate:

* Rotational modulation
* Pulsations
* Binary variability
* Stellar activity

### Morphology Classification

Candidates are grouped into categories:

* Periodic variable
* Transient event
* Low amplitude variable

Classification uses observed light curve behavior.

## 4. Diagnostic Generation

For each candidate, diagnostic products are generated:

* Light curve plots
* Variability statistics
* Period information
* Outlier analysis

Stored in:

```
results/validation/diagnostics/
```

These diagnostics allow manual inspection of candidate behavior.

## 5. VSX Crossmatching

Candidates are crossmatched with the International Variable Star Index.

Purpose:

* Remove already-known variable stars
* Identify potentially new candidates

Results:

```
vsx_coordinate_crossmatch.csv
potential_new_variables.csv
known_variables.csv
```

Candidates without VSX matches are flagged for further analysis.

## 6. Gaia Stellar Characterization

Potential new candidates are linked with Gaia DR3.

Retrieved properties include:

* Gaia source ID
* Stellar temperature
* Radius
* Luminosity
* BP-RP color
* Parallax

This provides physical context for variability behavior.

## Results

The validation pipeline produces:

* Ranked candidate list
* Light curve visualizations
* Variability classifications
* VSX validation
* Gaia stellar properties
* Candidate reports

Example outputs:

```
results/validation/
├── candidate_validation_summary.csv
├── candidate_validation_report.txt
├── variable_characterization.csv
├── vsx_coordinate_crossmatch.csv
└── gaia_characterization.csv
```

## Technologies Used

### Programming

* Python
* NumPy
* Pandas

### Astronomy

* Lightkurve
* Astroquery
* Astropy
* MAST API
* Gaia Archive
* VSX

### Analysis

* Lomb-Scargle periodograms
* Statistical variability metrics
* Time-series analysis
* Automated classification

## Future Improvements

Potential future extensions:

* Machine learning based light curve classifier
* Automated stellar type estimation
* Better flare detection
* Transit/variable star separation
* Deep learning models for morphology classification
* Submission of strong candidates to variable star databases

## Author

Akshara Bharath

## Project Status

Active research project.

Current focus:

* Validating high-confidence TESS variable candidates
* Identifying possible previously unclassified variables
* Adding stellar characterization through Gaia DR3