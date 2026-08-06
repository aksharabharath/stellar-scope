import pandas as pd
from pathlib import Path


LOG = Path("data/metadata/download_log.csv")
OUT = Path("data/metadata/dataset_v1.csv")
PROCESSED = Path("data/processed")


df = pd.read_csv(LOG)

# Keep only usable successful downloads
df = df[
    (df["status"] == "success") &
    (df["rows"] > 0)
]


manifest = []

for _, row in df.iterrows():

    tic = int(row["tic_id"])

    filename = f"TIC_{tic}.csv"

    if (PROCESSED / filename).exists():

        manifest.append(
            {
                "tic_id": tic,
                "rows": int(row["rows"]),
                "filename": filename
            }
        )


manifest = pd.DataFrame(manifest)

manifest.to_csv(
    OUT,
    index=False
)


print(
    f"Created dataset manifest: {len(manifest)} stars"
)

print(manifest.head())