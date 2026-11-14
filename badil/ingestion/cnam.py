import re

import pandas as pd

from badil.paths import DATA_RAW, DATA_PROCESSED

RAW_FILE = DATA_RAW / "cnam" / "liste_medicaments_vei_2026-06-17.xlsx"
OUTPUT_FILE = DATA_PROCESSED / "medicines_cnam.csv"
LIST_DATE = re.search(r"(\d{4}-\d{2}-\d{2})", RAW_FILE.name).group(1)

COLUMN_MAP = {
    "CODE_PCT": "pct_code",
    "NOM_COMMERCIAL": "brand_name",
    "PRIX_PUBLIC": "public_price",
    "TARIF_REFERENCE": "reference_tariff",
    "CATEGORIE": "category",
    "DCI": "active_ingredient",
    "AP": "prior_authorization",
}


def load_raw() -> pd.DataFrame:
    return pd.read_excel(RAW_FILE, sheet_name="Export Worksheet")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_MAP)
    df["pct_code"] = df["pct_code"].astype(str).str.strip()
    df["brand_name"] = df["brand_name"].str.strip()
    df["active_ingredient"] = df["active_ingredient"].str.strip().str.upper()
    df["prior_authorization"] = df["prior_authorization"].str.strip() == "O"
    df = df.drop_duplicates(subset="pct_code")
    return df.dropna(subset=["pct_code", "active_ingredient"])


def run() -> pd.DataFrame:
    df = clean(load_raw())
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    return df


if __name__ == "__main__":
    result = run()
    print(result.shape)
    print(result.head())
