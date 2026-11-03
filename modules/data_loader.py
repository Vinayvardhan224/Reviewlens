"""
ReviewLens Data Loader Module.
Loads, caches, and indexes catalogue products and consumer reviews with strict
data validation and provenance tagging.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import pandas as pd
from modules.validation import validate_products_df, validate_reviews_df, ValidationResult

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

PROVENANCE_DISCLOSURE = (
    "Dataset Notice: Product specifications and customer reviews in this platform "
    "are prepared demonstrations (data_type='prepared_demo') curated for transparent "
    "explainable NLP analysis. Outbound links connect directly to Amazon India for live retailer listings."
)


def load_products_raw() -> pd.DataFrame:
    """Load products CSV from data/products.csv."""
    path = DATA_DIR / "products.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")


def load_reviews_raw() -> pd.DataFrame:
    """Load reviews CSV from data/reviews.csv."""
    path = DATA_DIR / "reviews.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")


def get_all_products() -> pd.DataFrame:
    """Retrieve validated DataFrame of all catalogue products."""
    df = load_products_raw()
    if df.empty:
        return df

    # Ensure price is numeric
    if "price" in df.columns:
        df["price_num"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    else:
        df["price_num"] = 0.0

    # Ensure rating is numeric
    if "rating" in df.columns:
        df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce").fillna(4.0)
    else:
        df["rating_num"] = 4.0

    return df


def get_all_reviews() -> pd.DataFrame:
    """Retrieve validated DataFrame of all customer reviews."""
    df = load_reviews_raw()
    if df.empty:
        return df

    if "review_rating" in df.columns:
        df["review_rating_num"] = pd.to_numeric(df["review_rating"], errors="coerce").fillna(3.0)
    return df


def get_product_by_id(product_id: str) -> Optional[Dict[str, Any]]:
    """Get single product record by product_id with parsed specifications."""
    df = get_all_products()
    if df.empty or "product_id" not in df.columns:
        return None

    matches = df[df["product_id"] == product_id]
    if matches.empty:
        return None

    prod = matches.iloc[0].to_dict()

    # Parse specifications JSON
    raw_specs = prod.get("specifications")
    specs_dict = {}
    if isinstance(raw_specs, str) and raw_specs.strip().startswith("{"):
        try:
            specs_dict = json.loads(raw_specs)
        except Exception:
            specs_dict = {"Details": raw_specs}
    elif isinstance(raw_specs, dict):
        specs_dict = raw_specs

    prod["parsed_specs"] = specs_dict
    return prod


def get_reviews_for_product(product_id: str) -> pd.DataFrame:
    """Filter reviews specifically for the requested product_id."""
    df = get_all_reviews()
    if df.empty or "product_id" not in df.columns:
        return pd.DataFrame()

    matched = df[df["product_id"] == product_id].copy()
    return matched.reset_index(drop=True)


def format_currency_inr(amount: float) -> str:
    """Format numeric value into Indian Rupee string (e.g., ₹39,999)."""
    try:
        val = int(round(float(amount)))
        # Indian numbering system grouping
        s = str(val)
        if len(s) <= 3:
            return f"₹{s}"
        last_three = s[-3:]
        remaining = s[:-3]
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        formatted = ",".join(groups) + "," + last_three
        return f"₹{formatted}"
    except Exception:
        return f"₹{amount}"


def get_provenance_badge() -> Dict[str, str]:
    """Return provenance tag and label for UI components."""
    return {
        "tag": "Prepared Demo Data",
        "description": PROVENANCE_DISCLOSURE,
        "badge_color": "#2563EB",
        "icon": "🛡️"
    }
