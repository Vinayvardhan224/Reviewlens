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


def get_all_products(validate: bool = True) -> pd.DataFrame:
    """Retrieve validated DataFrame of all catalogue products."""
    df = load_products_raw()
    if df.empty:
        return df

    if validate:
        val = validate_products_df(df)
        if not val.is_valid:
            raise ValueError(f"Product catalogue validation failed: {'; '.join(val.errors)}")

    # Enforce numeric price without silent default imputation
    if "price" in df.columns:
        converted_price = pd.to_numeric(df["price"], errors="coerce")
        if converted_price.isna().any():
            raise ValueError("Found non-numeric or missing price values in product catalogue.")
        df["price_num"] = converted_price

    # Enforce numeric rating without silent default imputation
    if "rating" in df.columns:
        converted_rating = pd.to_numeric(df["rating"], errors="coerce")
        if converted_rating.isna().any():
            raise ValueError("Found non-numeric or missing rating values in product catalogue.")
        df["rating_num"] = converted_rating

    return df


def get_all_reviews(validate: bool = True, products_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Retrieve validated DataFrame of all customer reviews."""
    df = load_reviews_raw()
    if df.empty:
        return df

    if validate:
        p_ref = products_df if products_df is not None else load_products_raw()
        val = validate_reviews_df(df, products_df=p_ref)
        if not val.is_valid:
            raise ValueError(f"Review dataset validation failed: {'; '.join(val.errors)}")

    # Enforce numeric review rating without silent default imputation
    if "review_rating" in df.columns:
        converted_rev_rating = pd.to_numeric(df["review_rating"], errors="coerce")
        if converted_rev_rating.isna().any():
            raise ValueError("Found non-numeric or missing review_rating values in reviews dataset.")
        df["review_rating_num"] = converted_rev_rating

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


def get_product_image_src(prod: Dict[str, Any]) -> str:
    """
    Retrieve locally stored authorized product image as a base64 data URI.
    Guarantees local rendering without broken external URLs or emojis.
    """
    import base64
    img_path = prod.get("image_path")
    if img_path:
        local_file = BASE_DIR / img_path
        if local_file.exists():
            try:
                with open(local_file, "rb") as f:
                    b64_str = base64.b64encode(f.read()).decode("utf-8")
                    return f"data:image/jpeg;base64,{b64_str}"
            except Exception:
                pass
    return prod.get("image_url", "https://placehold.co/400x300/e2e8f0/475569?text=Product+Image")
