"""
ReviewLens Data Validation Module.
Validates products and reviews datasets against schema rules, integrity constraints,
orphan foreign keys, and statistical anomalies.
"""

from typing import Dict, Any, List, Optional
import pandas as pd


REQUIRED_PRODUCT_COLUMNS = [
    "product_id",
    "product_name",
    "brand",
    "category",
    "price",
    "rating"
]

REQUIRED_REVIEW_COLUMNS = [
    "review_id",
    "product_id"
]


class ValidationResult:
    """Encapsulates validation outcome, errors, warnings, and summary statistics."""
    def __init__(self):
        self.is_valid: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.stats: Dict[str, Any] = {}

    def add_error(self, message: str) -> None:
        self.is_valid = False
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats
        }


def validate_products_df(df: pd.DataFrame) -> ValidationResult:
    """Validate a products DataFrame."""
    res = ValidationResult()

    if df is None or df.empty:
        res.add_error("Products dataset is empty or None.")
        return res

    # 1. Check required columns
    missing_cols = [c for c in REQUIRED_PRODUCT_COLUMNS if c not in df.columns]
    if missing_cols:
        res.add_error(f"Missing required product columns: {missing_cols}")

    # 2. Check for duplicate product_ids
    if "product_id" in df.columns:
        dups = df["product_id"].duplicated().sum()
        if dups > 0:
            res.add_error(f"Found {dups} duplicate product_id values.")

        null_ids = df["product_id"].isna().sum()
        if null_ids > 0:
            res.add_error(f"Found {null_ids} null or empty product_id values.")

    # 3. Check price numeric and non-negative
    if "price" in df.columns:
        converted_price = pd.to_numeric(df["price"], errors="coerce")
        non_numeric_price = converted_price.isna().sum()
        if non_numeric_price > 0:
            res.add_error(f"Found {non_numeric_price} non-numeric or missing price entries.")
        else:
            negative_prices = (converted_price < 0).sum()
            if negative_prices > 0:
                res.add_error(f"Found {negative_prices} negative price entries.")

    # 4. Check rating in 1..5 range
    if "rating" in df.columns:
        numeric_ratings = pd.to_numeric(df["rating"], errors="coerce")
        missing_ratings = numeric_ratings.isna().sum()
        if missing_ratings > 0:
            res.add_error(f"Found {missing_ratings} non-numeric or missing rating entries.")
        out_of_bounds = ((numeric_ratings < 1.0) | (numeric_ratings > 5.0)).sum()
        if out_of_bounds > 0:
            res.add_error(f"Found {out_of_bounds} ratings outside valid range [1.0, 5.0].")

    # 5. Check provenance / data_type
    if "data_type" in df.columns:
        demo_count = (df["data_type"] == "prepared_demo").sum()
        res.stats["demo_products_count"] = int(demo_count)
    else:
        res.add_warning("Optional column 'data_type' missing; assuming prepared demo data.")

    res.stats["total_products"] = len(df)
    res.stats["categories"] = df["category"].dropna().unique().tolist() if "category" in df.columns else []
    res.stats["brands"] = df["brand"].dropna().unique().tolist() if "brand" in df.columns else []

    return res


def validate_reviews_df(df: pd.DataFrame, products_df: Optional[pd.DataFrame] = None) -> ValidationResult:
    """Validate a reviews DataFrame and optionally verify referential integrity."""
    res = ValidationResult()

    if df is None or df.empty:
        res.add_error("Reviews dataset is empty or None.")
        return res

    # 1. Required identifier columns
    missing_cols = [c for c in REQUIRED_REVIEW_COLUMNS if c not in df.columns]
    if missing_cols:
        res.add_error(f"Missing required review columns: {missing_cols}")

    # Check for presence of review text (either 'review_text' or 'text')
    has_text_col = ("review_text" in df.columns) or ("text" in df.columns)
    if not has_text_col:
        res.add_error("Missing required review text column ('review_text' or 'text').")

    # Check for presence of review rating (either 'review_rating' or 'rating')
    has_rating_col = ("review_rating" in df.columns) or ("rating" in df.columns)
    if not has_rating_col:
        res.add_error("Missing required rating column ('review_rating' or 'rating').")

    # 2. Duplicate review_id
    if "review_id" in df.columns:
        dups = df["review_id"].duplicated().sum()
        if dups > 0:
            res.add_error(f"Found {dups} duplicate review_id values.")

    # 3. Empty review texts
    text_col = "review_text" if "review_text" in df.columns else ("text" if "text" in df.columns else None)
    if text_col:
        empty_texts = df[text_col].isna().sum() + (df[text_col].astype(str).str.strip() == "").sum()
        if empty_texts == len(df) and len(df) > 0:
            res.add_error(f"All review texts in column '{text_col}' are empty or whitespace-only.")
        elif empty_texts > 0:
            res.add_warning(f"Found {empty_texts} empty review text entries.")

    # 4. Rating range
    rating_col = "review_rating" if "review_rating" in df.columns else ("rating" if "rating" in df.columns else None)
    if rating_col:
        num_ratings = pd.to_numeric(df[rating_col], errors="coerce")
        nan_ratings = num_ratings.isna().sum()
        if nan_ratings > 0:
            res.add_error(f"Found {nan_ratings} non-numeric or missing review ratings.")
        invalid_ratings = ((num_ratings < 1.0) | (num_ratings > 5.0)).sum()
        if invalid_ratings > 0:
            res.add_error(f"Found {invalid_ratings} review ratings outside [1.0, 5.0].")

    # 5. Referential integrity / orphan reviews
    if products_df is not None and "product_id" in products_df.columns and "product_id" in df.columns:
        valid_product_ids = set(products_df["product_id"].dropna().unique())
        review_product_ids = set(df["product_id"].dropna().unique())
        orphan_ids = review_product_ids - valid_product_ids
        if orphan_ids:
            res.add_error(f"Found reviews for {len(orphan_ids)} unknown product IDs (orphans): {list(orphan_ids)[:3]}")

        # Review count distribution & sample size warnings
        counts_per_product = df["product_id"].value_counts()
        low_sample_products = counts_per_product[counts_per_product < 5].index.tolist()
        if low_sample_products:
            res.add_warning(f"{len(low_sample_products)} products have fewer than 5 reviews (low statistical power).")

    res.stats["total_reviews"] = len(df)
    if "product_id" in df.columns:
        res.stats["unique_products_reviewed"] = int(df["product_id"].nunique())

    return res
