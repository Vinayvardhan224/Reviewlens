"""
ReviewLens Search & Filtering Component.
Sidebar and inline controls for category, brand, price slider, and sorting.
"""

from typing import Dict, Any, List
import streamlit as st
import pandas as pd


def render_sidebar_filters(df_products: pd.DataFrame) -> Dict[str, Any]:
    """
    Renders filter controls in Streamlit sidebar and returns applied filters.
    """
    st.sidebar.markdown("### 🎛️ Filter Catalogue")

    # 1. Department / Category
    all_categories = ["All Departments"] + sorted([c for c in df_products["category"].dropna().unique()])
    selected_cat = st.sidebar.selectbox("Department", all_categories, index=0)

    # 2. Brand multiselect
    available_brands = sorted([b for b in df_products["brand"].dropna().unique()])
    selected_brands = st.sidebar.multiselect("Brand", available_brands, default=[])

    # 3. Price range
    min_p = float(df_products["price_num"].min()) if not df_products.empty else 0.0
    max_p = float(df_products["price_num"].max()) if not df_products.empty else 250000.0
    price_range = st.sidebar.slider(
        "Price Range (₹)",
        min_value=int(min_p),
        max_value=int(max_p),
        value=(int(min_p), int(max_p)),
        step=5000,
        format="₹%d"
    )

    # 4. Minimum Rating
    min_rating = st.sidebar.slider("Minimum Rating", min_value=1.0, max_value=5.0, value=3.5, step=0.1)

    # 5. Sorting
    sort_option = st.sidebar.selectbox(
        "Sort By",
        [
            "Featured",
            "Price: Low to High",
            "Price: High to Low",
            "Customer Rating: High to Low"
        ],
        index=0
    )

    return {
        "category": selected_cat if selected_cat != "All Departments" else None,
        "brands": selected_brands,
        "price_min": price_range[0],
        "price_max": price_range[1],
        "min_rating": min_rating,
        "sort_by": sort_option
    }


def apply_filters(df_products: pd.DataFrame, search_query: str, filters: Dict[str, Any]) -> pd.DataFrame:
    """Filter and sort products based on user selections."""
    if df_products.empty:
        return df_products

    res = df_products.copy()

    # Search query
    if search_query and search_query.strip():
        q = search_query.strip().lower()
        res = res[
            res["product_name"].str.lower().str.contains(q, na=False) |
            res["brand"].str.lower().str.contains(q, na=False) |
            res["category"].str.lower().str.contains(q, na=False) |
            res["description"].str.lower().str.contains(q, na=False)
        ]

    # Category
    if filters.get("category"):
        res = res[res["category"].str.lower() == filters["category"].lower()]

    # Brands
    if filters.get("brands"):
        res = res[res["brand"].isin(filters["brands"])]

    # Price
    if "price_min" in filters and "price_max" in filters:
        res = res[
            (res["price_num"] >= filters["price_min"]) &
            (res["price_num"] <= filters["price_max"])
        ]

    # Rating
    if "min_rating" in filters:
        res = res[res["rating_num"] >= filters["min_rating"]]

    # Sorting
    sort_by = filters.get("sort_by", "Featured")
    if sort_by == "Price: Low to High":
        res = res.sort_values(by="price_num", ascending=True)
    elif sort_by == "Price: High to Low":
        res = res.sort_values(by="price_num", ascending=False)
    elif sort_by == "Customer Rating: High to Low":
        res = res.sort_values(by="rating_num", ascending=False)

    return res.reset_index(drop=True)
