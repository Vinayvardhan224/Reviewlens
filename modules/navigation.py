"""
ReviewLens Navigation Configuration and Taxonomies.
Defines grouped navigation hierarchies:
  - ANALYZE: Core review intelligence, live diagnostic lab, and benchmarks.
  - COMPARE: Side-by-side product comparisons and preference matchmaker.
  - SHOP (demo): Curated e-commerce catalogue and product detail views.
  - MORE: Data ingestion audit and methodology transparency.
"""

from typing import Dict, List

NAV_GROUPS: Dict[str, List[str]] = {
    "ANALYZE": [
        "🔍 Review Intelligence",
        "🧪 Live Review Analyzer",
        "📊 Original Analytics Dashboard",
        "📈 Model Evaluation & Benchmarks",
    ],
    "COMPARE": [
        "⚔️ Product Comparison",
        "🎯 Smart Recommendations",
    ],
    "SHOP (demo)": [
        "🏠 Product Explorer & Storefront",
        "📱 Smartphones",
        "💻 Laptops",
        "🎧 Headphones",
        "📷 Cameras",
        "🛍️ All Products",
        "📦 Product Details",
    ],
    "MORE": [
        "📂 Data Ingestion & Validator",
        "📖 Methodology & Transparency",
        "📲 Install App",
    ],
}

# Flattened list of all valid navigation views
ALL_NAV_VIEWS: List[str] = [view for group_views in NAV_GROUPS.values() for view in group_views]


def get_group_for_view(view_name: str) -> str:
    """Return the name of the navigation group for a given view."""
    # Resolve aliases
    normalized = "🏠 Product Explorer & Storefront" if view_name == "🏠 Home / Discovery" else view_name
    for group_name, views in NAV_GROUPS.items():
        if normalized in views:
            return group_name
    return "ANALYZE"
