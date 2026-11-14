"""
ReviewLens Product Card Component.
Presents uniform, structured product cards with authorized local imagery,
pricing in INR, original price, discount savings badge, star rating, and actions.
All user/data strings are escaped against markup injection.
"""

from typing import Dict, Any, Callable, Optional
import html
import streamlit as st
from modules.data_loader import format_currency_inr, get_product_image_src


def render_product_card(
    prod: Dict[str, Any],
    is_in_compare: bool = False,
    on_inspect: Optional[Callable[[str], None]] = None,
    on_toggle_compare: Optional[Callable[[str], None]] = None
) -> None:
    """
    Renders an individual product discovery card inside a Streamlit container.
    Uses authorized locally stored images and transparent illustrative pricing labels.
    """
    pid = prod.get("product_id", "")
    raw_name = prod.get("product_name", "Product")
    raw_brand = prod.get("brand", "")
    raw_category = prod.get("category", "")
    price = float(prod.get("price", 0.0))
    orig_price = float(prod.get("original_price", price))
    rating = float(prod.get("rating", 4.0))
    review_count = int(prod.get("review_count", 0))

    # Escape HTML to prevent injection
    esc_name = html.escape(str(raw_name))
    esc_brand = html.escape(str(raw_brand))
    esc_category = html.escape(str(raw_category))

    # Retrieve authorized local image
    img_src = get_product_image_src(prod)
    amazon_url = prod.get("product_url") or prod.get("amazon_url") or f"https://www.amazon.in/s?k={raw_name}"

    formatted_price = format_currency_inr(price)

    # Calculate discount savings
    price_html = f'<span style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF;">{formatted_price}</span>'
    if orig_price > price:
        formatted_orig = format_currency_inr(orig_price)
        discount_pct = int(round((orig_price - price) / orig_price * 100))
        price_html += (
            f' <span style="font-size: 0.85rem; color: #94A3B8; text-decoration: line-through;">{formatted_orig}</span>'
            f' <span style="font-size: 0.72rem; font-weight: 700; color: #34D399; background: rgba(16, 185, 129, 0.2); padding: 0.15rem 0.45rem; border-radius: 4px;">{discount_pct}% OFF</span>'
        )

    card_html = (
        '<div style="background: rgba(22, 17, 46, 0.8); border: 1px solid rgba(155, 92, 255, 0.25); border-radius: 14px; '
        'padding: 1rem; margin-bottom: 1.2rem; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45); transition: all 0.25s ease;">'
        '<div style="height: 160px; overflow: hidden; border-radius: 10px; margin-bottom: 0.8rem; '
        'background: rgba(12, 9, 28, 0.75); border: 1px solid rgba(255, 255, 255, 0.05); display: flex; align-items: center; justify-content: center;">'
        f'<img src="{img_src}" style="max-height: 100%; max-width: 100%; object-fit: contain; border-radius: 8px;" '
        f'alt="{esc_name}" />'
        '</div>'
        '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">'
        f'<span style="background: rgba(155, 92, 255, 0.15); color: #C4B5FD; font-size: 0.74rem; font-weight: 600; padding: 0.2rem 0.5rem; border-radius: 6px;">{esc_category} • {esc_brand}</span>'
        '<span style="background: rgba(37, 99, 235, 0.18); color: #93C5FD; font-size: 0.72rem; font-weight: 600; padding: 0.2rem 0.45rem; border-radius: 6px;">🛡️ Demo (Illustrative)</span>'
        '</div>'
        '<div style="font-weight: 700; font-size: 0.95rem; color: #FFFFFF; min-height: 2.6rem; line-height: 1.3rem; margin-bottom: 0.4rem; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;">'
        f'{esc_name}'
        '</div>'
        f'<div style="display: flex; align-items: baseline; gap: 0.4rem; margin-bottom: 0.4rem; flex-wrap: wrap;">{price_html}</div>'
        '<div style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.82rem; color: #B9B3D0; margin-bottom: 0.8rem;">'
        f'<span style="color: #F59E0B; font-weight: 700;">★ {rating:.1f}</span>'
        f'<span>({review_count:,} sample ratings)</span>'
        '</div>'
        '</div>'
    )

    with st.container():
        st.markdown(card_html, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔍 Details", key=f"btn_inspect_{pid}", use_container_width=True):
                if on_inspect:
                    on_inspect(pid)

        with col2:
            btn_label = "✓ In Tray" if is_in_compare else "⚔️ Compare"
            btn_type = "primary" if is_in_compare else "secondary"
            if st.button(btn_label, key=f"btn_comp_{pid}", type=btn_type, use_container_width=True):
                if on_toggle_compare:
                    on_toggle_compare(pid)

        link_html = (
            '<div style="text-align: center; margin-top: 0.3rem;">'
            f'<a href="{amazon_url}" target="_blank" style="font-size: 0.78rem; color: #FF9900; font-weight: 600; text-decoration: none;">'
            'Search on Amazon India ↗'
            '</a>'
            '</div>'
        )
        st.markdown(link_html, unsafe_allow_html=True)
