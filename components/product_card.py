"""
ReviewLens Product Card Component.
Presents uniform, structured product cards with imagery, pricing in INR,
original price, discount savings badge, star rating, and actions.
"""

from typing import Dict, Any, Callable, Optional
import streamlit as st
from modules.data_loader import format_currency_inr


def render_product_card(
    prod: Dict[str, Any],
    is_in_compare: bool = False,
    on_inspect: Optional[Callable[[str], None]] = None,
    on_toggle_compare: Optional[Callable[[str], None]] = None
) -> None:
    """
    Renders an individual product discovery card inside a Streamlit container.
    """
    pid = prod.get("product_id", "")
    name = prod.get("product_name", "Product")
    brand = prod.get("brand", "")
    category = prod.get("category", "")
    price = float(prod.get("price", 0.0))
    orig_price = float(prod.get("original_price", price))
    rating = float(prod.get("rating", 4.0))
    review_count = int(prod.get("review_count", 0))
    img_url = prod.get("image_url", "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=600")
    amazon_url = prod.get("product_url") or prod.get("amazon_url") or f"https://www.amazon.in/s?k={name}"

    formatted_price = format_currency_inr(price)

    # Calculate discount savings
    price_html = f'<span style="font-size: 1.25rem; font-weight: 800; color: #0F172A;">{formatted_price}</span>'
    if orig_price > price:
        formatted_orig = format_currency_inr(orig_price)
        discount_pct = int(round((orig_price - price) / orig_price * 100))
        price_html += (
            f' <span style="font-size: 0.85rem; color: #94A3B8; text-decoration: line-through;">{formatted_orig}</span>'
            f' <span style="font-size: 0.72rem; font-weight: 700; color: #16A34A; background: #DCFCE7; padding: 0.15rem 0.45rem; border-radius: 4px;">{discount_pct}% OFF</span>'
        )

    card_html = (
        '<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; '
        'padding: 1rem; margin-bottom: 1.2rem; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04); transition: all 0.2s ease;">'
        '<div style="height: 160px; overflow: hidden; border-radius: 8px; margin-bottom: 0.8rem; '
        'background: #F8FAFC; display: flex; align-items: center; justify-content: center;">'
        f'<img src="{img_url}" style="max-height: 100%; max-width: 100%; object-fit: contain; border-radius: 6px;" '
        f'alt="{name}" onerror="this.onerror=null;this.src=\'https://placehold.co/400x300/e2e8f0/475569?text=Product+Image\';" />'
        '</div>'
        '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">'
        f'<span style="background: #F1F5F9; color: #475569; font-size: 0.75rem; font-weight: 600; padding: 0.2rem 0.5rem; border-radius: 4px;">{category} • {brand}</span>'
        '<span style="background: #EFF6FF; color: #2563EB; font-size: 0.72rem; font-weight: 600; padding: 0.2rem 0.45rem; border-radius: 4px;">🛡️ Demo Data</span>'
        '</div>'
        '<div style="font-weight: 700; font-size: 0.95rem; color: #0F172A; min-height: 2.6rem; line-height: 1.3rem; margin-bottom: 0.4rem; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;">'
        f'{name}'
        '</div>'
        f'<div style="display: flex; align-items: baseline; gap: 0.4rem; margin-bottom: 0.4rem; flex-wrap: wrap;">{price_html}</div>'
        '<div style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.85rem; color: #64748B; margin-bottom: 0.8rem;">'
        f'<span style="color: #F59E0B; font-weight: 700;">★ {rating:.1f}</span>'
        f'<span>({review_count:,} ratings)</span>'
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
            'View on Amazon India ↗'
            '</a>'
            '</div>'
        )
        st.markdown(link_html, unsafe_allow_html=True)
