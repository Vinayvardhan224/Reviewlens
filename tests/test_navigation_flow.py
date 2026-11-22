"""
Test Streamlit programmatic navigation flow with AppTest.
Verifies that navigate_to synchronizes with sidebar_nav_radio without raising
StreamlitWidgetAlreadyInstantiatedError across multiple button clicks.
"""

import unittest
from streamlit.testing.v1 import AppTest


class TestNavigationFlow(unittest.TestCase):

    def test_navigation_without_widget_instantiated_error(self):
        code = """
import streamlit as st

NAV_OPTIONS = ["Home", "Smartphones", "Details"]

if "nav_view" not in st.session_state:
    st.session_state["nav_view"] = "Home"

# Handle pending navigation before sidebar_nav_radio is instantiated
if st.session_state.get("pending_nav"):
    target = st.session_state.pop("pending_nav")
    st.session_state["sidebar_nav_radio"] = target
    st.session_state["nav_view"] = target

def navigate_to(view):
    st.session_state["nav_view"] = view
    st.session_state["pending_nav"] = view
    st.rerun()

# When sidebar_nav_radio is initialized in session_state, do NOT pass index
if "sidebar_nav_radio" not in st.session_state:
    st.session_state["sidebar_nav_radio"] = st.session_state["nav_view"]

sel = st.sidebar.radio("Go to", NAV_OPTIONS, key="sidebar_nav_radio")
st.session_state["nav_view"] = sel

st.markdown(f"Active Page: {st.session_state['nav_view']}")

if st.session_state["nav_view"] == "Home":
    if st.button("Browse Smartphones"):
        navigate_to("Smartphones")
elif st.session_state["nav_view"] == "Smartphones":
    if st.button("Inspect Details"):
        navigate_to("Details")
"""
        at = AppTest.from_string(code).run(timeout=10)
        self.assertFalse(at.exception)
        self.assertEqual(at.radio[0].value, "Home")
        self.assertIn("Home", at.markdown[0].value)

        # 1. Click 'Browse Smartphones'
        at.button[0].click().run(timeout=10)
        self.assertFalse(at.exception)
        self.assertEqual(at.radio[0].value, "Smartphones")
        self.assertIn("Smartphones", at.markdown[0].value)

        # 2. Click 'Inspect Details'
        at.button[0].click().run(timeout=10)
        self.assertFalse(at.exception)
        self.assertEqual(at.radio[0].value, "Details")
        self.assertIn("Details", at.markdown[0].value)

        # 3. Select 'Home' on sidebar radio directly
        at.radio[0].set_value("Home").run(timeout=10)
        self.assertFalse(at.exception)
        self.assertEqual(at.radio[0].value, "Home")
        self.assertIn("Home", at.markdown[0].value)

    def test_every_view_belongs_to_exactly_one_group(self):
        """Assert that every view in ReviewLens navigation belongs to exactly one group with zero overlap."""
        from modules.navigation import NAV_GROUPS, ALL_NAV_VIEWS, get_group_for_view

        expected_groups = ["ANALYZE", "COMPARE", "SHOP (demo)", "MORE"]
        self.assertEqual(list(NAV_GROUPS.keys()), expected_groups)

        seen_views = []
        for g_name, views in NAV_GROUPS.items():
            self.assertGreater(len(views), 0, f"Group {g_name} should not be empty")
            for v in views:
                self.assertNotIn(v, seen_views, f"View '{v}' is duplicated across groups")
                seen_views.append(v)
                self.assertEqual(get_group_for_view(v), g_name)

        self.assertEqual(len(seen_views), len(ALL_NAV_VIEWS))
        self.assertNotIn("🔐 Sign In / Profile", seen_views, "Sign In / Profile must be removed from radio nav list")


if __name__ == "__main__":
    unittest.main()
