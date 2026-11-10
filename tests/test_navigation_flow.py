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
        at = AppTest.from_string(code).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.radio[0].value, "Home")
        self.assertIn("Home", at.markdown[0].value)

        # 1. Click 'Browse Smartphones'
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.radio[0].value, "Smartphones")
        self.assertIn("Smartphones", at.markdown[0].value)

        # 2. Click 'Inspect Details'
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.radio[0].value, "Details")
        self.assertIn("Details", at.markdown[0].value)

        # 3. Select 'Home' on sidebar radio directly
        at.radio[0].set_value("Home").run()
        self.assertFalse(at.exception)
        self.assertEqual(at.radio[0].value, "Home")
        self.assertIn("Home", at.markdown[0].value)


if __name__ == "__main__":
    unittest.main()
