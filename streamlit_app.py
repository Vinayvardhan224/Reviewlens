"""
ReviewLens Entrypoint for Streamlit Cloud Deployment.
Redirects execution to app.py.
"""

import runpy
from pathlib import Path

app_script = Path(__file__).resolve().parent / "app.py"
runpy.run_path(str(app_script), run_name="__main__")
