"""
ReviewLens Entrypoint for Streamlit Cloud Deployment.
Redirects execution to app.py.
"""

import sys
import runpy
from pathlib import Path

# Purge any stale in-memory cached modules from previous hot reloads on Streamlit Cloud
for _mod_name in list(sys.modules.keys()):
    if _mod_name.startswith("modules.") or _mod_name.startswith("reviewlens.") or _mod_name.startswith("components."):
        sys.modules.pop(_mod_name, None)

app_script = Path(__file__).resolve().parent / "app.py"
runpy.run_path(str(app_script), run_name="__main__")
