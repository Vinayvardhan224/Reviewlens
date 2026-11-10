"""
Streamlit Multipage Entrypoint: ReviewLens Smart Shopping & Product Intelligence.
"""

import sys
from pathlib import Path
import runpy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

app_script = PROJECT_ROOT / "app.py"
runpy.run_path(str(app_script), run_name="__main__")
