"""Where things live. Everything is relative to the repository root.

DATA and RESULTS can be redirected with the NMDS_DATA / NMDS_RESULTS environment
variables (the tests use this to write to a temporary directory).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
DATA = Path(os.environ.get("NMDS_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("NMDS_RESULTS", ROOT / "results"))
FIGURES = ROOT / "figures"
STYLE = CODE / "mpl_style.txt"  # matplotlib style file used by the figure notebooks
