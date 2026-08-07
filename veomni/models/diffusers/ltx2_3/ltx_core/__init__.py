import sys
from pathlib import Path


_ltx_core_parent = str(Path(__file__).resolve().parent.parent)
if _ltx_core_parent not in sys.path:
    sys.path.insert(0, _ltx_core_parent)
