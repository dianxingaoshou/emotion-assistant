"""All mutable files stay inside this project; never alter the user's HOME."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def configure_environment():
    for path in (RUNTIME / "cache", RUNTIME / "tmp", DATA, MODELS, REPORTS):
        path.mkdir(parents=True, exist_ok=True)
    directories = {
        "HF_HOME": RUNTIME / "cache" / "huggingface",
        "HF_HUB_CACHE": RUNTIME / "cache" / "huggingface" / "hub",
        "TORCH_HOME": RUNTIME / "cache" / "torch",
        "MODELSCOPE_CACHE": RUNTIME / "cache" / "modelscope",
        "XDG_CACHE_HOME": RUNTIME / "cache",
        "MPLCONFIGDIR": RUNTIME / "cache" / "matplotlib",
        "NUMBA_CACHE_DIR": RUNTIME / "cache" / "numba",
        "TEMP": RUNTIME / "tmp",
        "TMP": RUNTIME / "tmp",
    }
    for key, value in directories.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    sys.pycache_prefix = str(RUNTIME / "cache" / "pycache")
