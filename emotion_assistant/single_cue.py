"""Class-dependent recalibration for a version-matched, surviving cue channel."""
from functools import lru_cache
import math
from pathlib import Path

from .calibration import LABELS, MAPPING_VERSION, TARGET, fingerprint, read_json


def log_features(scores):
    logs = [math.log(max(1e-6, scores[label])) for label in LABELS]
    center = sum(logs) / len(logs)
    return [value - center for value in logs] + [1.0]


class SingleCueCalibration:
    def __init__(self, document):
        spec = {key: value for key, value in document.items() if key != "version"}
        if (document.get("schema") != "single-cue-log-probability-v1"
                or document.get("target") != TARGET or document.get("mapping_version") != MAPPING_VERSION
                or document.get("label_order") != list(LABELS) or document.get("version") != fingerprint(spec)):
            raise ValueError("invalid single-cue calibration")
        self.version = document["version"]
        self.branches = {}
        for key, fit in document["branches"].items():
            weights = fit["coefficients"]
            if (key not in ("visual", "audio") or not fit["model_version"]
                    or len(weights) != len(LABELS) or any(len(row) != len(LABELS) + 1 for row in weights)
                    or not all(math.isfinite(x) for row in weights for x in row)
                    or not 0 < fit["blend"] <= 1):
                raise ValueError("invalid single-cue coefficients")
            self.branches[key] = fit

    def apply(self, key, branch):
        fit = self.branches.get(key)
        if fit is None:
            return None, "not_selected_on_validation"
        if branch.get("versions") != [fit["model_version"]]:
            return None, "version_mismatch"
        scores = branch["scores"]
        features = log_features(scores)
        logits = [sum(a * b for a, b in zip(row, features)) for row in fit["coefficients"]]
        peak = max(logits)
        exp = [math.exp(value - peak) for value in logits]
        total = sum(exp)
        blend = fit["blend"]
        return {key: (1 - blend) * scores[key] + blend * value / total
                for key, value in zip(LABELS, exp)}, "applied"


@lru_cache(maxsize=1)
def bundled_single_cue():
    path = Path(__file__).parent / "assets" / "single-cue-calibration.json"
    try:
        return SingleCueCalibration(read_json(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, AttributeError, TypeError, OverflowError):
        return None
