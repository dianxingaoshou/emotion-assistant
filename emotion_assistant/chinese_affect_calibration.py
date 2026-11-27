"""Research calibration of reader-perceived Chinese VA, on the CVAS 1–9 scale.

Aggregate affine fits use 188 valid ratings out of 192 held-away calibration
sentences. Attribution: Chinese EmoBank, Lee, Li & Yu (2022). The original Qwen
outputs, including the source quote, remain available in the returned record.
"""

from .chinese_affect import AXES


VERSION = "cvas-zh-affine-v1"
FITTED_RATER = "qwen-zh-expressed-va-v1"
FITTED_MODEL = "qwen3:8b"
N_FIT = 188
# (intercept, nonnegative slope); fit before reading confirmation results.
COEFFICIENTS = {
    "valence": (2.374059286610004, 0.4980813013794064),
    "arousal": (3.2888845080251228, 0.3543039078855547),
}


def apply_chinese_calibration(result):
    """Augment a validated result from the fitted rating method; preserve refusals and raw scores."""
    if (result.get("status") != "ok" or result.get("model") != FITTED_MODEL
            or result.get("version") != FITTED_RATER or result.get("scope") != "expressed_text"
            or any(type(result.get(axis)) is not int or not 1 <= result[axis] <= 9 for axis in AXES)):
        return result
    scores = {}
    for axis in AXES:
        intercept, slope = COEFFICIENTS[axis]
        scores[axis] = min(9.0, max(1.0, intercept + slope * result[axis]))
    return {**result, "calibrated": scores, "calibration_version": VERSION,
            "calibration_n": N_FIT, "calibration_scope": "academic_research_CVAS"}
