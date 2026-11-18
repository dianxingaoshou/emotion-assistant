"""Calibration for the *mapped observable cue*, distinct from subjective distress.

Both branches expose mapped probabilities. Scaling log probabilities here fits
that common task, without treating emotion2vec probabilities as native logits.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


LABELS = ("positive", "neutral", "negative", "uncertain")
TARGET = "external_cue_positive_neutral_negative_unmapped_v1"
MAPPING_VERSION = "canonical-emotieff-emotion2vec-v1"
GROUPS = ("visual_only", "audio_only", "joint")
FEATURES = ("quality", "coverage", "top1", "margin", "entropy", "unmapped_mass",
            "disagreement_js", "joint", "speech_seconds_over_10", "age_seconds_over_8")
MIN_FIT = 24
MIN_GROUP = 12


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique)


def scale_probabilities(scores, temperature):
    logs = [math.log(max(1e-12, scores[key])) / temperature for key in LABELS]
    maximum = max(logs)
    exps = [math.exp(value - maximum) for value in logs]
    total = sum(exps)
    return dict(zip(LABELS, (p / total for p in exps)))


def gate_features(branch, disagreement, joint, speech_seconds):
    scores = branch["scores"]
    ordered = sorted(scores.values(), reverse=True)
    entropy = -sum(p * math.log(p) for p in ordered if p) / math.log(len(LABELS))
    return (branch["quality"], branch["coverage"], ordered[0], ordered[0] - ordered[1],
            entropy, scores["uncertain"], disagreement, float(joint),
            min(1.0, max(0, speech_seconds) / 10), min(1.0, max(0, branch["age"]) / 8))


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class BranchFit:
    modality: str
    model_version: str
    temperature: float
    intercept: float
    coefficients: tuple

    def reliability(self, features):
        log_odds = self.intercept + sum(a * b for a, b in zip(self.coefficients, features))
        return 1 / (1 + math.exp(-max(-30, min(30, log_odds))))


@dataclass(frozen=True)
class Calibration:
    status: str = "unfitted"
    version: str | None = None
    branches: tuple = ()
    groups: tuple = ()

    def resolve(self, branches, group):
        if self.status != "fitted":
            return self.status, {}, None
        fitted = {fit.modality: fit for fit in self.branches}
        if any(key not in fitted for key in branches) or group not in dict(self.groups):
            return "missing_modality", {}, None
        if any(branch["versions"] != [fitted[key].model_version] for key, branch in branches.items()):
            return "version_mismatch", {}, None
        return "fitted", fitted, dict(self.groups)[group]

    @classmethod
    def load(cls, path):
        try:
            data = read_json(Path(path).read_text(encoding="utf-8-sig"))
            return cls.from_document(data)
        except FileNotFoundError:
            return cls()
        except (ValueError, TypeError, OSError, KeyError, AttributeError, OverflowError):
            return cls("invalid_artifact")

    @classmethod
    def from_document(cls, data):
        if not isinstance(data, dict):
            raise ValueError("calibration artifact must be an object")
        version = data.get("version")
        if (data.get("schema") != 1 or data.get("target") != TARGET
                or data.get("mapping_version") != MAPPING_VERSION
                or data.get("features") != list(FEATURES)
                or not isinstance(data.get("branches"), dict) or not isinstance(data.get("groups"), dict)
                or version != fingerprint({k: v for k, v in data.items() if k != "version"})):
            raise ValueError("incompatible calibration artifact")
        branches = []
        for key, item in data["branches"].items():
            coefs, temp, intercept = item["coefficients"], item["temperature"], item["intercept"]
            if (key not in ("visual", "audio") or not isinstance(item["model_version"], str)
                    or not item["model_version"].strip() or not _finite(temp) or not .1 <= temp <= 10
                    or not _finite(intercept) or not isinstance(coefs, list) or len(coefs) != len(FEATURES)
                    or not all(_finite(v) for v in coefs)
                    or type(item["n_fit"]) is not int or item["n_fit"] < MIN_FIT):
                raise ValueError("invalid fitted modality")
            branches.append(BranchFit(key, item["model_version"], temp, intercept, tuple(coefs)))
        groups = []
        for key, item in data["groups"].items():
            threshold = item["threshold"]
            if (key not in GROUPS or type(item["n_policy"]) is not int or item["n_policy"] < MIN_GROUP
                    or (threshold is not None and (not _finite(threshold) or not 0 <= threshold <= 1))):
                raise ValueError("invalid modality decision policy")
            groups.append((key, threshold))
        if not branches or not groups:
            raise ValueError("insufficient fitted modalities or decision groups")
        return cls("fitted", version, tuple(branches), tuple(groups))
