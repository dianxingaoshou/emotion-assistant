"""Conditional fusion and selective decisions for aligned, eligible modalities."""
from dataclasses import asdict, dataclass
import math

from .calibration import Calibration, LABELS, MAPPING_VERSION, fingerprint, gate_features, scale_probabilities
from .evidence import jensen_shannon_divergence, raw_prediction_diagnostics
from .single_cue import bundled_single_cue


@dataclass(frozen=True)
class FusionConfig:
    strategy: str = "quality_v2"
    min_quality: float = .2
    visual_weight: float = 1.0
    audio_weight: float = 1.0
    conflict_js_threshold: float = .18
    conflict_dominance: float = .6
    opposite_threshold: float = .5
    mapping_version: str = MAPPING_VERSION
    version: str = "aligned-cue-policy-v1"
    adapt_single_modality: bool = False

    def __post_init__(self):
        if self.strategy not in ("quality_v2", "calibrated_fusion") or self.mapping_version != MAPPING_VERSION:
            raise ValueError("unsupported fusion policy")
        if type(self.adapt_single_modality) is not bool:
            raise ValueError("adapt_single_modality must be a boolean")
        for number in (self.min_quality, self.conflict_js_threshold, self.conflict_dominance,
                       self.opposite_threshold, self.visual_weight, self.audio_weight):
            if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number):
                raise ValueError("invalid fusion parameter")
        if (not all(0 <= x <= 1 for x in (self.min_quality, self.conflict_js_threshold,
                                          self.conflict_dominance, self.opposite_threshold))
                or min(self.visual_weight, self.audio_weight) <= 0):
            raise ValueError("invalid quality, conflict threshold or base weight")


def divergence_of(branches):
    if "visual" in branches and "audio" in branches:
        return jensen_shannon_divergence(branches["visual"]["scores"], branches["audio"]["scores"])
    return None


class FusionPolicy:
    def __init__(self, config, calibration=None, baseline_version=None, configuration_issue=None,
                 *, single_cue_model=None):
        self.settings = config
        self.calibration = calibration if config.strategy == "calibrated_fusion" and calibration else Calibration()
        self.configuration_issue = configuration_issue
        self.single_cue = single_cue_model if single_cue_model is not None else (
            bundled_single_cue() if config.adapt_single_modality else None)
        basis = {**asdict(config), "calibration_version": self.calibration.version,
                 "calibration_load_status": self.calibration.status, "configuration_issue": configuration_issue}
        if config.adapt_single_modality:
            basis["single_cue_version"] = self.single_cue.version if self.single_cue else None
        else:
            # Preserve the fingerprint of historical unadapted policies.
            basis.pop("adapt_single_modality")
        self.basis_hash = fingerprint(basis)
        self._config = {**basis, "baseline_version": baseline_version}
        self.config_hash = fingerprint(self._config)

    @property
    def config(self):
        return dict(self._config)

    def _conflict(self, branches, divergence):
        if divergence is None:
            return False
        v, a = branches["visual"]["scores"], branches["audio"]["scores"]
        vtop, atop = max(v, key=v.get), max(a, key=a.get)
        cfg = self.settings
        return ((v["positive"] >= cfg.opposite_threshold and a["negative"] >= cfg.opposite_threshold)
                or (a["positive"] >= cfg.opposite_threshold and v["negative"] >= cfg.opposite_threshold)
                or (vtop != atop and "uncertain" not in (vtop, atop)
                    and min(v[vtop], a[atop]) >= cfg.conflict_dominance
                    and divergence >= cfg.conflict_js_threshold))

    def evaluate(self, branches):
        from .fusion_explanation import explain_fusion
        result = self._evaluate(branches)
        result["evidence_explanation"] = explain_fusion(branches, result, self._evaluate)
        return result

    def _evaluate(self, branches):
        """Return decisions and separate raw canonical scores, quality and fitted reliability."""
        cfg = self.settings
        branches = {k: v for k, v in branches.items() if v["quality"] * v["coverage"] > 0}
        group = "joint" if len(branches) == 2 else next(iter(branches)) + "_only" if branches else "unavailable"
        state, fitted, threshold = ("disabled", {}, None)
        if cfg.strategy == "calibrated_fusion":
            state, fitted, threshold = self.calibration.resolve(branches, group)
        diagnostics = {k: v.get("diagnostics") or raw_prediction_diagnostics(v["scores"])
                       for k, v in branches.items()}
        result = {"status": group, "calibration_status": state, "calibration_version": self.calibration.version,
                  "strategy_used": "calibrated_fusion" if state == "fitted" else "quality_v2",
                  "per_modality_scores": {k: dict(v["scores"]) for k, v in branches.items()},
                  "per_modality_output_scores": {},
                  "per_modality_quality": {k: v["quality"] for k, v in branches.items()},
                  "per_modality_coverage": {k: v["coverage"] for k, v in branches.items()},
                  "per_modality_age_seconds": {k: v["age"] for k, v in branches.items()},
                  "speech_seconds": branches.get("audio", {}).get("duration"),
                  "predictive_uncertainty": diagnostics, "reliability": {}, "unmapped_mass": None,
                  "input_quality": 0.0, "decision": "abstain", "decision_reasons": [],
                  "fusion_basis_hash": self.basis_hash}
        result.update(available_modalities=sorted(branches),
                      missing_modalities=sorted({"visual", "audio"}.difference(branches)),
                      missing_modality_calibration="not_applicable")
        reasons = result["decision_reasons"]
        if self.configuration_issue:
            reasons.append(self.configuration_issue)
        if cfg.strategy == "calibrated_fusion" and state != "fitted":
            reasons.append("fallback_" + state)
        if not branches:
            reasons.append("no_usable_modalities")
            return result
        divergence = divergence_of(branches)
        conflict = self._conflict(branches, divergence)
        base = {"visual": cfg.visual_weight, "audio": cfg.audio_weight}
        effective = {k: v["quality"] * v["coverage"] for k, v in branches.items()}
        weights = {k: base[k] * effective[k] for k in branches}
        result["input_quality"] = sum(weights[k] * effective[k] for k in branches) / sum(weights.values())
        mapped = {k: v["scores"] for k, v in branches.items()}
        if state == "fitted":
            duration = branches.get("audio", {}).get("duration", 0)
            for k, branch in branches.items():
                features = gate_features(branch, divergence or 0, group == "joint", duration)
                reliability = fitted[k].reliability(features)
                result["reliability"][k] = reliability
                weights[k] *= reliability
                mapped[k] = scale_probabilities(branch["scores"], fitted[k].temperature)
        elif len(branches) == 1 and cfg.adapt_single_modality:
            key = next(iter(branches))
            if self.single_cue is None:
                result["missing_modality_calibration"] = "artifact_unavailable"
            else:
                adjusted, status = self.single_cue.apply(key, branches[key])
                result["missing_modality_calibration"] = status
                if adjusted is not None:
                    mapped[key] = adjusted
                    result["strategy_used"] = "single_cue_calibration"
        denominator = sum(weights.values())
        weights = {k: weights.get(k, 0) / denominator for k in ("visual", "audio")}
        scores = {label: sum(weights[k] * mapped[k][label] for k in branches) for label in LABELS}
        result["per_modality_output_scores"] = {k: dict(scores) for k, scores in mapped.items()}
        top = max(scores, key=scores.get)
        result.update(weights=weights, scores=scores, valid=True, quality=result["input_quality"], conflict=conflict,
                      conflict_diagnostics={"jensen_shannon": divergence, "threshold": cfg.conflict_js_threshold},
                      disagreement_js=divergence or 0.0, decision_threshold=threshold,
                      unmapped_mass=sum(weights[k] * branches[k]["scores"]["uncertain"] for k in branches),
                      reason="modalities_conflict_user_confirmation" if conflict else group)
        if conflict:
            reasons.append("modalities_disagree")
        elif state == "fitted" and (threshold is None or scores[top] < threshold):
            reasons.append("below_" + group + "_acceptance")
        elif top == "uncertain":
            reasons.append("unmapped_class")
        else:
            result["decision"] = "ask" if top == "negative" else "observe_only"
            reasons.append("negative_candidate" if top == "negative" else "no_negative_candidate")
        return result
