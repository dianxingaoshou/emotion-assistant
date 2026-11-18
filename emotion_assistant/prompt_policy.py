"""Session timing decisions, separate from estimating an emotion cue."""
from collections import deque

from .evidence import SustainedEvidenceGate


class ConflictGate(SustainedEvidenceGate):
    """Only sustained, fully covered, high-quality joint disagreement can clarify."""
    def __init__(self):
        super().__init__(required_seconds=12)

    def _qualifies(self, record):
        quality = record.get("per_modality_quality") or {}
        return (record.get("valid") is True and record.get("conflict") is True
                and record.get("status") == "joint" and record.get("visual_coverage", 0) >= .999
                and all(quality.get(modality, 0) >= .7 for modality in ("audio", "visual")))

    def observe(self, result, now):
        # Keep revisions even when they retract a formerly joint observation.
        if result.get("audio_event_id"):
            super().observe(result, now)

    def diagnostics(self, now):
        detail = super().diagnostics(now)
        if detail["ready"]:
            episodes = {item["audio_event_id"] for item in self._records.values()
                        if item["t_start"] < detail["window_end"] and item["t_end"] > detail["window_start"]
                        and self._qualifies(item)}
            detail["ready"] = len(episodes) >= 2
        return detail


class PromptTimingPolicy:
    version = "timing-v1"
    modes = ("available", "focused", "quiet")

    def __init__(self):
        self.mode = "available"
        self.history = deque(maxlen=128)

    def restore(self, history):
        self.history.clear()
        self.history.extend(history)

    def evaluate(self, now, evidence, cooldown=False):
        recent = [(kind, stamp) for kind, stamp in self.history if 0 <= now - stamp < 3600]
        asked = sum(kind == "asked" for kind, _ in recent)
        refusals = [stamp for kind, stamp in recent if kind == "declined"]
        if self.mode != "available":
            reason = self.mode
        elif cooldown or (refusals and now - max(refusals) < 1200):
            reason = "cooldown"
        elif len(refusals) >= 2:
            reason = "recent_declines"
        elif asked >= 2:
            reason = "hourly_budget"
        else:
            reason = evidence or "insufficient_evidence"
        action = "ask" if evidence and reason == evidence else "observe"
        return {"action": action, "reason": reason, "policy_version": self.version,
                "context": {"availability": self.mode, "recent_prompts": asked,
                            "recent_declines": len(refusals), "evidence": evidence}}
