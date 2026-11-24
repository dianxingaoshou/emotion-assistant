"""Counterfactual scores for the same time interval and the same fusion policy."""


def explain_fusion(branches, result, evaluate_subset):
    """Drop one available modality, keeping the full result's target fixed.

    The deployed policy abstains on an empty input and supplies no probability.
    Its missing score stays undefined, including for single-modality attribution.
    """
    available = sorted(result.get("per_modality_scores", {}))
    scores = result.get("scores") or {}
    target = max(scores, key=scores.get) if scores else None
    explanation = {
        "method": "leave_one_modality_out_v1",
        "available": available,
        "target": target,
        "score": scores.get(target),
        "without": {},
    }
    if target is None:
        return explanation
    for removed in available:
        comparison = evaluate_subset({key: branches[key] for key in available if key != removed})
        subset_scores = comparison.get("scores") or {}
        value = subset_scores.get(target)
        explanation["without"][removed] = {
            "target_score": value,
            "decision": comparison["decision"],
            "score_difference": None if value is None else scores[target] - value,
        }
    return explanation


def evidence_text(explanation):
    """One human-readable counterfactual for each available sensor."""
    if not explanation or not explanation.get("target"):
        return ""
    names = {"visual": "画面", "audio": "声音"}
    labels = {"positive": "正向", "neutral": "中性", "negative": "负向", "uncertain": "未映射"}
    target = labels.get(explanation["target"], explanation["target"])
    entries = []
    for key, outcome in explanation["without"].items():
        value = outcome["target_score"]
        if value is None:
            return f"本段只有{names[key]}提供有效线索。移除后没有可用分数。"
        entries.append(f"去掉{names[key]} {value:.1%}")
    return f"同一时段的{target}分数：合用 {explanation['score']:.1%}，" + "，".join(entries) + "。"
