"""Compare frozen teachers on CH-SIMS and a separately prepared CMMA test cohort."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_acoustic_continuity_review as acoustic_continuity_review
from scripts import chsims_teacher_acoustic_overlap_review as acoustic_overlap_review
from scripts import chsims_teacher_directional as directional
from scripts import chsims_teacher_directional_pair as pair
from scripts import chsims_teacher_dynamic_conflict as dynamic_conflict
from scripts import chsims_teacher_context_evidence as context_evidence
from scripts import chsims_teacher_context_review as context_review
from scripts import chsims_teacher_complete_review as complete_review
from scripts import chsims_teacher_complete_directional_review as complete_directional_review
from scripts import chsims_teacher_dual_evidence_review as dual_evidence_review
from scripts import chsims_teacher_minicpm_protected as minicpm_protected
from scripts import chsims_teacher_minicpm_review as minicpm_review
from scripts import chsims_teacher_neutral_guard as neutral_guard
from scripts import chsims_teacher_neutral_agreement_review as neutral_agreement_review
from scripts import chsims_teacher_neutral_corroboration as neutral_corroboration
from scripts import chsims_teacher_neutral_consensus_review as neutral_consensus_review
from scripts import chsims_teacher_neutral_pair_review as neutral_pair_review
from scripts import chsims_teacher_neutral_precision as neutral_precision
from scripts import chsims_teacher_neutral_precision_review as neutral_precision_review
from scripts import chsims_teacher_neutral_reliability_review as neutral_reliability_review
from scripts import chsims_teacher_precision_review as precision_review
from scripts import chsims_teacher_protected_continuity_review as protected_continuity_review
from scripts import chsims_teacher_refinement as refinement
from scripts import chsims_teacher_robustness as prior
from scripts import chsims_teacher_source_calibration as source_calibration
from scripts import chsims_teacher_source_guard as source_guard
from scripts import chsims_teacher_source_review as source_review
from scripts import chsims_teacher_supported_review as supported_review
from scripts import chsims_teacher_temporal_neutral_review as temporal_neutral_review
from scripts import chsims_teacher_timeline as timeline
from scripts import chsims_teacher_window_agreement as window_agreement
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_omni_model_comparison import AUDIO_MINI, predict as linear_predict
from scripts.chsims_teacher_cross_source import group_ids, labels, metrics
from scripts.chsims_teacher_data import CACHE, sha_file
from scripts.chsims_teacher_supervision import numbers, read_full_report
from scripts.evaluate_chsims_teacher_refinement import CASES, SUMMARY, condition_metrics, fixed_desktop, model_config


SOURCE_EXPERIMENTS = ("source-review", "source-guard")
AUDIO_REVIEWERS = {"minicpm_review": minicpm_review, "minicpm_protected": minicpm_protected,
                   "context_review": context_review, "window_agreement": window_agreement,
                   "source_calibration": source_calibration, "dynamic_conflict": dynamic_conflict,
                   "neutral_guard": neutral_guard, "complete_precision": neutral_guard,
                   "complete_precision_joint": neutral_guard, "supported_review": neutral_guard,
                   "supported_review_corroborated": neutral_guard, "supported_review_complete_only": neutral_guard,
                   "neutral_consensus_review": neutral_guard, "temporal_neutral_review": neutral_guard,
                   "neutral_consensus_acoustic": neutral_guard, "dual_evidence_review": neutral_guard,
                   "neutral_precision_review": neutral_guard, "neutral_pair_review": neutral_guard,
                   "neutral_reliability_review": neutral_guard, "acoustic_continuity_review": neutral_guard,
                   "protected_continuity_review": neutral_guard, "acoustic_overlap_review": neutral_guard,
                   "neutral_agreement_review": neutral_guard}
AUDIO_EXPERIMENTS = tuple(kind.replace("_", "-") for kind in AUDIO_REVIEWERS)
PRECISION_REVIEWERS = {"complete_precision": complete_review, "complete_precision_joint": complete_directional_review,
                       "supported_review": supported_review, "supported_review_corroborated": supported_review,
                       "supported_review_complete_only": supported_review, "neutral_consensus_review": supported_review,
                       "temporal_neutral_review": supported_review, "neutral_consensus_acoustic": supported_review,
                       "dual_evidence_review": supported_review, "neutral_precision_review": supported_review,
                       "neutral_pair_review": supported_review, "neutral_reliability_review": supported_review,
                       "acoustic_continuity_review": supported_review, "protected_continuity_review": supported_review,
                       "acoustic_overlap_review": supported_review, "neutral_agreement_review": supported_review}
TEMPORAL_NEUTRAL_REVIEWERS = {"neutral_consensus_review": neutral_consensus_review,
                              "temporal_neutral_review": temporal_neutral_review,
                              "neutral_consensus_acoustic": neutral_consensus_review,
                              "dual_evidence_review": dual_evidence_review,
                              "neutral_precision_review": neutral_precision_review,
                              "neutral_pair_review": neutral_pair_review,
                              "neutral_reliability_review": neutral_reliability_review,
                              "acoustic_continuity_review": acoustic_continuity_review,
                              "protected_continuity_review": protected_continuity_review,
                              "acoustic_overlap_review": acoustic_overlap_review,
                              "neutral_agreement_review": neutral_agreement_review}
NEUTRAL_PRECISION_REVIEWERS = {"supported_review": neutral_precision,
                               "supported_review_corroborated": neutral_corroboration, **TEMPORAL_NEUTRAL_REVIEWERS}
PRECISION_EXPERIMENTS = tuple(kind.replace("_", "-") for kind in PRECISION_REVIEWERS)
WEIGHT_REVIEWERS = ("dynamic_conflict", "neutral_guard", *PRECISION_REVIEWERS)
ACOUSTIC_REVIEWERS = ("source_calibration", *WEIGHT_REVIEWERS)
CONTEXT_REVIEWERS = ("context_review", "window_agreement", *ACOUSTIC_REVIEWERS)


def guarded_reference(qwen, mini, scores, available, model):
    result = scores.copy()
    for code, settings in model["corrections"].items():
        chosen = available == int(code)
        if not chosen.any() or settings["proposal"] == "previous":
            continue
        rule = model["audio_blends"][code]
        adjusted = refinement.blend_audio_encoders(qwen[chosen], mini[chosen],
                                                   rule["mini_share"], rule["when_uncertain"])
        result[chosen] = refinement.accept_corrections(scores[chosen], adjusted, mini[chosen], settings)
    return result


def inference(cache, report, data_cache, checkpoint, old_model, original, data, correction=None):
    rows = report["rows"]
    inputs = data["variants"]
    mini = refinement.mini_variants(data_cache, rows, inputs)
    configs = {method: {mask: entry[method] for mask, entry in old_model["heads"].items()}
               for method in ("qwen", "mini_hybrid")}
    fixed, emotion2vec = fixed_desktop(data_cache, report)
    observations = timeline.score_examples(data, checkpoint["specialists"], original["heads"], np.arange(len(rows)))
    if correction is not None:
        time = data["time"]
        desktop = {"no_visual": time["desktop"],
                   "synchronous_gap": directional.desktop_gap_scores(data_cache, rows, time)}
    predictions, diagnostics = {}, {}
    kind = correction.get("kind") if correction is not None else None
    compare_audio = kind in AUDIO_REVIEWERS
    compare_pair = compare_audio or kind in ("source_review", "source_guard")
    initial_weights = np.asarray(checkpoint["text_weight"])
    for name, (state, mask) in CASES.items():
        blocks = inputs[state]
        old, availability = prior.robust_predict(blocks, mask, original)
        reference = old["teacher"]
        mini_scores = refinement.new_predict(cache, mini[state], mask, configs["mini_hybrid"], None)
        qwen = refinement.new_predict(cache, blocks, mask, configs["qwen"], reference)
        teacher_previous = guarded_reference(qwen, mini_scores, reference, availability, old_model)
        teacher = teacher_previous.copy()
        unguarded = teacher_previous.copy()
        methods = {"teacher_previous": labels(teacher_previous), "teacher_robust": labels(reference),
                   "minicpm_audio_hybrid": labels(mini_scores), "desktop_fixed_v1": fixed[name]}
        if name in observations:
            scoped = observations[name]
            initial, source, active, _, _ = timeline.example_features(scoped, data["time"], name, initial_weights)
            if not np.allclose(source[active], teacher_previous[active], atol=1e-9):
                raise ValueError("The timeline correction must be anchored to the same frozen teacher")
            unguarded[active] = timeline.apply_residual(initial, source, checkpoint["policies"][name], active)[active]

            policy = checkpoint["protected"][name]
            design, _, active, reliability, evidence = timeline.example_features(
                scoped, data["time"], name, np.asarray(policy["text_weight"]))
            candidate = timeline.apply_residual(design, source, policy, active)
            teacher[active] = timeline.protect_directions(source, candidate, evidence, policy["max_margin"],
                                                         policy["corroboration"])[active]

            x, _, eligible, _, equal_evidence = timeline.example_features(scoped, data["time"], name, None)
            if name == "synchronous_gap":
                x[:, -2:] = design[:, -2:]
            equal = teacher_previous.copy()
            candidate = timeline.apply_residual(x, source, policy, eligible)
            equal[eligible] = timeline.protect_directions(source, candidate, equal_evidence, policy["max_margin"],
                                                          policy["corroboration"])[eligible]
            methods["equal_evidence_weights"] = labels(equal)
            diagnostics[name] = {"eligible": active, "weight": reliability, "changed":
                                  np.asarray(labels(teacher)) != np.asarray(methods["teacher_previous"])}
            if correction is not None:
                if labels(desktop[name]) != fixed[name]:
                    raise ValueError(f"The score replay must use the original desktop's {name} inputs")
                batch = directional.describe_evidence(scoped, time, name, policy, desktop[name])
                if not np.allclose(batch["previous"][active], teacher[active], atol=1e-9):
                    raise ValueError(f"The directional model must start from the frozen timeline teacher in {name}")
        if correction is not None:
            methods["teacher_pre_timeline"] = methods["teacher_previous"]
            methods["teacher_previous"] = labels(teacher)
            if compare_pair:
                methods["teacher_timeline"] = methods["teacher_previous"]
            if name in observations:
                if compare_pair:
                    preceding = pair.predict(batch, correction["preceding"]["policies"][name])
                    teacher[active] = preceding[active]
                    methods["teacher_previous"] = labels(teacher)
                    if name == source_review.CONDITION:
                        reviews, disputed = source_review.predict(batch, preceding, correction["review"])
                        if "guard" in correction:
                            reviews = source_guard.predict(batch, preceding, reviews, correction["guard"])
                        teacher[active] = reviews[active]
                        diagnostics[name]["conflict"] = disputed
                    if compare_audio:
                        audio_review = correction["audio_review"]
                        voice = linear_predict(audio_review["voice"], {key: mini[state][key] for key in AUDIO_MINI}, None)
                        methods["teacher_previous"] = labels(teacher)
                        checked = minicpm_review.design(batch, voice, mini_scores, teacher,
                                                        minicpm_review.voice_quality(time, name))
                        if kind in CONTEXT_REVIEWERS:
                            checked.update(context_evidence.cues(scoped, time, name))
                        if kind in ACOUSTIC_REVIEWERS and name == source_calibration.NO_VISUAL:
                            checked.update(source_calibration.acoustic_inputs(rows, scoped["audio"]))
                            if kind in WEIGHT_REVIEWERS:
                                checked["quality"] = dynamic_conflict.quality_input(checked)
                        teacher = AUDIO_REVIEWERS[kind].infer(checked, audio_review["policies"][name]).copy()
                        if kind in PRECISION_REVIEWERS:
                            methods["teacher_previous"] = labels(teacher)
                            if name == precision_review.CONDITION and correction["precision"] is not None:
                                if kind in TEMPORAL_NEUTRAL_REVIEWERS:
                                    checked["retained"] = neutral_consensus_review.retained_text_audio(
                                        observations["synchronous_gap"]["windows"], time["windows"])
                                refiner = NEUTRAL_PRECISION_REVIEWERS.get(kind, precision_review)
                                teacher = refiner.infer(checked, teacher, correction["precision"])
                        diagnostics[name]["voice"] = voice
                        diagnostics[name]["voice_active"] = checked["active"]
                else:
                    corrector = pair if correction.get("kind") == "pair" else directional
                    teacher[active] = corrector.predict(batch, correction["policies"][name])[active]
                diagnostics[name]["changed"] = np.asarray(labels(teacher)) != np.asarray(methods["teacher_previous"])
        if kind in PRECISION_REVIEWERS and name == "complete":
            teacher = PRECISION_REVIEWERS[kind].infer(blocks, teacher, correction["complete"])
        if name in emotion2vec:
            methods["emotion2vec_scores"] = emotion2vec[name]
        methods["teacher_timeline_unguarded"] = labels(unguarded)
        methods["teacher"] = labels(teacher)
        if mask == 0 and any(label != "abstain" for arm in methods.values() for label in arm):
            raise ValueError("A system guessed an emotion from empty evidence")
        predictions[name] = methods
    return predictions, diagnostics, observations


def has_release_gain(report):
    cases = report["cases"]
    intact = cases["complete"]["official"]["metrics"]
    missing = report["averages"]["missing"]["official"]["metrics"]
    unseen = cases["no_visual"]["unseen_sources"]["metrics"]
    gap = cases["synchronous_gap"]["official"]["metrics"]
    references = ("teacher_previous", "teacher_robust")
    measures = ("accuracy", "macro_f1")
    return {"complete_maintains_accuracy_and_macro_f1": all(intact["teacher"][key] >= intact[arm][key]
                                                             for key in measures for arm in references),
            "missing_improves_accuracy_and_macro_f1": all(missing["teacher"][key] > missing[arm][key]
                                                          for key in measures for arm in references),
            "unseen_without_video_exceeds_original_desktop": unseen["teacher"]["accuracy"] > unseen["desktop_fixed_v1"]["accuracy"],
            "synchronous_gap_keeps_neutral_recall": gap["teacher"]["recall"]["neutral"]["hits"] >=
                                                     gap["teacher_previous"]["recall"]["neutral"]["hits"]}


def has_directional_gain(report):
    cases = report["cases"]
    intact = cases["complete"]["official"]["metrics"]
    missing = report["averages"]["missing"]["official"]["metrics"]
    no_video = cases["no_visual"]["unseen_sources"]["metrics"]
    gap = cases["synchronous_gap"]["unseen_sources"]["metrics"]
    keys = ("accuracy", "macro_f1")
    new, previous = (gap[arm]["recall"] for arm in ("teacher", "teacher_previous"))
    return {"complete_maintains_accuracy_and_macro_f1": all(intact["teacher"][key] >= intact["teacher_previous"][key]
                                                             for key in keys),
            "missing_improves_accuracy_and_macro_f1": all(missing["teacher"][key] > missing["teacher_previous"][key]
                                                          for key in keys),
            "unseen_without_video_exceeds_original_desktop": no_video["teacher"]["accuracy"] >
                                                            no_video["desktop_fixed_v1"]["accuracy"],
            "unseen_synchronous_gap_recovers_negative": new["negative"]["hits"] > previous["negative"]["hits"],
            "unseen_synchronous_gap_keeps_neutral_and_positive": all(new[cls]["hits"] >= previous[cls]["hits"]
                                                                   for cls in ("neutral", "positive"))}


def has_external_source_gain(result):
    no_video = result["cases"]["no_visual"]["external"]["metrics"]
    gap = result["cases"]["synchronous_gap"]["external"]["metrics"]
    return {"without_video_exceeds_original_desktop": all(no_video["teacher"][metric] >
                   no_video["desktop_fixed_v1"][metric] for metric in ("accuracy", "macro_f1")),
            "synchronous_gap_keeps_neutral_and_positive": all(gap["teacher"]["recall"][label]["hits"] >=
                   gap["teacher_previous"]["recall"][label]["hits"] for label in ("neutral", "positive"))}


def audio_release_metrics(result, external):
    groups = ("external",) if external else ("official", "unseen_sources")
    scenarios = [result["cases"][case][group]["metrics"] for case in timeline.CASES for group in groups]
    intact = [result["cases"]["complete"][group]["metrics"] for group in groups]
    missing = result["averages"]["missing"][groups[0]]["metrics"]
    measures = ("accuracy", "macro_f1")
    differences = [missing["teacher"][key] - missing["teacher_previous"][key] for key in measures]
    # Different orderings of per-condition scores can round an equal average differently.
    rounding = 1e-12
    return {"complete_maintains_accuracy_and_macro_f1": all(x["teacher"][k] >= x["teacher_previous"][k]
                                                             for x in intact for k in measures),
            "missing_no_regression": all(delta >= -rounding for delta in differences),
            "missing_improves_accuracy_and_macro_f1": all(delta > rounding for delta in differences),
            "preserves_polar_recall": all(x["teacher"]["recall"][label]["hits"] >=
                                          x["teacher_previous"]["recall"][label]["hits"] for x in scenarios
                                          for label in ("negative", "positive")),
            "neutral_recall_improves": any(x["teacher"]["recall"]["neutral"]["hits"] >
                                           x["teacher_previous"]["recall"]["neutral"]["hits"] for x in scenarios)}


def validate_external_exclusions(cache, manifest, rows):
    fingerprints = manifest.get("excluded_cohorts_sha256", [])
    if "excluded_cohort_sha256" in manifest:
        fingerprints = [manifest["excluded_cohort_sha256"]]
    if not fingerprints:
        return
    originals = []
    found = set()
    for folder in ("cmma-timeline", "cmma-source-review"):
        path = cache.parent / folder / "cohort.json"
        if path.is_file() and (digest := sha_file(path)) in fingerprints:
            originals.extend(json.loads(path.read_text("utf8"))["rows"])
            found.add(digest)
    old_conversations = {row["id"].split("/")[0] for row in originals}
    old_media = {row["raw_sha256"] for row in originals}
    if (found != set(fingerprints) or len(old_conversations) != manifest["excluded_conversations"]
            or old_conversations & set(group_ids(rows)) or old_media & {row["raw_sha256"] for row in manifest["rows"]}):
        raise ValueError("The new CMMA cohort must exclude the earlier conversations and videos")


def evaluate(cache, output, external=None, experiment=None):
    current_path = cache / timeline.DIRECTORY / "timeline-fusion.json"
    checkpoint = json.loads(current_path.read_text("utf8"))
    test = read_full_report(cache, "test")
    previous_path, old, original = model_config(cache, test)
    original_path = cache / "missing-evidence/fusion-heads.json"
    if (checkpoint["code"] != sha_file(Path(timeline.__file__)) or checkpoint["reference"] != sha_file(original_path)
            or checkpoint["metadata"] != test["meta_sha256"]):
        raise ValueError("The timeline model and previous teacher must be frozen against the same release")
    if experiment in PRECISION_EXPERIMENTS:
        joint = experiment == "complete-precision-joint"
        kind = experiment.replace("-", "_")
        correction = neutral_guard.load(cache, current_path)
        if kind == "neutral_consensus_acoustic":
            precision = neutral_consensus_review.load(cache, correction["splits"], acoustic_support=True)
            correction_path = neutral_consensus_review.ACOUSTIC_MODEL
        elif kind in NEUTRAL_PRECISION_REVIEWERS:
            refiner = NEUTRAL_PRECISION_REVIEWERS[kind]
            precision = refiner.load(cache, correction["splits"])
            correction_path = refiner.MODEL
        elif kind == "supported_review_complete_only":
            precision, correction_path = None, neutral_guard.MODEL
        else:
            precision = precision_review.load(cache, correction["splits"], joint)
            correction_path = precision_review.JOINT_MODEL if joint else precision_review.MODEL
        correction = {**correction, "kind": kind, "complete": PRECISION_REVIEWERS[kind].load(cache),
                      "precision": precision}
    elif experiment == "neutral-guard":
        correction, correction_path = neutral_guard.load(cache, current_path), neutral_guard.MODEL
    elif experiment == "dynamic-conflict":
        correction, correction_path = dynamic_conflict.load(cache, current_path), dynamic_conflict.MODEL
    elif experiment == "source-calibration":
        correction, correction_path = source_calibration.load(cache, current_path), source_calibration.MODEL
    elif experiment == "window-agreement":
        correction, correction_path = window_agreement.load(cache, current_path), window_agreement.MODEL
    elif experiment == "context-review":
        correction, correction_path = context_review.load(cache, current_path), context_review.MODEL
    elif experiment == "minicpm-protected":
        correction, correction_path = minicpm_protected.load(cache, current_path), minicpm_protected.MODEL
    elif experiment == "minicpm-review":
        correction, correction_path = minicpm_review.load(cache, current_path), minicpm_review.MODEL
    elif experiment == "source-guard":
        correction, correction_path = source_guard.load(cache, current_path), source_guard.MODEL
    elif experiment == "source-review":
        correction, correction_path = source_review.load(cache, current_path), source_review.MODEL
    elif experiment == "pair":
        correction, correction_path = pair.load(cache, current_path), pair.MODEL
    elif experiment == "directional":
        correction, correction_path = directional.load(cache, current_path), directional.DIRECTIONAL
    else:
        correction = None
    if correction is not None and checkpoint["splits"] != correction["splits"]:
        raise ValueError("The directional and timeline teachers must use the same development data")
    dataset = external or cache
    report = json.loads((external / "report.json").read_text("utf8")) if external else test
    rows = report["rows"]
    if external:
        manifest_path = external / "cohort.json"
        manifest = json.loads(manifest_path.read_text("utf8"))
        expected_rows = [(row["id"], row["truth"]) for row in manifest["rows"]]
        if (expected_rows != [(row["id"], row["truth"]) for row in rows]
                or manifest["labels_sha256"] != report["meta_sha256"]):
            raise ValueError("The external predictions do not match the frozen CMMA cohort")
        validate_external_exclusions(cache, manifest, rows)
    test_ids = {row["id"] for row in rows}
    training_sources = set()
    for phase, digest in checkpoint["splits"].items():
        if digest != sha_file(cache / f"evaluation-{phase}.json"):
            raise ValueError(f"The {phase} data changed after fitting")
        development = read_full_report(cache, phase)["rows"]
        if test_ids & {row["id"] for row in development}:
            raise ValueError("The test sample is in the development data")
        training_sources.update(group_ids(development))
    if report["models"] != original["models"]:
        raise ValueError("The previous and current teachers require the same encoders")
    data = timeline.read_data(dataset, report)
    if data["time"]["provenance"] != checkpoint["timeline"]:
        raise ValueError("The timeline extractor changed after training")
    expected = checkpoint["extractors"]
    if external:
        prep = {key: value for key, value in data["receipt"]["prepare"].items() if key != "code"}
        ref_prep = {key: value for key, value in expected["prepare"].items() if key != "code"}
        consistent = prep == ref_prep and all(data["receipt"][key] == expected[key] for key in ("omni", "desktop"))
    else:
        consistent = data["receipt"] == expected
    if not consistent:
        raise ValueError("Gap extraction differs from the frozen benchmark")

    prediction, trace, observations = inference(cache, report, dataset, checkpoint, old, original, data, correction)
    groups = {"external" if external else "official": np.arange(len(rows))}
    if correction is not None:
        if experiment in PRECISION_EXPERIMENTS:
            reference_file = "neutral-guard-test-predictions.json"
        elif experiment in AUDIO_EXPERIMENTS:
            reference_file = "source-guard-test-predictions.json"
        elif experiment in SOURCE_EXPERIMENTS:
            reference_file = "pair-test-predictions.json"
        else:
            reference_file = "test-predictions.json"
        frozen = json.loads((dataset / timeline.DIRECTORY / reference_file).read_text("utf8"))
        if frozen["ids"] != [row["id"] for row in rows]:
            raise ValueError("The previous timeline evaluator used different test IDs")
        for name, methods in prediction.items():
            for arm, reference in (("teacher_previous", "teacher"), ("desktop_fixed_v1", "desktop_fixed_v1"),
                                   ("minicpm_audio_hybrid", "minicpm_audio_hybrid")):
                if methods[arm] != frozen["cases"][name][reference]:
                    raise ValueError(f"Directional comparison changed frozen {arm} predictions in {name}")
    if not external:
        groups["unseen_sources"] = np.flatnonzero(~np.isin(group_ids(rows), list(training_sources)))
        prior_run = json.loads((cache / refinement.DIRECTORY / "test-predictions.json").read_text("utf8"))
        if prior_run["ids"] != [row["id"] for row in rows]:
            raise ValueError("The previous evaluator used a different test set")
        refinement_arm = "teacher_pre_timeline" if correction is not None else "teacher_previous"
        for name, methods in prediction.items():
            for method, old_key in ((refinement_arm, "teacher"), ("teacher_robust", "teacher_previous"),
                                    ("desktop_fixed_v1", "desktop_fixed_v1"), ("minicpm_audio_hybrid", "minicpm_audio_hybrid")):
                if methods[method] != prior_run["cases"][name][old_key]:
                    raise ValueError(f"Prior {method} replay changed under {name}")
    cases, diagnostics = {}, {}
    for name, methods in prediction.items():
        cases[name] = {cohort: condition_metrics(rows, methods, members) for cohort, members in groups.items()}
        if name in trace:
            values = trace[name]
            diagnostics[name] = {}
            for cohort, indices in groups.items():
                eligible = indices[values["eligible"][indices]]
                weights = values["weight"][eligible]
                if weights.ndim == 2:
                    weights = weights / np.maximum(weights.sum(1, keepdims=True), 1e-9)
                diagnostics[name][cohort] = {"eligible": len(eligible), "changed": int(values["changed"][indices].sum()),
                                             "weight_quantiles": np.quantile(weights, [.1, .5, .9], axis=0).tolist()
                                             if len(eligible) else []}
                if "voice_active" in values:
                    diagnostics[name][cohort]["voice_available"] = int(values["voice_active"][indices].sum())
        if "emotion2vec_scores" in methods and not external:
            cases[name]["audio_annotation"] = {cohort: condition_metrics(rows, methods, members, label="audio")
                                               for cohort, members in groups.items()}
        print(name, [(cohort, [(arm, score["correct"], round(score["macro_f1"], 4))
                                for arm, score in cases[name][cohort]["metrics"].items()]) for cohort in groups], flush=True)

    summaries = {}
    truth = numbers(rows, "full")
    arms = tuple(prediction["complete"])
    for name, conditions in SUMMARY.items():
        summaries[name] = {cohort: {
            "metrics": {arm: {metric: float(np.mean([cases[case][cohort]["metrics"][arm][metric] for case in conditions]))
                              for metric in ("accuracy", "macro_f1", "coverage")} for arm in arms},
            "teacher_minus_previous_95_source_bootstrap": prior.average_delta(rows, truth, prediction, conditions,
                                                                               indices, "teacher_previous"),
        } for cohort, indices in groups.items()}
    modality_metrics = {} if external else {channel: {cohort: metrics(numbers(rows, channel),
                                                                       observations["no_visual"][channel], indices)
                                                      for cohort, indices in groups.items()}
                                                  for channel in ("text", "audio")}
    result = {"model_sha256": sha_file(current_path), "reference_sha256": sha_file(previous_path),
              "test_ids_sha256": report["selected_ids_sha256"], "metadata_sha256": report["meta_sha256"],
              "cohorts": {cohort: {"n": len(indices), "groups": len(set(group_ids(rows)[indices]))}
                          for cohort, indices in groups.items()}, "cases": cases, "averages": summaries,
              "weights": diagnostics, "specialist_metrics": modality_metrics, "extractors": data["receipt"],
              "validation": {name: {key: policy[key] for key in ("share", "neutral_offset", "validation", "previous_validation")}
                             for name, policy in checkpoint["policies"].items()},
              "protected_validation": {
                  name: {key: policy[key] for key in ("share", "neutral_offset", "max_margin", "corroboration",
                                                     "validation", "previous_validation")}
                  for name, policy in checkpoint["protected"].items()}}
    if external:
        result["external_cohort"] = {"source": manifest["source"], "selection": manifest["selection"],
                                     "manifest_sha256": sha_file(manifest_path)}
        result["external_cohort"].update({key: manifest[key] for key in ("excluded_cohort_sha256",
                "excluded_cohorts_sha256", "excluded_conversations", "one_per_available_conversation") if key in manifest})
    else:
        result["release_metrics"] = has_directional_gain(result) if correction is not None else has_release_gain(result)
    if correction is not None:
        result["reference_sha256"] = sha_file(current_path)
        result["model_sha256"] = sha_file(cache / timeline.DIRECTORY / correction_path)
        if experiment in AUDIO_EXPERIMENTS:
            result["reference_sha256"] = correction["audio_review"]["guard"]
            result["release_metrics"] = audio_release_metrics(result, external)
            if experiment in PRECISION_EXPERIMENTS:
                result["reference_sha256"] = sha_file(cache / timeline.DIRECTORY / neutral_guard.MODEL)
                result["complete_model_sha256"] = sha_file(
                    cache / timeline.DIRECTORY / PRECISION_REVIEWERS[correction["kind"]].MODEL)
                fields = ("rule", "rules", "before", "after")
                result["complete_development"] = {key: correction["complete"][key] for key in fields
                                                  if key in correction["complete"]}
                if correction["precision"] is not None:
                    precision_fields = (*fields, "guard", "previous", "nearby_source_balanced_gain",
                                        "calibration_development")
                    result["precision_development"] = {key: correction["precision"][key] for key in precision_fields
                                                       if key in correction["precision"]}
            result["voice_review_validation"] = {case: {key: value for key, value in policy.items() if key != "residual"}
                                                 for case, policy in correction["audio_review"]["policies"].items()}
            if not external:
                audio_truth = numbers(rows, "audio")
                result["minicpm_voice_label_A"] = {case: {name: metrics(audio_truth, trace[case]["voice"], indices)
                        for name, indices in groups.items()} for case in timeline.CASES}
        elif experiment in SOURCE_EXPERIMENTS:
            result["reference_sha256"] = correction["previous_model"]
            result["source_review_validation"] = correction["review"]
            if "guard" in correction:
                result["source_guard_validation"] = correction["guard"]
            selected = trace[source_review.CONDITION]["conflict"]
            result["source_review_conflicts"] = {}
            for cohort, indices in groups.items():
                members = indices[selected[indices]]
                result["source_review_conflicts"][cohort] = condition_metrics(
                    rows, prediction[source_review.CONDITION], members) if len(members) else {"n": 0}
            if external:
                result["release_metrics"] = has_external_source_gain(result)
        else:
            summary_keys = ("C", "focus_on_disagreement") if experiment == "pair" else ("anchor", "penalty", "share")
            result["directional_validation"] = {case: {**policy, "rule": None if policy["rule"] is None else {
                key: policy["rule"][key] for key in summary_keys}} for case, policy in correction["policies"].items()}
    save_json(output, result)
    prediction_file = f"{experiment}-test-predictions.json" if correction is not None else "test-predictions.json"
    save_json(dataset / timeline.DIRECTORY / prediction_file,
              {"ids": [row["id"] for row in rows], "cases": prediction})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    parser.add_argument("--external", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--directional", action="store_const", const="directional", dest="experiment",
                       help="compare the frozen residual candidate")
    modes.add_argument("--pair", action="store_const", const="pair", dest="experiment",
                       help="compare the frozen negative–neutral review")
    modes.add_argument("--source-review", action="store_const", const="source-review", dest="experiment",
                       help="compare the frozen audio/window conflict review against the preceding pair model")
    modes.add_argument("--source-guard", action="store_const", const="source-guard", dest="experiment",
                       help="compare the frozen conflict review with neutral protection against the preceding pair model")
    modes.add_argument("--minicpm-review", action="store_const", const="minicpm-review", dest="experiment",
                       help="compare the frozen MiniCPM-o voice review against the released teacher")
    modes.add_argument("--minicpm-protected", action="store_const", const="minicpm-protected", dest="experiment",
                       help="compare the frozen polarity-protected voice review against the released teacher")
    modes.add_argument("--context-review", action="store_const", const="context-review", dest="experiment",
                       help="compare the frozen context-aware MiniCPM-o review against the released teacher")
    modes.add_argument("--window-agreement", action="store_const", const="window-agreement", dest="experiment",
                       help="compare the frozen review requiring agreement between retained modalities")
    modes.add_argument("--source-calibration", action="store_const", const="source-calibration", dest="experiment",
                       help="compare the source-calibrated acoustic and retained-window review")
    modes.add_argument("--dynamic-conflict", action="store_const", const="dynamic-conflict", dest="experiment",
                       help="compare source-learned text and acoustic weights under conflict")
    modes.add_argument("--neutral-guard", action="store_const", const="neutral-guard", dest="experiment",
                       help="compare the learned confirmation required for a negative review")
    modes.add_argument("--complete-precision", action="store_const", const="complete-precision", dest="experiment",
                       help="compare source-trained full-input and missing-visual precision reviews")
    modes.add_argument("--complete-precision-joint", action="store_const", const="complete-precision-joint", dest="experiment",
                       help="compare direction-specific full-input and missing-visual reviews")
    modes.add_argument("--supported-review", action="store_const", const="supported-review", dest="experiment",
                       help="compare full-input evidence checks and missing-visual neutral precision")
    modes.add_argument("--supported-review-corroborated", action="store_const", const="supported-review-corroborated",
                       dest="experiment", help="compare full-input evidence checks and two acoustic heads")
    modes.add_argument("--supported-review-complete-only", action="store_const", const="supported-review-complete-only",
                       dest="experiment", help="compare full-input evidence checks with the released missing-input policy")
    modes.add_argument("--neutral-consensus-review", action="store_const", const="neutral-consensus-review",
                       dest="experiment", help="compare grouped text and audio calibration for neutral predictions")
    modes.add_argument("--neutral-consensus-acoustic", action="store_const", const="neutral-consensus-acoustic",
                       dest="experiment", help="compare neutral calibration with an acoustic support requirement")
    modes.add_argument("--temporal-neutral-review", action="store_const", const="temporal-neutral-review",
                       dest="experiment", help="compare neutral review using retained text and acoustic evidence")
    modes.add_argument("--dual-evidence-review", action="store_const", const="dual-evidence-review",
                       dest="experiment", help="compare whole-clip and retained evidence for proposed neutral revisions")
    modes.add_argument("--neutral-precision-review", action="store_const", const="neutral-precision-review",
                       dest="experiment", help="compare a source-trained confirmation of neutral revisions")
    modes.add_argument("--neutral-pair-review", action="store_const", const="neutral-pair-review",
                       dest="experiment", help="compare source-trained positive and neutral odds without video")
    modes.add_argument("--neutral-reliability-review", action="store_const", const="neutral-reliability-review",
                       dest="experiment", help="compare reliability-aware confirmation of neutral revisions")
    modes.add_argument("--acoustic-continuity-review", action="store_const", const="acoustic-continuity-review",
                       dest="experiment", help="compare the continuity of whole-clip and retained speech evidence")
    modes.add_argument("--protected-continuity-review", action="store_const", const="protected-continuity-review",
                       dest="experiment", help="confirm speech revisions with the frozen neutral reliability review")
    modes.add_argument("--acoustic-overlap-review", action="store_const", const="acoustic-overlap-review",
                       dest="experiment", help="limit positive evidence from overlapping acoustic views")
    modes.add_argument("--neutral-agreement-review", action="store_const", const="neutral-agreement-review",
                       dest="experiment", help="calibrate same-track reinforcement against whole-clip neutral agreement")
    args = parser.parse_args()
    with threadpool_limits(limits=5, user_api="blas"):
        evaluate(args.cache, args.output, args.external, args.experiment)


if __name__ == "__main__":
    main()
