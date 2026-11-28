"""Compare frozen teachers on CH-SIMS and a separately prepared CMMA test cohort."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_directional as directional
from scripts import chsims_teacher_directional_pair as pair
from scripts import chsims_teacher_refinement as refinement
from scripts import chsims_teacher_robustness as prior
from scripts import chsims_teacher_source_guard as source_guard
from scripts import chsims_teacher_source_review as source_review
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import group_ids, labels, metrics
from scripts.chsims_teacher_data import CACHE, sha_file
from scripts.chsims_teacher_supervision import numbers, read_full_report
from scripts.evaluate_chsims_teacher_refinement import CASES, SUMMARY, condition_metrics, fixed_desktop, model_config


SOURCE_EXPERIMENTS = ("source-review", "source-guard")


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
    compare_pair = correction is not None and correction.get("kind") in ("source_review", "source_guard")
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
                else:
                    corrector = pair if correction.get("kind") == "pair" else directional
                    teacher[active] = corrector.predict(batch, correction["policies"][name])[active]
                diagnostics[name]["changed"] = np.asarray(labels(teacher)) != np.asarray(methods["teacher_previous"])
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


def evaluate(cache, output, external=None, experiment=None):
    current_path = cache / timeline.DIRECTORY / "timeline-fusion.json"
    checkpoint = json.loads(current_path.read_text("utf8"))
    test = read_full_report(cache, "test")
    previous_path, old, original = model_config(cache, test)
    original_path = cache / "missing-evidence/fusion-heads.json"
    if (checkpoint["code"] != sha_file(Path(timeline.__file__)) or checkpoint["reference"] != sha_file(original_path)
            or checkpoint["metadata"] != test["meta_sha256"]):
        raise ValueError("The timeline model and previous teacher must be frozen against the same release")
    if experiment == "source-guard":
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
        if "excluded_cohort_sha256" in manifest:
            earlier_manifest = cache.parent / "cmma-timeline/cohort.json"
            earlier = json.loads(earlier_manifest.read_text("utf8"))["rows"]
            old_conversations = {row["id"].split("/")[0] for row in earlier}
            old_media = {row["raw_sha256"] for row in earlier}
            if (manifest["excluded_cohort_sha256"] != sha_file(earlier_manifest)
                    or len(old_conversations) != manifest["excluded_conversations"]
                    or old_conversations & set(group_ids(rows))
                    or old_media & {row["raw_sha256"] for row in manifest["rows"]}):
                raise ValueError("The new CMMA cohort must exclude the earlier conversations and videos")
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
        reference_file = "pair-test-predictions.json" if experiment in SOURCE_EXPERIMENTS else "test-predictions.json"
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
        if "excluded_cohort_sha256" in manifest:
            result["external_cohort"].update({key: manifest[key] for key in
                    ("excluded_cohort_sha256", "excluded_conversations")})
    else:
        result["release_metrics"] = has_directional_gain(result) if correction is not None else has_release_gain(result)
    if correction is not None:
        result["reference_sha256"] = sha_file(current_path)
        result["model_sha256"] = sha_file(cache / timeline.DIRECTORY / correction_path)
        if experiment in SOURCE_EXPERIMENTS:
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
    args = parser.parse_args()
    with threadpool_limits(limits=5, user_api="blas"):
        evaluate(args.cache, args.output, args.external, args.experiment)


if __name__ == "__main__":
    main()
