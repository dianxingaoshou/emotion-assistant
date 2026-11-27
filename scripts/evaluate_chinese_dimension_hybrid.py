"""Compare Chinese character regression, local Qwen and their calibrated mixture on new CVAS sentences."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant import chinese_dimensions as zh
from emotion_assistant.chinese_affect_calibration import COEFFICIENTS, FITTED_MODEL, FITTED_RATER, apply_chinese_calibration
from scripts.calibrate_chinese_affect import MODEL_DIGEST, frozen_backend, infer, prepared
from scripts.evaluate_chinese_affect import DIGEST, URL
from scripts.evaluate_chinese_emotions import SOURCES, load_source, opencc_translation


def raw_matrix(sentences):
    columns, values, indptr = [], [], [0]
    for sentence in sentences:
        for k, count in zh.feature_counts(sentence).items():
            columns.append(k)
            values.append(1 + np.log(count))
        indptr.append(len(columns))
    return csr_matrix((values, columns, indptr), shape=(len(sentences), zh.BUCKETS))


def idf_from_train(matrix):
    frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
    return np.where(frequency > 0, np.log((1 + matrix.shape[0]) / (1 + frequency)) + 1, 0)


def tfidf(matrix, idf):
    return normalize(matrix.multiply(idf).tocsr())


def training(rows, translation):
    texts = [zh.normalized_text(row["text"], translation) for row in rows]
    matrix = raw_matrix(texts)
    gold = np.array([[row[axis] for axis in zh.AXES] for row in rows])
    regularizers = (0.1, 0.5, 1.0, 3.0, 10.0)
    cv = np.zeros((len(regularizers), len(zh.AXES)))
    for a, b in KFold(5, shuffle=True, random_state=1306).split(texts):
        idf = idf_from_train(matrix[a])
        x, validation = tfidf(matrix[a], idf), tfidf(matrix[b], idf)
        for j, alpha in enumerate(regularizers):
            ridge = Ridge(alpha=alpha, solver="lsqr", tol=1e-7).fit(x, gold[a])
            cv[j] += np.abs(ridge.predict(validation) - gold[b]).sum(axis=0)
    alphas = [regularizers[i] for i in np.argmin(cv, axis=0)]
    idf = idf_from_train(matrix)
    x = tfidf(matrix, idf)
    fits = [Ridge(alpha=alpha, solver="lsqr", tol=1e-7).fit(x, gold[:, j]) for j, alpha in enumerate(alphas)]
    known = sorted({ord(ch) for sentence in texts for ch in zh.HAN.findall(sentence)})
    parameters = {"idf": idf.astype("float32"), "weights": np.array([fit.coef_ for fit in fits], dtype="float32"),
                  "intercept": np.array([fit.intercept_ for fit in fits]),
                  "known_chars": np.array(known, dtype=np.uint32), "trad": np.array(list(translation), dtype=np.uint32),
                  "simp": np.array(list(translation.values()), dtype=np.uint32)}
    return parameters, alphas, fits


def predictions(rows, params, translation):
    lexical, semantic, gold = [], [], []
    known = frozenset(params["known_chars"].tolist())
    for row in rows:
        sentence = zh.normalized_text(row["text"], translation)
        supported = (len(row["text"]) <= 512 and zh.is_chinese_text(row["text"])
                     and zh.evidence(sentence, known)[0])
        lexical.append(zh.score_normalized(sentence, params).round(4) if supported else [np.nan, np.nan])
        qwen = row["result"]
        if qwen["status"] == "ok":
            semantic.append([np.clip(COEFFICIENTS[axis][0] + COEFFICIENTS[axis][1] * qwen[axis], 1, 9)
                             for axis in zh.AXES])
        else:
            semantic.append([np.nan, np.nan])
        gold.append([row[axis] for axis in zh.AXES])
    return np.array(lexical), np.array(semantic), np.array(gold)


def fit_mixture(lexical, semantic, gold):
    valid = np.isfinite(lexical).all(axis=1) & np.isfinite(semantic).all(axis=1)
    delta = semantic[valid] - lexical[valid]
    denom = (delta ** 2).sum(axis=0)
    weights = np.divide(((gold[valid] - lexical[valid]) * delta).sum(axis=0),
                        denom, out=np.zeros(2), where=denom > 0)
    return np.clip(weights, 0, 1)


def metrics(scores, gold):
    rows = {}
    for i, axis in enumerate(zh.AXES):
        x, y = scores[:, i], gold[:, i]
        corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 and np.ptp(x) > 1e-12 and np.ptp(y) > 1e-12 else None
        rows[axis] = {"mae": round(float(np.abs(x-y).mean()), 4) if len(x) else None,
                      "pearson": round(float(corr), 4) if corr is not None else None}
    return rows


def gains(baseline, alternative, gold):
    if len(gold) == 0:
        return {}
    diffs = np.abs(baseline-gold) - np.abs(alternative-gold)
    replicates = np.random.default_rng(91732).choice(len(gold), (5000, len(gold)))
    intervals = np.quantile(diffs[replicates].mean(axis=1), [0.025, 0.975], axis=0)
    return {axis: {"mae_reduction": round(float(diffs[:, j].mean()), 4),
                   "paired_bootstrap_95": [round(float(v), 4) for v in intervals[:, j]]}
            for j, axis in enumerate(zh.AXES)}


def compare(rows, params, translation, prior):
    offline, qwen, gold = predictions(rows, params, translation)
    both = np.isfinite(offline).all(axis=1) & np.isfinite(qwen).all(axis=1)
    mixed = (1 - params["mixture"]) * offline + params["mixture"] * qwen
    common = {"character_ridge": offline[both], "calibrated_qwen": qwen[both], "mixture": mixed[both]}
    # Evaluate all attempts as well. A missing branch falls back to the available branch;
    # rows with no output use the training mean solely for aggregate scoring.
    prior = np.broadcast_to(prior, gold.shape)
    ridge_all = np.where(np.isfinite(offline), offline, prior)
    mix_all = np.where(np.isfinite(mixed), mixed, np.where(np.isfinite(qwen), qwen, ridge_all))
    return {"attempted": len(rows), "offline_valid": int(np.isfinite(offline[:, 0]).sum()),
            "qwen_status_counts": dict(Counter(row["result"]["status"] for row in rows)), "paired_valid": int(both.sum()),
            "paired": {name: metrics(y, gold[both]) for name, y in common.items()},
            "paired_mae_reduction_vs_offline":
                {name: gains(offline[both], common[name], gold[both]) for name in ("calibrated_qwen", "mixture")},
            "all_with_declared_fallback": {"training_mean": metrics(prior, gold), "character_ridge": metrics(ridge_all, gold),
                 "qwen_with_training_mean": metrics(np.where(np.isfinite(qwen), qwen, prior), gold),
                 "mixture_with_available_branch": metrics(mix_all, gold),
                 "mixture_mae_reduction_vs_offline": gains(ridge_all, mix_all, gold)}}


def duplicate_key(row, translation):
    return "".join(ch for ch in zh.normalized_text(row["text"], translation) if ch.isalnum())


def verify_bundled(test, params, translation):
    lexical, semantic, _ = predictions(test, params, translation)
    paired = 0
    for row, ridge, qwen in zip(test, lexical, semantic):
        offline = zh.predict_chinese_dimensions(row["text"])
        if np.isfinite(ridge).all():
            actual = [offline["dimensions"][axis]["score"] for axis in zh.AXES]
            np.testing.assert_allclose(actual, ridge, rtol=0, atol=1e-7)
        elif offline["status"] == "ok":
            raise ValueError("Desktop model should have abstained")
        # Qwen cache contains validated grades and statuses. No original quotations leave the machine.
        validated = dict(row["result"], model=FITTED_MODEL, version=FITTED_RATER, scope="expressed_text")
        mixed = zh.blend_chinese_dimensions(offline, apply_chinese_calibration(validated))
        if np.isfinite(ridge).all() and np.isfinite(qwen).all():
            actual = [mixed["dimensions"][axis]["score"] for axis in zh.AXES]
            expected = (1 - params["mixture"]) * ridge + params["mixture"] * qwen
            np.testing.assert_allclose(actual, expected, rtol=0, atol=5.1e-5)
            paired += 1
        elif mixed["status"] == "ok":
            raise ValueError("Mixture should have abstained")
    return {"sentences_checked": len(test), "mixtures_checked": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cache_dir = ROOT / ".runtime/chinese-emobank"
    parser.add_argument("--data", type=Path, default=cache_dir / "CVAS_all.csv")
    parser.add_argument("--previous-cache", type=Path, default=cache_dir / "qwen-zh-va-calibration.jsonl")
    parser.add_argument("--fresh-cache", type=Path, default=cache_dir / "qwen-zh-va-new-confirmation.jsonl")
    parser.add_argument("--opencc-cache", type=Path, default=ROOT / ".runtime/cped")
    parser.add_argument("--download-normalizer", action="store_true", help="Download pinned OpenCC character table and license if missing")
    parser.add_argument("--verify-bundled", action="store_true", help="Compare the installed runtime and its abstentions on every new test sentence")
    parser.add_argument("--offline", action="store_true", help="Score existing local Qwen ratings without a server")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/chinese-dimension-hybrid.json")
    parser.add_argument("--export", type=Path, help="Write aggregate weights for the academic research desktop application")
    args = parser.parse_args()
    paths = [args.data, args.previous_cache, args.fresh_cache, args.output]
    if args.export:
        paths.append(args.export)
    if (len(set(p.resolve() for p in paths)) != len(paths) or args.output.suffix != ".json"
            or any(path.suffix != ".jsonl" for path in (args.previous_cache, args.fresh_cache))
            or (args.export and args.export.suffix != ".npz")):
        parser.error("Choose distinct source CSV, JSONL caches, JSON report and NPZ weights")
    rows = prepared(args.data)
    mapping = opencc_translation(args.opencc_cache, args.download_normalizer)
    translation = {code: ord(target) if isinstance(target, str) else target for code, target in mapping.items()}
    license_text = load_source(args.opencc_cache, "opencc-license.txt", args.download_normalizer)
    # The old pilot (96) and previous confirmation (160) are not used to train any branch.
    # Reserve the next 160 by exactly the same content-hash ordering, without consulting labels.
    calibration, fresh = rows[96:288], rows[448:608]
    blocked = {duplicate_key(row, translation) for row in rows[:608]}
    train = []
    for row in rows[608:]:
        key = duplicate_key(row, translation)
        if key and key not in blocked:
            train.append(row)
            blocked.add(key)
    prior_keys = {duplicate_key(r, translation) for r in rows[:448]}
    fresh_unique = []
    for r in fresh:
        key = duplicate_key(r, translation)
        if key and key not in prior_keys:
            fresh_unique.append(r["id"])
            prior_keys.add(key)
    params, alphas, fitted_estimators = training(train, translation)
    fingerprint = frozen_backend(args.offline)
    calibration = infer(calibration, args.previous_cache, fingerprint, 3, args.offline)
    calibration_ridge, calibration_qwen, calibration_gold = predictions(calibration, params, translation)
    params["mixture"] = fit_mixture(calibration_ridge, calibration_qwen, calibration_gold)
    test = infer(fresh, args.fresh_cache, fingerprint, 3, args.offline)
    # Confirm all unrounded runtime estimates with the sklearn fit before reporting scores.
    fresh_texts = [zh.normalized_text(r["text"], translation) for r in test]
    fresh_matrix = tfidf(raw_matrix(fresh_texts), params["idf"])
    reference = np.clip(np.stack([reg.predict(fresh_matrix) for reg in fitted_estimators], axis=1), 1, 9)
    runtime_scores = np.array([zh.score_normalized(s, params) for s in fresh_texts])
    max_delta = float(np.abs(runtime_scores-reference).max())
    if max_delta > 1e-5:
        raise ValueError("Desktop character scorer disagrees with sklearn reference")
    prior = np.array([np.mean([row[axis] for row in train]) for axis in zh.AXES])
    unique_ids = set(fresh_unique)
    report = {"version": zh.HYBRID, "source": URL, "source_sha256": DIGEST,
              "papers": ["Lee, Li & Yu (2022), Chinese EmoBank: Building Valence-Arousal Resources for Dimensional Sentiment Analysis, https://doi.org/10.1145/3489141",
                         "Yu et al. (2016), Building Chinese Affective Resources in Valence-Arousal Dimensions, https://aclanthology.org/N16-1066/"],
              "terms": "https://www.mis.yzu.edu.tw/nlplab/resources/chinese-emobank_download.html",
              "source_use": "Academic research only; no source text, labels or verbatim model quotations in released artefacts.",
              "script_variants": {"contributors": "OpenCC contributors; first, one-character candidate from TSCharacters in ver.1.1.9",
                                  "map_source": SOURCES["TSCharacters.txt"][0], "map_digest": SOURCES["TSCharacters.txt"][1],
                                  "license_source": SOURCES["opencc-license.txt"][0], "license_digest": SOURCES["opencc-license.txt"][1],
                                  "notice": "Apache-2.0 license included verbatim inside the weights archive."},
              "axes": list(zh.AXES), "scale": [1, 9], "rater_digest": MODEL_DIGEST, "rater_fingerprint": fingerprint,
              "selection": "SHA256(zh-va-v1|text) sorted, 1-based. Exclude old pilot 1-96 and old test 289-448; calibrate 97-288; new test 449-608; train from 609 onward. Prior studies used a complement-wide constant mean, but no fresh row was individually examined or rated by Qwen for previous tests. Public foundation-model pretraining overlap unknown.",
              "normalized_overlap_policy": "Training excludes duplicates within training and across all reserved pools after NFKC, Han script normalization, case folding, and punctuation/whitespace removal. Sensitivity subset additionally removes normalized prior or within-test copies.",
              "official_rows": len(rows), "fit_candidates": len(rows)-608, "fit_unique": len(train),
              "fit_alphas_valence_arousal": alphas, "ridge_method": "Character 1/2/3-grams, CRC32 65536 buckets, sublinear TF and smoothed IDF, L2 normalization; alpha per dimension by shuffled five-fold MAE only within training.",
              "mixture_method": "One bounded least squares coefficient per axis on the 192 calibration rows with both outputs available. Qwen uses the unchanged affine transform previously fitted on these calibration rows. Both weights fixed before comparing new test gold.",
              "calibration_attempted": len(calibration),
              "calibration_valid_for_both": int(np.isfinite(calibration_ridge[:, 0] + calibration_qwen[:, 0]).sum()),
              "qwen_weights": dict(zip(zh.AXES, params["mixture"].tolist())),
              "ridge_export_vs_sklearn_max_abs_delta": max_delta,
              "reproduction_note": "Before final export the development OS-specific Han conversion was replaced by fixed licensed OpenCC; the same splits, scoring policy, alphas grid and mixture procedure were rerun without optimizing on test. The frozen Qwen ratings were reused.",
              "new_test_ids_sha256": hashlib.sha256("\n".join(r["id"] for r in fresh).encode()).hexdigest(),
              "bootstrap": "5,000 paired sentence-resamples with replacement; unadjusted axis-specific percentile intervals for absolute-error reduction. Positive favors Qwen/mixture.",
              "fresh_confirmation": compare(test, params, translation, prior), "deduplicated_test_size": len(unique_ids)}
    if len(unique_ids) < len(test):
        report["fresh_confirmation_excluding_normalized_duplicates"] = compare([r for r in test if r["id"] in unique_ids],
                                                                              params, translation, prior)
    if args.verify_bundled:
        report["bundled_parity"] = verify_bundled(test, params, translation)
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.export, **params, opencc_license=license_text,
                            source_attribution="Academic research: Chinese EmoBank, Lee, Li & Yu (2022), Yu et al. (2016). No corpus sentences distributed. Includes first-candidate numeric OpenCC v1.1.9 character map, OpenCC contributors.")
        report["weights_sha256"] = hashlib.sha256(args.export.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({k: report[k] for k in ("fit_unique", "qwen_weights", "fresh_confirmation")}), flush=True)


if __name__ == "__main__":
    main()
