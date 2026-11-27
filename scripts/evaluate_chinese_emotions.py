"""Train an offline single-utterance model and test it on CPED's held-out TV shows."""

import argparse
from collections import Counter
import csv
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
from urllib.request import urlopen

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant import chinese_emotions as runtime


CPED_COMMIT = "1e4b81c28a123f22387e06664f37e5dc9322380f"
CPED = f"https://raw.githubusercontent.com/scutcyr/CPED/{CPED_COMMIT}/data/CPED"
OPENCC = "https://raw.githubusercontent.com/BYVoid/OpenCC/ver.1.1.9"
SOURCES = {
    "train_split.csv": (CPED + "/train_split.csv", "7bce6a11f0bf0ac353d28e96f6cfa03ff5e83b95810647abb55d18ff7299e487"),
    "valid_split.csv": (CPED + "/valid_split.csv", "f434a04d21640bb38c849f3e9598c2b7b2fbdb49be6a7365b2ae016019bd389b"),
    "test_split.csv": (CPED + "/test_split.csv", "a4bcde678438ba1f95224d8e772e40f1f1df7c72f348796b3fafd421c2443a53"),
    "TSCharacters.txt": (OPENCC + "/data/dictionary/TSCharacters.txt", "6b5a0a799bea2bb22c001f635eaa3fc2904310f0c08addbff275477a80ecf09a"),
    "opencc-license.txt": (OPENCC + "/LICENSE", "b534e465949558eec2597b04f5092b5e161236a68dfbfd04d547592ac3964308"),
}
CLASSES = tuple(runtime.LABEL_NAMES)


def load_source(cache, name, download):
    path = cache / name
    url, digest = SOURCES[name]
    if path.is_file():
        data = path.read_bytes()
    elif download:
        with urlopen(url, timeout=90) as response:
            data = response.read()
    else:
        raise FileNotFoundError(f"Missing {path}. Use --download to obtain the official files.")
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Checksum mismatch for {name}")
    if not path.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data.decode("utf-8-sig")


def read_split(data):
    texts, gold, tvs = [], [], set()
    for row in csv.DictReader(StringIO(data, newline="")):
        texts.append(row["Utterance"])
        gold.append(CLASSES.index(row["Emotion"]))
        tvs.add(row["TV_ID"])
    return texts, np.asarray(gold), tvs


def opencc_translation(cache, download=False):
    mapping = {}
    for row in load_source(cache, "TSCharacters.txt", download).splitlines():
        source, candidates = row.split("\t")
        target = candidates.split()[0]
        if len(source) == len(target) == 1 and source != target:
            mapping[source] = target
    return str.maketrans(mapping)


def prepare_model(cache, download):
    translation = opencc_translation(cache, download)
    train, gold, shows = read_split(load_source(cache, "train_split.csv", download))
    normalized = [runtime.normalized_text(text, translation) for text in train]
    vec = TfidfVectorizer(analyzer=runtime.char_ngrams, min_df=3, max_features=30_000,
                          sublinear_tf=True, norm="l2")
    x_train = vec.fit_transform(normalized)
    freq = np.bincount(gold)
    weights = np.sqrt(len(gold) / (len(CLASSES) * freq))
    estimator = LogisticRegression(C=3, solver="lbfgs", max_iter=450, tol=0.0002,
                                   class_weight=dict(enumerate(weights)))
    with threadpool_limits(limits=1):
        estimator.fit(x_train, gold)
    if max(estimator.n_iter_) >= estimator.max_iter:
        raise RuntimeError("Training did not converge")
    terms = vec.get_feature_names_out()
    model = {"terms": {v: i for i, v in enumerate(terms)}, "labels": CLASSES, "translation": translation,
             "weights": estimator.coef_.T.astype("float32"), "bias": estimator.intercept_, "idf": vec.idf_}
    return model, terms, vec, estimator, normalized, gold, shows


def predictions(texts, model):
    scores = np.zeros((len(texts), len(CLASSES)))
    valid = np.zeros(len(texts), dtype=bool)
    for i, text in enumerate(texts):
        _, evidence = runtime.score_text(text, model)
        if evidence is not None:
            scores[i] = evidence
            valid[i] = True
    return scores, valid


def gated(scores, eligible, threshold):
    output = np.full(len(scores), -1)
    for i in np.flatnonzero(eligible):
        decision = runtime.select_label(scores[i], threshold)
        if decision is not None:
            output[i] = decision
    return output


def choose_threshold(scores, eligible, gold):
    # Select widest coverage achieving 45% accuracy with >=10% coverage on development.
    # If none achieves the target, maximize accepted accuracy subject to the coverage floor.
    candidates = []
    for value in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        output = gated(scores, eligible, value)
        accepted = output != -1
        if np.mean(accepted) >= .10:
            accuracy = np.mean(output[accepted] == gold[accepted])
            candidates.append((value, float(np.mean(accepted)), float(accuracy)))
    if not candidates:
        raise ValueError("Insufficient development coverage at all candidate thresholds")
    target = [c for c in candidates if c[2] >= .45]
    return max(target, key=lambda item: item[1]) if target else max(candidates, key=lambda item: item[2])


def metrics(gold, output):
    accepted = output >= 0
    correct = int(np.sum(gold == output))
    ids = np.arange(len(CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(gold, output, labels=ids, zero_division=0)
    per_class = {label: {"support": int(support[i]), "predicted": int(np.sum(output == i)),
                         "precision": round(float(precision[i]), 4), "recall": round(float(recall[i]), 4),
                         "f1": round(float(f1[i]), 4)} for i, label in enumerate(CLASSES)}
    return {"n": len(gold), "correct": correct, "incorrect": int(accepted.sum()) - correct,
            "abstained": int((~accepted).sum()), "coverage": round(float(accepted.mean()), 4),
            "accuracy_all": round(correct / len(gold), 4),
            "accuracy_accepted": round(correct / int(accepted.sum()), 4) if accepted.any() else None,
            "macro_f1_all": round(float(f1.mean()), 4), "per_class": per_class}


def run(cache, download, asset_output):
    model, terms, vec, estimator, training_norm, train_gold, train_shows = prepare_model(cache, download)
    dev, dev_gold, dev_shows = read_split(load_source(cache, "valid_split.csv", download))
    dev_scores, dev_eligible = predictions(dev, model)
    threshold, dev_coverage, dev_accuracy = choose_threshold(dev_scores, dev_eligible, dev_gold)
    mapping = model["translation"]
    payload = BytesIO()
    np.savez_compressed(payload, terms=np.asarray(terms, dtype=str), labels=np.asarray(CLASSES),
                        traditional="".join(map(chr, mapping)), simplified="".join(mapping.values()),
                        weights=model["weights"], bias=model["bias"], idf=model["idf"], threshold=threshold,
                        attribution="CPED: Chen et al. (2022). OpenCC ver.1.1.9: first-candidate character mapping; "
                                    "derived from TSCharacters.txt. OpenCC contributors.",
                        opencc_license=load_source(cache, "opencc-license.txt", download))
    data = payload.getvalue()
    exported = runtime.unpack_model(data)
    # The entire analysis below uses independent shows, after weights and gate were frozen.
    test, gold, test_shows = read_split(load_source(cache, "test_split.csv", download))
    if train_shows & dev_shows or train_shows & test_shows or dev_shows & test_shows:
        raise ValueError("Overlapping shows in official splits")
    scores, eligible = predictions(test, exported)
    test_norm = [runtime.normalized_text(t, mapping) for t in test]
    validation_norm = [runtime.normalized_text(t, mapping) for t in dev]
    seen = set(training_norm) | set(validation_norm)
    novel = np.asarray([t not in seen for t in test_norm])
    forced = np.where(eligible, scores.argmax(axis=1), -1)
    selected = gated(scores, eligible, threshold)
    expected = estimator.predict_proba(vec.transform(test_norm))
    error = float(np.max(np.abs(scores[eligible] - expected[eligible])))
    if error > 1e-5:
        raise RuntimeError("Desktop classifier failed score parity against training library")
    dominant = Counter(train_gold.tolist()).most_common(1)[0][0]
    report = {
        "task": "Chinese text only; one CPED emotion as annotated in TV dialogue",
        "source": "https://github.com/scutcyr/CPED", "paper": "https://arxiv.org/abs/2205.14727",
        "data_commit": CPED_COMMIT, "source_sha256": {name: entry[1] for name, entry in SOURCES.items()},
        "model": runtime.MODEL, "asset_sha256": hashlib.sha256(data).hexdigest(), "sklearn": sklearn.__version__,
        "split_counts": {"train": len(train_gold), "development": len(dev), "test": len(gold)},
        "show_counts": [len(train_shows), len(dev_shows), len(test_shows)], "overlapping_shows": 0,
        "model_inputs": ["Utterance"], "features": "character 1–3 grams, sublinear TF-IDF, up to 30000 features",
        "preprocessing": "NFKC, lowercase, OpenCC first-candidate traditional-to-simplified character mapping, strip whitespace",
        "training": "multinomial logistic regression C=3, inverse square-root class frequency, train split only",
        "gate": {"source": "official development split", "target_accepted_accuracy": .45,
                 "minimum_coverage": .10, "threshold": threshold, "minimum_margin": runtime.MIN_MARGIN,
                 "development_coverage": dev_coverage, "development_accepted_accuracy": dev_accuracy},
        "test_metrics": {"train_majority": metrics(gold, np.full_like(gold, dominant)),
                         "ungated_top1_among_eligible": metrics(gold, forced),
                         "desktop": metrics(gold, selected)},
        "new_wording_test": {"definition": "Normalized text never seen in training or development; test rows retained in order.",
                             "excluded_rows": int((~novel).sum()), "desktop": metrics(gold[novel], selected[novel])},
        "score_parity_max_abs": error,
        "scope": "Scripted mainland-Chinese TV dialogue. Reference annotators had context; the model only reads one utterance. "
                 "A single dominant expression, not simultaneous labels, felt intensity, or an individual's self-report.",
    }
    if asset_output:
        asset_output.parent.mkdir(exist_ok=True, parents=True)
        asset_output.write_bytes(data)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / ".runtime/cped")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--asset-output", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/cped-chinese-emotions.json")
    args = parser.parse_args()
    report = run(args.data, args.download, args.asset_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for method, result in report["test_metrics"].items():
        print(method, {k: v for k, v in result.items() if k != "per_class"})
    print("asset SHA-256:", report["asset_sha256"])
