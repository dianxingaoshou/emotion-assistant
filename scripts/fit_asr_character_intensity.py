"""Train one ASR-specific auxiliary regressor on source- and text-disjoint competition training data."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from benchmark_mosei_upload import attachment_rows
from extract_asr_intensity import BASE_SEED, PRIVATE
from fit_english_intensity import read_splits
from emotion_assistant.asr_word_intensity import SCHEMA, char_terms, version_of
from emotion_assistant.english_polarity import tokens


HOMOPHONES = (("their", "there", "they're"), ("your", "you're"), ("it's", "its"), ("to", "too", "two"),
              ("then", "than"), ("which", "witch"), ("right", "write"), ("hear", "here"), ("no", "know"),
              ("for", "four"), ("one", "won"), ("our", "hour"), ("see", "sea"), ("by", "buy", "bye"),
              ("would", "wood"), ("whole", "hole"), ("new", "knew"), ("where", "wear"), ("be", "bee"))
ALTERNATIVES = {w: [other for other in group if other != w] for group in HOMOPHONES for w in group}


def corrupt(text, rng):
    result = []
    for word in tokens(text):
        if rng.random() < .12:
            continue
        if word in ALTERNATIVES and rng.random() < .08:
            word = rng.choice(ALTERNATIVES[word])
        result.append(word)
    return " ".join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()
    book, = args.data.rglob("label.xlsx")
    raw, = args.data.rglob("label-100.xlsx")
    videos = attachment_rows(raw)
    sources = {path.parent.name for _, path, _, _ in videos}
    words = {tuple(tokens(t)) for _, _, _, t in videos}
    rows = [r for r in read_splits(book)["train"] if r[0] not in sources and tuple(tokens(r[1])) not in words]
    x, y, rng = [], [], random.Random(BASE_SEED)
    for _, text, label, _ in rows:
        x.extend((text, corrupt(text, rng)))
        y.extend((label, label))
    features = TfidfVectorizer(analyzer=char_terms, max_features=12000, min_df=3, sublinear_tf=True)
    design = features.fit_transform(x)
    ridge = Ridge(alpha=4., solver="lsqr", tol=1e-7).fit(design, y)
    model = {"schema": SCHEMA, "intercept": round(float(ridge.intercept_), 9),
             "terms": {term: [round(float(features.idf_[i]), 9), round(float(ridge.coef_[i]), 9)]
                       for term, i in features.vocabulary_.items()}}
    model["model_version"] = version_of(model)
    payload = json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    (PRIVATE / "asr-character-intensity.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    meta = {"training_workbook_sha256": hashlib.sha256(book.read_bytes()).hexdigest(), "excluded_raw_sources": len(sources),
            "clean_train_clips": len(rows), "clean_train_sources": len({r[0] for r in rows}),
            "training_rows_after_augmentation": len(y), "alpha": 4., "ngram_sizes": [3, 4, 5],
            "max_features": 12000, "features": len(model["terms"]), "seed": BASE_SEED,
            "delete_probability": .12, "alternate_probability_on_remaining_listed_words": .08,
            "model_version": model["model_version"]}
    (PRIVATE / "asr-character-training.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
