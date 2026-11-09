"""Evaluate the matcher on hand-written CV / job-description pairs.

Usage (from the repository root):

    python eval/evaluate.py

The 20 pairs in eval/pairs.json are synthetic and were written by hand. The
numbers show whether matching and non-matching pairs separate; they are not a
benchmark of real hiring outcomes.
"""
import json
import sys
from pathlib import Path

from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.analysis import analyze_texts  # noqa: E402

THRESHOLD = 0.5  # fixed in advance, not tuned on these pairs
SCORES = ("sbert_score", "tfidf_score", "advanced_score")


def main():
    pairs = json.loads((ROOT / "eval" / "pairs.json").read_text(encoding="utf-8"))
    rows = []
    for pair in pairs:
        result = analyze_texts(pair["cv"], pair["jd"])
        rows.append({"id": pair["id"], "label": pair["label"], **{k: round(result[k], 4) for k in SCORES}})

    y = [1 if r["label"] == "match" else 0 for r in rows]
    summary = {}
    for key in SCORES:
        pos = [r[key] for r in rows if r["label"] == "match"]
        neg = [r[key] for r in rows if r["label"] == "no-match"]
        preds = [1 if r[key] >= THRESHOLD else 0 for r in rows]
        summary[key] = {
            "mean_match": round(sum(pos) / len(pos), 4),
            "mean_no_match": round(sum(neg) / len(neg), 4),
            "min_match": min(pos),
            "max_no_match": max(neg),
            "roc_auc": round(float(roc_auc_score(y, [r[key] for r in rows])), 4),
            "accuracy_at_threshold": round(sum(p == t for p, t in zip(preds, y)) / len(y), 4),
        }

    print(f"{'pair':<22}{'label':<10}{'sbert':>7}{'tfidf':>7}{'adv':>7}")
    for r in rows:
        print(
            f"{r['id']:<22}{r['label']:<10}{r['sbert_score']:>7.3f}"
            f"{r['tfidf_score']:>7.3f}{r['advanced_score']:>7.3f}"
        )
    print()
    print(f"n = {len(rows)} ({sum(y)} match, {len(y) - sum(y)} no-match), threshold = {THRESHOLD}")
    print(f"{'score':<16}{'mean match':>11}{'mean no':>9}{'gap':>7}{'min match':>11}{'max no':>8}{'AUC':>6}{'acc':>6}")
    for key, s in summary.items():
        print(
            f"{key:<16}{s['mean_match']:>11.3f}{s['mean_no_match']:>9.3f}"
            f"{s['mean_match'] - s['mean_no_match']:>7.3f}{s['min_match']:>11.3f}"
            f"{s['max_no_match']:>8.3f}{s['roc_auc']:>6.2f}{s['accuracy_at_threshold']:>6.2f}"
        )

    out = ROOT / "eval" / "results.json"
    out.write_text(json.dumps({"threshold": THRESHOLD, "pairs": rows, "summary": summary}, indent=2) + "\n")
    print(f"\nWrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
