"""External evaluation on the hand-written India-style test set.

    python evaluate.py

``data/india_test.csv`` (25 scams + 25 normal messages) is NEVER used for training
or for choosing weights / thresholds -- those come from the validation split in
train.py.  This is a deliberately hard, out-of-distribution check: the training data
is English spam / Enron-era email, the test messages are modern Indian SMS scams.

Caveats (please read before quoting any number):
  * n = 50, so every accuracy figure has a wide confidence interval (printed below).
  * The rule patterns were written / extended with knowledge of these 50 messages, so the
    "Rules only" row and the rule part of the "Hybrid" rows are optimistic and are NOT an
    unbiased estimate.  Only the "Model only" row is clean.  For an honest final number,
    write ~50 NEW messages without looking at the rules and evaluate once.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from common import ROOT, clean, hybrid_score, load_artifacts, red_flags, rule_score

vec, model, cfg = load_artifacts()
data = pd.read_csv(ROOT / "data/india_test.csv")
y = data["label"].values

p = model.predict_proba(vec.transform(data["text"].apply(clean)))[:, 1]
flags = [red_flags(t) for t in data["text"]]
n = np.array([len(f) for f in flags])
final = np.array([hybrid_score(pp, nn, cfg["weight"], cfg["rule_cap"]) for pp, nn in zip(p, n)])
rules = np.array([rule_score(nn, cfg["rule_cap"]) for nn in n])

variants = {
    "Model only (cutoff 0.5)": (p >= 0.5).astype(int),
    "Rules only (>= 1 red flag) [OPTIMISTIC - rules saw this set]": (n >= 1).astype(int),
    f"Hybrid, flagged (>= {cfg['thr_suspicious']})": (final >= cfg["thr_suspicious"]).astype(int),
    f"Hybrid, strict (>= {cfg['thr_phishing']})": (final >= cfg["thr_phishing"]).astype(int),
}

out = {"n": len(y), "caveat": "Rules were written with knowledge of this set; only the model-only row is unbiased.", "variants": {}, "auc": {"model": roc_auc_score(y, p), "rules": roc_auc_score(y, rules),
                                             "hybrid": roc_auc_score(y, final)}}
rng = np.random.RandomState(0)
for name, pred in variants.items():
    print(f"\n===== {name} =====")
    print(classification_report(y, pred, target_names=["Normal", "Scam"], zero_division=0))
    print(confusion_matrix(y, pred))
    boots = [(pred[i] == y[i]).mean() for i in (rng.randint(0, len(y), len(y)) for _ in range(5000))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    out["variants"][name] = {"accuracy": float((pred == y).mean()), "ci95": [float(lo), float(hi)],
                             "scam_recall": float(tp / (tp + fn)), "false_alarm_rate": float(fp / (fp + tn)),
                             "precision": float(tp / (tp + fp)) if tp + fp else 0.0}
    print(f"accuracy {(pred == y).mean():.2f}  (95% bootstrap CI {lo:.2f}-{hi:.2f})")

print("\nAUC  model={model:.2f}  rules={rules:.2f}  hybrid={hybrid:.2f}".format(**out["auc"]))

flagged = variants[f"Hybrid, flagged (>= {cfg['thr_suspicious']})"]
print("\n===== Mistakes (hybrid, flagged tier) =====")
mistakes = []
for i in np.where(flagged != y)[0]:
    kind = "MISSED SCAM" if y[i] == 1 else "FALSE ALARM"
    mistakes.append({"kind": kind, "model": float(p[i]), "hybrid": float(final[i]), "flags": flags[i], "text": data["text"][i]})
    print(f"\n[{kind}] model={p[i]:.0%} hybrid={final[i]:.0%} flags={flags[i]}\n    {data['text'][i]}")
out["mistakes"] = mistakes
print(f"\nTotal messages: {len(y)}  |  Mistakes: {len(mistakes)}")

(ROOT / "reports/external_test.json").write_text(json.dumps(out, indent=2))
