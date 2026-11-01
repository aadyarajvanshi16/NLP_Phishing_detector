"""Train the phishing / spam detector and write models + reports.

    python train.py

Protocol (so the numbers in the report are honest):
  * split the merged data 60 / 20 / 20 into train / validation / test
  * fit the TF-IDF + Logistic Regression model on *train* only
  * choose the hybrid weight and the SAFE / SUSPICIOUS / PHISHING thresholds on *validation* only
  * report on *test* once, and separately report cross-dataset transfer
The hand-written India test set (data/india_test.csv) is never used here;
see evaluate.py.
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

from common import MAX_CHARS, MODELS_DIR, ROOT, clean, hybrid_score, red_flags

SEED = 42
RULE_CAP = 3
WEIGHT_GRID = [0.5, 0.6, 0.7, 0.8]  # rules always keep at least 20 % of the score
REPORTS = ROOT / "reports"
(REPORTS / "figures").mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------ data
def load_data() -> pd.DataFrame:
    a = pd.read_csv(ROOT / "data/Phishing_Email.csv").dropna(subset=["Email Text", "Email Type"])
    a = a[a["Email Text"].str.strip().str.lower() != "empty"]
    A = pd.DataFrame({"text": a["Email Text"].astype(str).str[:MAX_CHARS],
                      "y": (a["Email Type"] == "Phishing Email").astype(int), "src": "email"})

    b = pd.read_csv(ROOT / "data/spam.csv", encoding="latin-1")[["v1", "v2"]].dropna()
    B = pd.DataFrame({"text": b["v2"], "y": (b["v1"] == "spam").astype(int), "src": "sms"})

    df = pd.concat([A, B], ignore_index=True)
    n0 = len(df)
    df["clean"] = df["text"].apply(clean)
    # drop exact duplicates AND near-duplicates that only differ in punctuation/digits/URLs
    df = df[df["clean"].str.len() > 0].drop_duplicates(subset="clean").reset_index(drop=True)
    print(f"Rows: {n0} -> {len(df)} after cleaning + de-duplication "
          f"(emails={int((df.src == 'email').sum())}, sms={int((df.src == 'sms').sum())})")
    return df


def new_vec():
    return TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=3,
                           sublinear_tf=True, stop_words="english")


def new_model():
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def metrics(y, pred) -> dict:
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "confusion": confusion_matrix(y, pred).tolist()}


# ------------------------------------------------------------ cross-dataset
def cross_dataset(df: pd.DataFrame) -> dict:
    out = {}
    A, B = df[df.src == "email"], df[df.src == "sms"]
    for name, tr, te in [("train_email_test_sms", A, B), ("train_sms_test_email", B, A)]:
        v = new_vec()
        m = new_model().fit(v.fit_transform(tr["clean"]), tr["y"])
        pred = m.predict(v.transform(te["clean"]))
        out[name] = metrics(te["y"], pred)
        print(f"\n== {name} ==")
        print(classification_report(te["y"], pred, target_names=["Safe", "Phishing/Spam"], zero_division=0))
    return out


# ------------------------------------------------------------------ tuning
def flag_counts(texts) -> np.ndarray:
    return np.array([len(red_flags(t)) for t in texts])


def tune(p_val, n_val, y_val) -> dict:
    """Pick hybrid weight by validation AUC, then thresholds by validation precision/recall targets."""
    best = max(WEIGHT_GRID, key=lambda w: roc_auc_score(y_val, [hybrid_score(p, n, w, RULE_CAP) for p, n in zip(p_val, n_val)]))
    s = np.array([hybrid_score(p, n, best, RULE_CAP) for p, n in zip(p_val, n_val)])
    prec, rec, thr = precision_recall_curve(y_val, s)
    prec, rec = prec[:-1], rec[:-1]
    # PHISHING: lowest threshold whose precision is >= 98 %  (very few false alarms)
    ok = np.where(prec >= 0.98)[0]
    thr_phish = float(thr[ok[0]]) if len(ok) else 0.6
    # SUSPICIOUS: highest threshold that still catches >= 98 % of phishing (high-recall alert tier)
    ok = np.where(rec >= 0.98)[0]
    thr_susp = float(thr[ok[-1]]) if len(ok) else 0.3
    thr_susp = min(thr_susp, thr_phish)
    return {"weight": best, "thr_suspicious": round(thr_susp, 3), "thr_phishing": round(thr_phish, 3)}


# --------------------------------------------------------------------- main
def main():
    df = load_data()
    cross = cross_dataset(df)

    strat = df["src"] + df["y"].astype(str)
    trainval, test = train_test_split(df, test_size=0.2, stratify=strat, random_state=SEED)
    train, val = train_test_split(trainval, test_size=0.25, stratify=(trainval["src"] + trainval["y"].astype(str)),
                                  random_state=SEED)
    print(f"\nSplit sizes  train={len(train)}  val={len(val)}  test={len(test)}")

    vec = new_vec()
    model = new_model().fit(vec.fit_transform(train["clean"]), train["y"])
    p = {k: model.predict_proba(vec.transform(d["clean"]))[:, 1] for k, d in [("val", val), ("test", test)]}
    n = {"val": flag_counts(val["text"]), "test": flag_counts(test["text"])}

    cfg = tune(p["val"], n["val"], val["y"].values)
    cfg.update({"rule_cap": RULE_CAP, "sklearn_version": sklearn.__version__, "seed": SEED,
                "n_train": len(train), "n_val": len(val), "n_test": len(test)})
    print("\nTuned on validation:", {k: cfg[k] for k in ("weight", "thr_suspicious", "thr_phishing")})

    # ----- test-set report (touched once)
    h = np.array([hybrid_score(pp, nn, cfg["weight"], RULE_CAP) for pp, nn in zip(p["test"], n["test"])])
    y = test["y"].values
    variants = {
        "model_only_0.5": (p["test"] >= 0.5).astype(int),
        "hybrid_flagged (>= suspicious)": (h >= cfg["thr_suspicious"]).astype(int),
        "hybrid_strict (>= phishing)": (h >= cfg["thr_phishing"]).astype(int),
    }
    report = {"config": cfg, "cross_dataset": cross, "test": {}, "test_by_source": {},
              "auc_test": {"model": roc_auc_score(y, p["test"]), "hybrid": roc_auc_score(y, h)}}
    for name, pred in variants.items():
        report["test"][name] = metrics(y, pred)
        print(f"\n== TEST: {name} ==")
        print(classification_report(y, pred, target_names=["Safe", "Phishing/Spam"], zero_division=0))
    for s in ("email", "sms"):
        msk = (test["src"] == s).values
        report["test_by_source"][s] = metrics(y[msk], variants["hybrid_flagged (>= suspicious)"][msk])

    names = vec.get_feature_names_out()
    order = model.coef_[0].argsort()
    report["top_phishing_terms"] = [names[i] for i in order[-25:][::-1]]
    report["top_safe_terms"] = [names[i] for i in order[:25]]
    print("\nTop phishing terms:", report["top_phishing_terms"][:12])
    print("Top safe terms    :", report["top_safe_terms"][:12])

    # ----- save
    joblib.dump(vec, MODELS_DIR / "vec.joblib")
    joblib.dump(model, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "config.json").write_text(json.dumps(cfg, indent=2))
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2))
    make_figures(y, p["test"], h, variants["hybrid_flagged (>= suspicious)"])
    print("\nSaved models/, reports/metrics.json and reports/figures/")


def make_figures(y, p_model, p_hybrid, pred):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for label, s in [("Model only", p_model), ("Hybrid (model + rules)", p_hybrid)]:
        pr, rc, _ = precision_recall_curve(y, s)
        ax[0].plot(rc, pr, label=label)
    ax[0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (held-out test)", ylim=(0, 1.02))
    ax[0].legend(loc="lower left")
    cm = confusion_matrix(y, pred)
    ax[1].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[1].text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Safe", "Phish/Spam"], yticklabels=["Safe", "Phish/Spam"],
              xlabel="Predicted", ylabel="Actual", title="Confusion matrix (hybrid, flagged)")
    fig.tight_layout()
    fig.savefig(REPORTS / "figures/test_performance.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
