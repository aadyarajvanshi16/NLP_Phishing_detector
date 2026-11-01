"""Compare preprocessing variants (an ablation study).

    python ablation.py

For every variant we train TF-IDF + Logistic Regression and measure
  * in-distribution AUC on a held-out split of the merged data
  * cross-dataset AUC (train on emails -> test on SMS, and the reverse)
  * AUC on the hand-written India test set (reported for information only)

The variant used by the shipped model is chosen by *cross-dataset* AUC, not by the
India test set, so the India set stays a fair external check.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from common import ROOT, clean
from train import load_data, new_model, new_vec

VARIANTS = {
    "A. letters only, URL -> 'urltoken'": dict(url_features=False, number_features=False),
    "B. + URL descriptors (shortener / IP / brand / TLD)": dict(url_features=True, number_features=False),
    "C. + number, money, phone tokens": dict(url_features=False, number_features=True),
    "D. + both": dict(url_features=True, number_features=True),
}


def fit_auc(tr, te, col):
    v = new_vec()
    m = new_model().fit(v.fit_transform(tr[col]), tr["y"])
    return roc_auc_score(te["y"], m.predict_proba(v.transform(te[col]))[:, 1]), v, m


def main():
    df = load_data()
    ext = pd.read_csv(ROOT / "data/india_test.csv")
    rows = []
    for name, kw in VARIANTS.items():
        df["c"] = df["text"].apply(lambda t: clean(t, **kw))
        ext["c"] = ext["text"].apply(lambda t: clean(t, **kw))
        A, B = df[df.src == "email"], df[df.src == "sms"]
        e2s, _, _ = fit_auc(A, B, "c")
        s2e, _, _ = fit_auc(B, A, "c")
        tr, te = train_test_split(df, test_size=0.2, stratify=df["src"] + df["y"].astype(str), random_state=42)
        ind, v, m = fit_auc(tr, te, "c")
        india = roc_auc_score(ext["label"], m.predict_proba(v.transform(ext["c"]))[:, 1])
        rows.append({"variant": name, "in_distribution_auc": ind, "email_to_sms_auc": e2s,
                     "sms_to_email_auc": s2e, "cross_mean_auc": (e2s + s2e) / 2, "india_auc_info_only": india})
        print(f"{name:55s} in-dist={ind:.3f}  e->s={e2s:.3f}  s->e={s2e:.3f}  india(info)={india:.3f}")

    table = pd.DataFrame(rows)
    best = table.loc[table["cross_mean_auc"].idxmax(), "variant"]
    print("\nBest by cross-dataset AUC:", best)
    (ROOT / "reports/ablation.json").write_text(json.dumps({"rows": rows, "selected_by_cross_dataset_auc": best}, indent=2))


if __name__ == "__main__":
    main()
