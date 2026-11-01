# 🎣 Phishing & Social-Engineering Detector

A hybrid NLP + rule-based detector for phishing / scam **emails and SMS**, with a Streamlit demo that
explains *why* a message was flagged (highlighted evidence, red-flag categories, and the words that
pushed the model's decision).

> Semester project for NLP. Prototype only — see [Limitations](#limitations-read-this).

## How it works

```
message ──► clean() ──► TF-IDF (1–2-grams) ──► Logistic Regression ──► model score ─┐
   │                                                                                 ├─► risk score ─► SAFE / SUSPICIOUS / PHISHING
   └──────► red-flag rules (urgency, OTP/KYC requests, fake links, …) ─► rule score ─┘
```

* **`clean()`** lowercases, masks URLs (adding descriptor words for shortened links, raw IPs, look-alike brand
  domains such as `sbi-kyc-update.in`, risky TLDs), strips HTML and non-letters.
* **Rules** (`common.py`) are regular-expression categories, e.g. *Urgency*, *Account threat*, *OTP / credential
  request*, *KYC demand*, *Authority threat*, *Advance fee*, *Suspicious domain*. "Do **not** share your OTP" is
  correctly ignored.
* **Risk score** = `0.8 × model + 0.2 × rules`. The weight and the two thresholds are chosen on a
  **validation split** (targets: ≥98 % recall for *SUSPICIOUS*, ≥98 % precision for *PHISHING*).

## Quick start

```bash
pip install -r requirements.txt
python train.py        # trains, tunes on validation, writes models/ and reports/
python evaluate.py     # external check on data/india_test.csv
python ablation.py     # preprocessing ablation (optional, ~1 min)
pytest -q              # unit tests
streamlit run app.py   # demo
```

Re-run `python train.py` after installing — the model files are tied to the scikit-learn version.

## Project layout

| Path | Purpose |
|---|---|
| `common.py` | The single source of truth: `clean()`, URL analysis, red-flag rules, scoring, explanations |
| `train.py` | Dedupe → 60/20/20 split → fit → tune on validation → report on test → save |
| `evaluate.py` | Evaluation on the hand-written India-style test set |
| `ablation.py` | Compares four preprocessing variants |
| `app.py` | Streamlit UI (analyze / performance / about) |
| `data/` | `Phishing_Email.csv` (emails), `spam.csv` (UCI SMS Spam Collection), `india_test.csv` (50 hand-written messages) |
| `reports/` | `metrics.json`, `external_test.json`, `ablation.json`, `figures/` (generated) |
| `tests/` | Unit tests for cleaning, URL analysis and rules |

## Results

Numbers below come from `reports/*.json` (seed 42). **Read the three tables together** — the first one alone is misleading.

### 1. Held-out test, same kind of data as training (n = 4,435)

| Setting | Accuracy | Precision | Recall |
|---|---|---|---|
| Model only (cutoff 0.5) | 96.3 % | 94.3 % | 94.0 % |
| Hybrid, *SUSPICIOUS or worse* (≥ 0.305) | 93.9 % | 85.8 % | 96.7 % |
| Hybrid, *PHISHING* (≥ 0.569) | 94.0 % | 98.0 % | 82.6 % |

In-distribution the rule layer adds nothing (the precision-recall curves overlap — `reports/figures/`), and on the
SMS slice the high-recall tier has only ~51 % precision.

### 2. Transfer between datasets (the honest test)

| Train → Test | Accuracy | Spam/phish precision | Spam/phish recall |
|---|---|---|---|
| Emails → SMS | 43.0 % | 15.7 % | 90.1 % |
| SMS → Emails | 60.8 % | 47.5 % | 39.0 % |

The model does **not** transfer between corpora. Its high in-distribution score reflects dataset-specific
vocabulary, not a general notion of phishing.

### 3. Hand-written India-style messages (n = 50; 25 scam / 25 normal)

| Setting | Accuracy (95 % CI) | Scam recall | False-alarm rate |
|---|---|---|---|
| **Model only** (cutoff 0.5) — *unbiased* | 64 % (50–76 %) | 40 % | 12 % |
| Hybrid, ≥ 0.305 — *optimistic* | 84 % (74–94 %) | 100 % | 32 % |
| Hybrid, ≥ 0.569 — *optimistic* | 72 % (60–84 %) | 48 % | 4 % |

⚠️ The rule patterns were written/extended with knowledge of these 50 messages, so any row that includes the
rules (and the "rules only = 100 %" figure in `evaluate.py`) is **not** an unbiased estimate. Only the model-only
row is clean, and it shows the model struggles on modern Indian scams. A fair final number needs ~50 *new*
messages written without looking at the rules.

### 4. Preprocessing ablation (AUC)

| Variant | In-distribution | Email→SMS | SMS→Email |
|---|---|---|---|
| A. letters only, URL → `urltoken` | 0.991 | 0.768 | 0.594 |
| B. + URL descriptors *(shipped)* | 0.991 | 0.767 | 0.594 |
| C. + number / money / phone tokens | 0.992 | 0.732 | 0.453 |
| D. + both | 0.992 | 0.732 | 0.453 |

Number/phone/money placeholder tokens **hurt** transfer, so they are disabled. A and B tie on the cross-dataset
criterion; B is kept because URL properties are meaningful for phishing, but this is a judgement call, not a
measured win.

## Limitations (read this)

* **The "phishing" class is largely spam.** Strongest model words: *click, free, remove, money…*; strongest "safe"
  words: *enron, wrote, language, university, linguistics*. The model partly learned *which corpus* a message came from.
* **No vocabulary for modern Indian scams** (*OTP, KYC, UPI, Aadhaar* never appear in training).
* **Everyday transactional messages are over-flagged** (food/ride delivery, bank debit alerts, OTP messages).
* Text only: no sender, headers, attachments or live URL reputation. English-centric rules.
* Small external test set; ~2 % of test messages still have a near-identical (cosine ≥ 0.99) twin in training.
* Prototype — do not use as a real security control.

## Next steps

1. Add **in-domain training data** (Indian SMS/email scams *and* legitimate transactional messages) — the single
   biggest improvement available.
2. Write a **fresh external test set** without looking at the rules, and evaluate once.
3. Add model baselines for the report: character n-grams, sentence embeddings, a fine-tuned multilingual transformer.
4. Add Hinglish support and sender / header features.

## Data

* `data/Phishing_Email.csv` — emails labelled *Phishing Email* / *Safe Email* (add your source link here).
* `data/spam.csv` — [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
* `data/india_test.csv` — 50 hand-written messages (placeholder numbers and links are fake).
