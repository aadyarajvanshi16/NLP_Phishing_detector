"""Streamlit demo:  streamlit run app.py"""
import html
import json

import pandas as pd
import streamlit as st

from common import MAX_CHARS, ROOT, clean, explain, hybrid_score, load_artifacts, red_flag_matches, rule_score, verdict

st.set_page_config(page_title="Phishing & Scam Detector", page_icon="🎣", layout="wide")


@st.cache_resource(show_spinner="Loading model…")
def get_artifacts():
    return load_artifacts()


try:
    vec, model, cfg = get_artifacts()
except Exception as e:  # missing or incompatible pickles
    st.error("Could not load the trained model. Run `python train.py` once, then restart the app.")
    st.caption(f"Details: {type(e).__name__}: {e}")
    st.stop()

import sklearn  # noqa: E402

if cfg.get("sklearn_version") != sklearn.__version__:
    st.warning(f"Model was trained with scikit-learn {cfg.get('sklearn_version')} but {sklearn.__version__} is "
               "installed. Re-run `python train.py` if results look odd.")

SAMPLES = {
    "— choose a sample —": "",
    "Scam: bank KYC (look-alike link)": "Dear customer, your SBI account will be blocked today. Update your KYC immediately: http://sbi-kyc-update.in/login",
    "Scam: prize + short link": "Congratulations! You are the winner of a $1000 gift card. Claim your reward now: http://bit.ly/claim-now",
    "Scam: 'digital arrest' call": "URGENT: your Aadhaar is linked to illegal activity. Cyber police will arrest you. Call this number immediately to avoid case.",
    "Scam: bank, raw IP link": "Dear customer, your account has been suspended. Verify your password immediately at http://192.168.4.12/secure-login or it will be permanently blocked within 24 hours.",
    "Legit: office message": "Hi team, the project meeting is moved to 3 PM tomorrow. Please review the attached agenda before joining. Thanks, Rahul",
    "Legit: OTP message (known weak spot)": "Your OTP for login is 482916. Do not share this code with anyone. Valid for 10 minutes.",
    "Legit: delivery update (known weak spot)": "Your Swiggy order is out for delivery. Arriving in 15 minutes.",
}

COLORS = {"SAFE": "#1b8a3b", "SUSPICIOUS": "#b7791f", "PHISHING": "#c53030"}
ICONS = {"SAFE": "✅", "SUSPICIOUS": "⚠️", "PHISHING": "🚨"}


def highlight(text: str, matches) -> str:
    spans = sorted((s, e) for items in matches.values() for s, e, _ in items)
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out, pos = [], 0
    for s, e in merged:
        out.append(html.escape(text[pos:s]))
        out.append(f"<mark style='background:#fed7d7;padding:0 2px;border-radius:3px'>{html.escape(text[s:e])}</mark>")
        pos = e
    out.append(html.escape(text[pos:]))
    return "<div style='line-height:1.7;white-space:pre-wrap'>" + "".join(out) + "</div>"


def load_json(name):
    p = ROOT / "reports" / name
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **NLP model** — TF-IDF word 1–2-grams + Logistic Regression gives a *model score*.\n"
        "2. **Rules** — red-flag patterns (urgency, OTP requests, KYC, fake links…) give a *rule score*.\n"
        f"3. **Risk score** = {cfg['weight']:.0%} model + {1 - cfg['weight']:.0%} rules.\n\n"
        f"Verdict: **SAFE** < {cfg['thr_suspicious']:.2f} ≤ **SUSPICIOUS** < {cfg['thr_phishing']:.2f} ≤ **PHISHING**"
    )
    st.caption("Weights and thresholds were tuned on a validation split, not on the demo samples.")
    st.divider()
    st.info("Prototype for a course project — it can be wrong. Never decide whether a message is safe "
            "based on this tool alone; contact the sender through an official channel.")

st.title("🎣 Phishing & Social-Engineering Detector")
st.caption("Hybrid NLP + rule-based detector for emails and SMS, with explanations.")

tab_analyze, tab_perf, tab_about = st.tabs(["🔍 Analyze a message", "📊 Model performance", "ℹ️ About & limits"])

# ---------------------------------------------------------------- analyze
with tab_analyze:
    choice = st.selectbox("Try a sample", list(SAMPLES.keys()))
    text = st.text_area("Paste an email or SMS", value=SAMPLES[choice], height=180,
                        placeholder="Paste the message text here…")
    if len(text) > MAX_CHARS:
        st.caption(f"Only the first {MAX_CHARS:,} characters are analysed.")

    if st.button("Analyze", type="primary") and text.strip():
        x = vec.transform([clean(text)])
        p = float(model.predict_proba(x)[0][1])
        matches = red_flag_matches(text)
        r = rule_score(len(matches), cfg["rule_cap"])
        final = hybrid_score(p, len(matches), cfg["weight"], cfg["rule_cap"])
        v = verdict(final, cfg)

        st.markdown(
            f"<div style='padding:14px 18px;border-radius:10px;background:{COLORS[v]}1a;border-left:6px solid {COLORS[v]}'>"
            f"<span style='font-size:1.6rem;font-weight:700;color:{COLORS[v]}'>{ICONS[v]} {v}</span>"
            f"<span style='margin-left:12px;opacity:.8'>risk score {final:.0%}</span></div>", unsafe_allow_html=True)
        st.progress(min(float(final), 1.0))

        c1, c2, c3 = st.columns(3)
        c1.metric("Risk score", f"{final:.0%}")
        c2.metric("Model score (NLP)", f"{p:.0%}")
        c3.metric("Rule score", f"{r:.0%}", help=f"{len(matches)} red-flag categories found")

        st.subheader("Evidence in the message")
        if matches:
            st.markdown(highlight(text[:MAX_CHARS], matches), unsafe_allow_html=True)
        else:
            st.caption("No rule-based red flags found in the text.")

        left, mid, right = st.columns(3)
        with left:
            st.markdown("**🚩 Red flags**")
            if matches:
                for name, items in matches.items():
                    note = items[0][2]
                    st.write(f"• **{name}**" + (f" — _{note}_" if len(note) > 0 and name in
                                               ("Suspicious domain", "Shortened URL", "IP-based URL") else ""))
            else:
                st.write("None")
        toward_phish, toward_safe = explain(vec, model, x)
        with mid:
            st.markdown("**Words pushing towards phishing**")
            for w, val in toward_phish or []:
                st.write(f"• **{w}** (+{val:.2f})")
            if not toward_phish:
                st.write("None")
        with right:
            st.markdown("**Words pushing towards safe**")
            for w, val in toward_safe or []:
                st.write(f"• **{w}** ({val:.2f})")
            if not toward_safe:
                st.write("None")

        if p < 0.3 and len(matches) == 0 and x.nnz < 3:
            st.caption("⚠️ Few known words in this message — the model has little to go on, so treat the score with caution.")

# ---------------------------------------------------------------- performance
with tab_perf:
    m, ext, abl = load_json("metrics.json"), load_json("external_test.json"), load_json("ablation.json")
    if not m:
        st.info("Run `python train.py` to generate the performance report.")
    else:
        st.subheader("1 · Held-out test (same kind of data as training)")
        rows = []
        for name, d in m["test"].items():
            rows.append({"Setting": name, "Accuracy": d["accuracy"], "Precision": d["precision"],
                         "Recall": d["recall"], "F1": d["f1"]})
        st.dataframe(pd.DataFrame(rows).style.format({c: "{:.1%}" for c in ["Accuracy", "Precision", "Recall", "F1"]}),
                     hide_index=True)
        fig = ROOT / "reports/figures/test_performance.png"
        if fig.exists():
            st.image(str(fig))
        st.caption("High numbers here only mean the model fits its own datasets well.")

        st.subheader("2 · Transfer between datasets (the honest test)")
        cd = m["cross_dataset"]
        rows = [{"Train → Test": k.replace("train_", "").replace("_test_", " → ").replace("email", "emails").replace("sms", "SMS"),
                 "Accuracy": d["accuracy"], "Spam/phish precision": d["precision"], "Spam/phish recall": d["recall"]}
                for k, d in cd.items()]
        st.dataframe(pd.DataFrame(rows).style.format({c: "{:.1%}" for c in rows[0] if c != "Train → Test"}),
                     hide_index=True)

        if ext:
            st.subheader("3 · Hand-written India-style messages (n = 50)")
            rows = [{"Setting": k, "Accuracy": d["accuracy"], "Scam recall": d["scam_recall"],
                     "False-alarm rate": d["false_alarm_rate"]} for k, d in ext["variants"].items()]
            st.dataframe(pd.DataFrame(rows).style.format({c: "{:.0%}" for c in ["Accuracy", "Scam recall", "False-alarm rate"]}),
                         hide_index=True)
            st.warning(ext["caveat"] + " Small sample: treat accuracy as ± 10–15 points.")
        if abl:
            st.subheader("4 · Preprocessing ablation (AUC)")
            df = pd.DataFrame(abl["rows"]).rename(columns={
                "variant": "Variant", "in_distribution_auc": "In-distribution", "email_to_sms_auc": "Email → SMS",
                "sms_to_email_auc": "SMS → Email", "cross_mean_auc": "Cross mean", "india_auc_info_only": "India (info only)"})
            st.dataframe(df.style.format({c: "{:.3f}" for c in df.columns if c != "Variant"}),
                         hide_index=True)

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown(
        """
**Data.** ~17k emails (Phishing_Email.csv) and ~5k SMS (UCI SMS Spam Collection). After removing duplicates,
about 22k messages remain, split 60/20/20 into train / validation / test.

**Pipeline.** `clean()` lowercases, masks URLs (and describes them: shortener, IP, look-alike brand, risky TLD),
strips HTML and non-letters → TF-IDF (1–2-grams) → class-balanced Logistic Regression → blended with the rule score.

**Known limitations**
- The training data is mostly *spam* (pharma, watches, lottery) and corporate/academic email, not modern phishing.
  The model has never seen words like *OTP*, *KYC*, *UPI* or *Aadhaar*.
- Everyday transactional messages (food delivery, order updates, bank debit alerts) are often over-flagged.
- Transfer between email and SMS data is weak (see *Model performance*).
- Only text is analysed — no sender, headers, attachments or live URL reputation.
- Rules are keyword-based and English-only; Hinglish and other languages are largely unsupported.

**Next steps:** train on in-domain Indian SMS/email data, add a multilingual transformer baseline, and evaluate
on a fresh test set written without looking at the rules.
        """
    )
