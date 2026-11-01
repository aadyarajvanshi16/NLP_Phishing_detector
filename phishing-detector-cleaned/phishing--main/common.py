"""Shared logic for training, evaluation and the Streamlit app.

Everything that must stay identical between ``train.py``, ``evaluate.py`` and
``app.py`` lives here: text cleaning, URL analysis, rule-based red flags and
the hybrid risk score.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import joblib

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
MAX_CHARS = 5000  # some training emails are millions of characters long

# --------------------------------------------------------------------------
# URL analysis
# --------------------------------------------------------------------------
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"')]+", re.I)

SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "cutt.ly", "rb.gy", "is.gd", "shorturl.at"}
RISKY_TLDS = {"xyz", "top", "club", "site", "icu", "buzz", "click", "link", "vip", "cfd", "sbs", "live"}
IP_HOST_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

# brand -> registrable domains that really belong to it
OFFICIAL_DOMAINS = {
    "sbi": {"sbi.co.in", "onlinesbi.sbi", "onlinesbi.com"},
    "hdfc": {"hdfcbank.com", "hdfc.com"},
    "hdfcbank": {"hdfcbank.com"},
    "icici": {"icicibank.com"},
    "icicibank": {"icicibank.com"},
    "pnb": {"pnbindia.in"},
    "paytm": {"paytm.com"},
    "amazon": {"amazon.in", "amazon.com"},
    "netflix": {"netflix.com"},
    "jio": {"jio.com"},
    "paypal": {"paypal.com"},
    "google": {"google.com"},
    "microsoft": {"microsoft.com"},
    "apple": {"apple.com"},
    "irctc": {"irctc.co.in"},
    "indiapost": {"indiapost.gov.in"},
    "incometax": {"incometax.gov.in"},
}

_SECOND_LEVEL = {"co", "com", "org", "net", "gov", "ac", "nic"}


def _host(url: str) -> str:
    url = url if re.match(r"https?://", url, re.I) else "http://" + url
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:  # malformed URLs (stray brackets, bad ports) are common in real spam
        return ""


def _registrable(host: str) -> str:
    labels = host.split(".")
    if len(labels) >= 3 and labels[-2] in _SECOND_LEVEL and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def url_issues(url: str) -> list[str]:
    """Return short, human-readable reasons why a URL looks suspicious."""
    host = _host(url)
    if not host:
        return []
    issues = []
    if host in SHORTENERS:
        issues.append("shortened link hides the real destination")
    if IP_HOST_RE.match(host):
        issues.append("uses a raw IP address instead of a domain")
    reg = _registrable(host)
    tokens = re.split(r"[.\-]", host)
    for brand, official in OFFICIAL_DOMAINS.items():
        if any(t == brand or (len(brand) >= 4 and t.startswith(brand)) for t in tokens):
            if reg not in official:
                issues.append(f"mentions '{brand}' but the domain is {reg}")
            break
    if reg.rsplit(".", 1)[-1] in RISKY_TLDS:
        issues.append(f"risky top-level domain (.{reg.rsplit('.', 1)[-1]})")
    if host.count("-") >= 2:
        issues.append("domain stuffed with hyphens")
    return issues


def _url_tokens(url: str, detail: bool = True) -> str:
    """Words that stand in for a URL inside the cleaned text (model input)."""
    toks = ["urltoken"]
    if not detail:
        return "urltoken"
    host = _host(url)
    if host in SHORTENERS:
        toks.append("urlshort")
    if IP_HOST_RE.match(host):
        toks.append("urlip")
    issues = " ".join(url_issues(url))
    if "mentions '" in issues:
        toks.append("urlbrand")
    if "risky top-level" in issues:
        toks.append("urlrisktld")
    if "hyphens" in issues:
        toks.append("urlhyphen")
    return " ".join(toks)


# --------------------------------------------------------------------------
# Text cleaning (model input)
# --------------------------------------------------------------------------
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s-]{8,}\d)|(?:\d{2}x{4,}\d{1,2})")
_MONEY_RE = re.compile(r"(?:rs\.?|inr|₹|\$|£|€)\s*\d[\d,\.]*")
_NUM_RE = re.compile(r"\d+")


def _num_token(m: re.Match) -> str:
    n = len(m.group(0))
    return " numshort " if n <= 3 else " nummid " if n <= 6 else " numlong "


# Defaults used by the shipped model. ablation.py compares the alternatives.
URL_FEATURES = True
NUMBER_FEATURES = False


def clean(text, url_features: bool | None = None, number_features: bool | None = None) -> str:
    """Normalise a message for the TF-IDF model.

    * URLs become ``urltoken`` (+ optional words describing the URL: shortener, IP,
      look-alike brand, risky TLD, hyphen-stuffed domain) -- ``url_features``
    * optionally, phone numbers / money amounts / number sizes become placeholder
      words instead of being deleted -- ``number_features``
    Everything else that isn't a letter is removed.
    """
    url_features = URL_FEATURES if url_features is None else url_features
    number_features = NUMBER_FEATURES if number_features is None else number_features
    t = str(text)[:MAX_CHARS]
    t = URL_RE.sub(lambda m: f" {_url_tokens(m.group(0), url_features)} ", t)
    t = t.lower()
    t = re.sub(r"<[^>]+>", " ", t)
    if number_features:
        t = _MONEY_RE.sub(" moneytoken ", t)
        t = _PHONE_RE.sub(" phonetoken ", t)
        t = _NUM_RE.sub(_num_token, t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------------------------
# Rule-based red flags
# --------------------------------------------------------------------------
RED_FLAGS = {
    "Urgency / pressure": r"\b(urgent(ly)?|immediately|right now|act now|final notice|last chance|hurry|"
                          r"within \d+ ?(hours?|hrs?|minutes?|mins?)|expires?( today| soon)?|"
                          r"avoid (disconnection|penalty|arrest))\b",
    "Account threat": r"\b(suspended|locked|deactivated|unauthori[sz]ed|compromised|blocked|disconnected|"
                      r"cancelled|on hold|will be (closed|terminated|disabled))\b",
    "Credential / OTP request": r"\b(verify|confirm|update|enter|share|send|submit|provide|reply with)\b"
                                r"[^.!?\n]{0,40}\b(password|passcode|otp|pin|cvv|card number|card details|"
                                r"bank details|account number|aadhaar|aadhar|pan)\b",
    "Money / prize lure": r"\b(winner|you (have )?won|prize|lottery|refund|claim|gift card|rewards?|"
                          r"cashback|lucky draw|congratulations|free (iphone|recharge|gift))\b",
    "KYC / ID verification demand": r"\b(kyc|e-?kyc|aadhaar|aadhar|pan (card|number)|sim (will be )?blocked)\b",
    "Authority threat": r"\b(police|arrest(ed)?|cyber ?(crime|cell|police)|court|legal action|customs|"
                        r"narcotics|illegal activity)\b",
    "Fee / advance payment": r"\b(registration fee|processing fee|redelivery fee|clearance (charge|fee)|"
                             r"delivery charges?|security deposit|penalty)\b",
    "Job / investment bait": r"\b(work from home|part[- ]time job|no experience needed|guaranteed|"
                             r"double your money|vip (stock )?group|"
                             r"earn (rs\.?|₹|inr)?\s?[\d,]+\s*(daily|per day|a day))\b|"
                             r"\b(join|message|contact)\b[^.!?\n]{0,25}\b(telegram|whatsapp)\b",
}
_COMPILED = {name: re.compile(pat, re.I) for name, pat in RED_FLAGS.items()}
_NEGATION_BEFORE = re.compile(r"\b(do not|don't|dont|never|not)\s*$", re.I)


def red_flag_matches(text: str) -> dict[str, list[tuple[int, int, str]]]:
    """Map flag name -> list of (start, end, note) so the UI can highlight evidence."""
    text = str(text)[:MAX_CHARS]
    found: dict[str, list[tuple[int, int, str]]] = {}
    for name, rx in _COMPILED.items():
        for m in rx.finditer(text):
            # "Do not share your OTP with anyone" is advice, not a request
            if name == "Credential / OTP request" and _NEGATION_BEFORE.search(text[max(0, m.start() - 15):m.start()]):
                continue
            found.setdefault(name, []).append((m.start(), m.end(), m.group(0)))

    urls = list(URL_RE.finditer(text))
    for m in urls:
        issues = url_issues(m.group(0))
        if not issues:
            continue
        if "shortened" in issues[0]:
            key = "Shortened URL"
        elif "raw IP" in issues[0]:
            key = "IP-based URL"
        else:
            key = "Suspicious domain"
        found.setdefault(key, []).append((m.start(), m.end(), "; ".join(issues)))
    if len(urls) >= 3:
        found["Many links"] = [(m.start(), m.end(), m.group(0)) for m in urls]
    return found


def red_flags(text: str) -> list[str]:
    return list(red_flag_matches(text).keys())


# --------------------------------------------------------------------------
# Hybrid scoring
# --------------------------------------------------------------------------
def rule_score(n_flags: int, cap: int) -> float:
    return min(n_flags / cap, 1.0)


def hybrid_score(p_model: float, n_flags: int, weight: float, cap: int) -> float:
    return weight * p_model + (1 - weight) * rule_score(n_flags, cap)


def verdict(score: float, cfg: dict) -> str:
    if score >= cfg["thr_phishing"]:
        return "PHISHING"
    if score >= cfg["thr_suspicious"]:
        return "SUSPICIOUS"
    return "SAFE"


# --------------------------------------------------------------------------
# Artifacts and explanations
# --------------------------------------------------------------------------
def load_artifacts():
    """Load vectoriser, model and config. Raises FileNotFoundError / other errors if stale."""
    cfg = json.loads((MODELS_DIR / "config.json").read_text())
    vec = joblib.load(MODELS_DIR / "vec.joblib")
    model = joblib.load(MODELS_DIR / "model.joblib")
    return vec, model, cfg


def explain(vec, model, x, k: int = 8):
    """Top words pushing towards phishing and towards safe for one vectorised message."""
    names = vec.get_feature_names_out()
    c = x.multiply(model.coef_[0]).tocoo()
    pairs = sorted(zip(c.col, c.data), key=lambda t: t[1])
    toward_safe = [(names[i], float(v)) for i, v in pairs[:k] if v < 0]
    toward_phish = [(names[i], float(v)) for i, v in pairs[::-1][:k] if v > 0]
    return toward_phish, toward_safe
