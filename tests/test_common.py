"""Run with:  pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import clean, hybrid_score, red_flags, url_issues, verdict  # noqa: E402


def test_clean_masks_urls_and_strips_noise():
    out = clean("Click <b>HERE</b>: http://bit.ly/abc123 now!!! 12345")
    assert "http" not in out and "abc" not in out
    assert "urltoken" in out and "urlshort" in out
    assert not any(ch.isdigit() for ch in out)


def test_clean_survives_garbage():
    assert clean(None) == "none"
    assert clean("http://[broken") is not None   # malformed URL must not raise
    assert len(clean("x" * 10_000_000)) <= 5000  # very long input is truncated


def test_url_issues():
    assert url_issues("http://bit.ly/x")
    assert url_issues("http://192.168.4.12/login")
    assert any("sbi" in i for i in url_issues("http://sbi-kyc-update.in/login"))
    assert url_issues("https://www.amazon.in/orders") == []
    assert url_issues("https://www.sbi.co.in/web/personal-banking") == []


def test_red_flags_detect_scams():
    scam = "Your account is suspended. Verify your password immediately: http://bit.ly/x"
    flags = red_flags(scam)
    assert {"Account threat", "Credential / OTP request", "Urgency / pressure", "Shortened URL"} <= set(flags)


def test_legit_otp_is_not_a_credential_request():
    assert "Credential / OTP request" not in red_flags("Your OTP is 482916. Do not share your OTP with anyone.")
    assert "Credential / OTP request" in red_flags("Please share your OTP to claim the prize")


def test_plain_message_has_no_flags():
    assert red_flags("Hi team, the meeting moved to 3 PM tomorrow. Thanks, Rahul") == []


def test_verdict_tiers():
    cfg = {"thr_suspicious": 0.3, "thr_phishing": 0.6}
    assert verdict(0.1, cfg) == "SAFE"
    assert verdict(0.3, cfg) == "SUSPICIOUS"
    assert verdict(0.9, cfg) == "PHISHING"
    assert hybrid_score(1.0, 3, 0.8, 3) == 1.0
    assert hybrid_score(0.0, 0, 0.8, 3) == 0.0
