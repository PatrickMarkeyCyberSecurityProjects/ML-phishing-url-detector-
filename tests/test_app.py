"""Tests for the deployed app (app/app.py) and its packaging.

The central test: the app gives exactly the scores the notebooks gave.
"""

import ast
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import evaluate as E
from src import features as F
from src import models as M

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
REFERENCE = ROOT / "reports" / "app_reference_scores.json"
REFERENCE_BANDS = ROOT / "reports" / "app_reference_bands.json"
needs_models = pytest.mark.skipif(
    not (ROOT / "models" / "review_operating.json").exists() or not REFERENCE_BANDS.exists(),
    reason="run notebooks 02 to 05 first")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app():
    return _load(APP_DIR / "app.py", "phishing_app")


INVENTED = [
    "https://www.example.com", "example.org/about/team", "http://secure-login.account-verify.example.net/signin/",
    "http://paypal.com.account-check.example.net/verify/login.php?session=8f2a", "http://192.0.2.44/wp-admin/update/",
    "HTTPS://WWW.Example.ORG/News/2025/Article?id=7#comments", "https://xn--pypal-4ve.example/",
    "https://user@login.example.com:8443/auth", "https://example.com/" + "a" * 250, "ftp://files.example.com/pub/readme.txt",
    "  https://docs.example.net/guides/getting-started/installation.html  ", "http://bit.example/3xYz9",
]


# ------------------------------------------------------------------ app scores equal notebook scores

@needs_models
def test_app_scores_equal_the_scores_saved_by_the_evaluation_notebook(app):
    reference = M.load_json(REFERENCE)["urls"]
    bands = {row["url"]: row for row in M.load_json(REFERENCE_BANDS)["urls"]}
    assert len(reference) == 20
    for entry in reference:
        got = app.score_url(entry["url"])
        assert got["ok"] and got["scored_url"] == entry["main_track_url"]
        for name in ("lgbm", "cnn"):
            assert got["models"][name]["raw_score"] == pytest.approx(entry[name]["raw_score"], abs=1e-6), entry["url"]
            assert got["models"][name]["score"] == pytest.approx(entry[name]["calibrated_score"], abs=1e-6)
            # Block is the frozen threshold, so it must agree with the main evaluation's verdict.
            assert (got["models"][name]["band"] == "Block") == (entry[name]["verdict"] == "Block")
            assert got["models"][name]["band"] == bands[entry["url"]][name]["band"]


@needs_models
def test_app_scores_equal_the_notebook_scoring_path_on_more_invented_urls(app):
    scorers = E.load_scorers()
    frame = pd.DataFrame({"url": [F.normalize_url(u, strip_query=F.MAIN_STRIP_QUERY) for u in INVENTED],
                          "url_full": [F.normalize_url(u) for u in INVENTED]})
    for name in ("lgbm", "cnn"):
        notebook = scorers[name]["raw"](frame)
        calibrated = M.apply_calibration(scorers[name]["operating"]["calibration"], notebook)
        for url, raw, score in zip(INVENTED, notebook, calibrated):
            got = app.score_url(url)["models"][name]
            assert got["raw_score"] == pytest.approx(raw, abs=1e-6), (name, url)
            assert got["score"] == pytest.approx(score, abs=1e-6), (name, url)


@needs_models
def test_app_thresholds_are_the_frozen_block_and_the_post_hoc_review(app):
    scorer = app.get_scorer()
    review = M.load_json(ROOT / "models" / "review_operating.json")
    for name in ("lgbm", "cnn"):
        frozen = M.load_json(ROOT / "models" / f"{name}_operating.json")["thresholds"]
        assert scorer.block[name] == frozen["block"]["raw_score"]
        assert scorer.review[name] == review["models"][name]["raw_score"]
        assert frozen["warn"]["raw_score"] < scorer.review[name] < scorer.block[name]
        assert scorer.band(name, scorer.block[name]) == "Block"
        assert scorer.band(name, scorer.review[name]) == "Review"
        assert scorer.band(name, np.nextafter(scorer.block[name], -np.inf)) == "Review"
        assert scorer.band(name, np.nextafter(scorer.review[name], -np.inf)) == "Allow"


@needs_models
def test_band_rates_shown_in_the_app_are_read_from_metrics_json(app):
    metrics = M.load_json(ROOT / "reports" / "metrics.json")
    rates = app.get_scorer().rates
    assert rates["rows"] == 50_000
    for name in ("lgbm", "cnn"):
        saved = metrics["results"]["phreshphish_eval"][name]["all"]["metrics"]
        post = metrics["post_hoc"]["review"]["results"]["phreshphish_eval"][name]["all"]
        shown = rates["models"][name]
        assert shown["block_detection"] == saved["block_recall"]["value"]
        assert shown["block_fpr"] == saved["block_fpr"]["value"]
        assert shown["review_or_block_detection"] == post["metrics"]["review_recall"]["value"]
        assert sum(shown["phishing_share"].values()) == pytest.approx(1.0)
        assert shown["phishing_share"]["Block"] == pytest.approx(shown["block_detection"])
        assert shown["legitimate_share"]["Block"] + shown["legitimate_share"]["Review"] == pytest.approx(
            shown["review_or_block_fpr"])
        html = app.render_idle(rates)       # no "This URL" tags, so each row reads straight across
        assert (f"<strong>Block</strong></span></th><td class=\"pud-num\">{shown['block_detection'] * 100:.1f}%</td>"
                f"<td class=\"pud-num\">{shown['block_fpr'] * 100:.2f}%</td>") in html


# ------------------------------------------------------------------ behaviour

@needs_models
def test_each_result_has_scores_bands_agreement_contributions_and_latency(app):
    result = app.score_url("http://secure-login.account-verify.example.net/signin/")
    assert set(result["models"]) == {"lgbm", "cnn"}
    for m in result["models"].values():
        assert m["band"] in ("Block", "Review", "Allow") and 0.0 <= m["score"] <= 1.0
    assert result["agree"] == (result["models"]["lgbm"]["band"] == result["models"]["cnn"]["band"])
    assert len(result["contributions"]) == 5
    sizes = [abs(c["contribution"]) for c in result["contributions"]]
    assert sizes == sorted(sizes, reverse=True)
    assert all(c["feature"] in F.FEATURE_NAMES for c in result["contributions"])
    assert 0 < result["scoring_ms"] < result["latency_ms"] < 5000 and result["explanation_ms"] > 0
    html = app.render(result, app.get_scorer().rates)
    assert ("(2/2 models)" in html) == result["agree"]
    assert (app.DISAGREE in html and "The two results are not combined" in html) == (not result["agree"])
    assert "Not a probability" in html and "Risk score is not a probability" in html and " ms " in html


@needs_models
def test_contributions_and_base_value_add_up_to_model_a_raw_score(app):
    scorer = app.get_scorer()
    for url in INVENTED:
        raw, X = scorer.raw_scores(url)
        contributions = scorer.contributions(X)
        assert len(contributions) == len(F.FEATURE_NAMES) + 1
        assert contributions.sum() == pytest.approx(raw["lgbm"], abs=1e-6)


@needs_models
@pytest.mark.parametrize("bad, expected", [
    ("", "Enter a URL"), ("   \t\n", "Enter a URL"), (None, "Enter a URL"), (12345, "Enter a URL"),
    ("x" * 2001, "limit is 2,000"), ("https://example.com/" + "a" * 5000, "limit is 2,000"),
    ("http://", "no host name"), ("hello world", "no host name"),
])
def test_bad_input_is_rejected_with_a_message_and_no_stack_trace(app, bad, expected):
    result = app.score_url(bad)
    assert result == {"ok": False, "error": result["error"]} and expected in result["error"]
    html = app.analyse(bad)
    shown = {"Enter a URL": "No input. Enter a URL to analyze.", "limit is 2,000": "limit is 2,000",
             "no host name": "Not a URL: no host name found."}[expected]
    assert "Traceback" not in html and "pud-error" in html and 'role="alert"' in html and shown in html


@needs_models
def test_input_at_the_length_limit_is_scored_and_odd_input_never_raises(app):
    assert app.score_url("https://example.com/" + "a" * (2000 - len("https://example.com/")))["ok"]
    for odd in ("https://[::1", "http://exa\x00mple.com/", "\ud800", "https://例え.example/パス", "%%%", "://", "a" * 2000,
                "javascript:alert(1)", "https://example.com/" + "%" * 300):
        result = app.score_url(odd)
        assert isinstance(result, dict) and "ok" in result
        assert "Traceback" not in app.analyse(odd)


@needs_models
def test_user_text_is_escaped_in_the_page(app):
    html = app.analyse("https://example.com/<script>alert(1)</script><img src=x onerror=alert(2)>")
    assert "<script>" not in html and "<img" not in html and "&lt;script&gt;" in html


@needs_models
def test_query_string_and_fragment_do_not_change_the_score_and_the_page_says_so(app):
    plain = app.score_url("https://shop.example.com/products/item-48213")
    with_query = app.score_url("https://shop.example.com/products/item-48213?ref=newsletter&id=7#reviews")
    for name in ("lgbm", "cnn"):
        assert with_query["models"][name]["raw_score"] == plain["models"][name]["raw_score"]
    assert any("query string" in note for note in with_query["notes"]) and not plain["notes"]


@needs_models
def test_bare_homepage_is_called_out_and_an_empty_path_equals_a_slash(app):
    a, b = app.score_url("https://www.example.com"), app.score_url("https://www.example.com/")
    assert a["models"] == b["models"]
    assert any("bare homepage" in note for note in a["notes"])
    assert not any("bare homepage" in note for note in app.score_url("https://www.example.com/a")["notes"])


@needs_models
def test_the_app_never_opens_a_network_connection_when_scoring(app, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the app tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    assert app.score_url("http://secure-login.account-verify.example.net/signin/")["ok"]
    assert "pud-detect" in app.analyse("https://www.example.com/")


@needs_models
def test_the_interface_states_the_limits_and_builds(app):
    limits = " ".join(lead + " " + rest for lead, rest in app.LIMITS)
    for phrase in ("The URL is not opened", "Query string and fragment ignored", "Bare homepages carry little signal",
                   "Risk score is not a probability", "Evadable with benign-looking edits",
                   "Decision support, not a standalone control", "results are never combined"):
        assert phrase in limits
    page = app.analyse("https://www.example.com/")
    evasion = app.get_scorer().evasion
    for name in ("lgbm", "cnn"):
        for when in ("before", "after"):
            assert f"{evasion[name][when] * 100:.1f}%" in page
    demo = app.build_demo()
    assert {"/analyse", "/score"} <= {f"/{d.api_name}" for d in demo.fns.values() if d.api_visibility == "public"}


def test_test_urls_are_invented_on_reserved_names(app):
    assert len(app.EXAMPLES) >= 4
    for _, url in app.EXAMPLES:
        host = F.split_url(url).host
        reserved = host == "192.0.2.44" or F.registered_domain(host) in ("example.com", "example.net", "example.org")
        assert reserved, url


def _visible_text(page: str) -> str:
    import html as H
    return re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", " ", page)))


@needs_models
def test_report_shows_each_risk_score_and_result_in_words(app):
    rates = app.get_scorer().rates
    for _, url in app.EXAMPLES:
        result = app.score_url(url)
        a, b = result["models"]["lgbm"], result["models"]["cnn"]
        page = app.render(result, rates)
        text = _visible_text(page)
        assert (f"Model A: {a['band']}, risk score {a['score'] * 100:.1f}. "
                f"Model B: {b['band']}, risk score {b['score'] * 100:.1f}.") in text
        widths = [float(x) for x in re.findall(r'class="pud-meter"[^>]*><span style="width: ([0-9.]+)%"', page)]
        assert widths == pytest.approx([a["score"] * 100, b["score"] * 100], abs=0.06)
        for letter, m in (("A", a), ("B", b)):
            row = (rf'<th scope="row">Model {letter}</th>.*?<strong>{m["band"]}</strong>.*?'
                   rf'{m["score"] * 100:.1f}.*?{re.escape(format(m["raw_score"], "+.2f"))}')
            assert re.search(row, page), letter
        assert text.count("Not a probability") >= 1 and "Calibrated scores (0 to 1)" in text
        assert f"{a['score']:.3f}" in text and f"{b['score']:.3f}" in text       # the 0 to 1 value, in details
        assert page.count("<span class=pud-tag>This URL</span>") == 2
        contributions = re.search(r'<table class="pud-contrib">.*?</table>', page, re.S).group(0)
        assert contributions.count("<tr>") == 6                                  # header and five features
        expected = app.STATUS[a["band"]] if result["agree"] else app.DISAGREE
        assert re.search(r'id="pud-status"[^>]*>.*?<span>([^<]+)</span>', page).group(1) == expected


@needs_models
def test_status_label_and_next_step_for_every_pair_of_results(app):
    rates = app.get_scorer().rates
    base = app.score_url("http://secure-login.account-verify.example.net/signin/")
    for a in app.BANDS:
        for b in app.BANDS:
            result = {**base, "agree": a == b,
                      "models": {"lgbm": {**base["models"]["lgbm"], "band": a},
                                 "cnn": {**base["models"]["cnn"], "band": b}}}
            text = _visible_text(app.render(result, rates))
            if a == b:
                assert f"{a} (2/2 models)" in text and app.NEXT_STEP[a] in text and app.DISAGREE not in text
            else:
                assert app.DISAGREE in text and app.NEXT_STEP["split"] in text
                assert not any(app.STATUS[x] in text for x in app.BANDS)
                assert "The two results are not combined" in text


@needs_models
def test_wording_never_calls_a_result_clean_or_safe(app):
    """Allow is the strongest reassurance, and it is always followed by its caveat."""
    rates = app.get_scorer().rates
    pages = [app.analyse(url) for _, url in app.EXAMPLES] + [app.render_idle(rates), app.analyse(""),
                                                             app.KEY, app.HEADER, app.PRIVACY_SHORT]
    for page in pages:
        text = _visible_text(page)
        assert "clean" not in text.lower()
        for match in re.finditer(r"\bsafe\b", text, re.I):
            assert text[max(0, match.start() - 30):match.end()].endswith("not proof that the URL is safe"), text
        for match in re.finditer(r"Allow \(2/2 models\)", text):
            assert text[match.end():].lstrip().startswith(app.NOT_PROOF)
    assert "not proof that the URL is safe" in app.ACTION["Allow"] and "not proof that the URL is safe" in \
        app.NOT_PROOF
    page = app.analyse("https://docs.example.org/guides/getting-started/installation.html")
    assert "Allow (2/2 models)</span></h2><p class=pud-notproof>Allow is not proof that the URL is safe" in page


@needs_models
def test_the_tool_speaks_in_a_neutral_voice(app):
    """A tool, not a person: no first or second person anywhere on the page or in the triage note."""
    scorer = app.get_scorer()
    pages = [app.analyse(url) for _, url in app.EXAMPLES] + [
        app.render_idle(scorer.rates), app.analyse(""), app.analyse("hello world"), app.KEY, app.HEADER,
        app.PRIVACY_SHORT, app.PRIVACY_FULL, app.TRIAGE_CAVEAT]
    pages += [app.triage_note(scorer.score(url), scorer.model_sha256) for _, url in app.EXAMPLES]
    source = (APP_DIR / "app.py").read_text(encoding="utf-8")
    pages += re.findall(r'gr\.(?:HTML|Button|Textbox)\((.*?)elem_', source, re.S)
    for page in pages:
        text = _visible_text(page)
        assert not re.search(r"\b(we|our|ours|us|you|your|yours)\b", text, re.I), text
        assert not re.search(r"\b(I|I'm|I've|me|my)\b", text), text
        assert "checker" not in text.lower(), text


@needs_models
def test_reliability_counts_match_metrics_json_and_add_up(app):
    metrics = M.load_json(ROOT / "reports" / "metrics.json")
    text = _visible_text(app.render_idle(app.get_scorer().rates))
    for name in ("lgbm", "cnn"):
        entry = metrics["post_hoc"]["review"]["results"]["phreshphish_eval"][name]["all"]
        phishing = {band: entry["bands"]["phishing"][band] / entry["phishing"] for band in app.BANDS}
        legitimate = {band: entry["bands"]["legitimate"][band] / entry["legitimate"] for band in app.BANDS}
        assert sum(phishing.values()) == pytest.approx(1) and sum(legitimate.values()) == pytest.approx(1)
        for band in app.BANDS:
            assert f"{band} {phishing[band] * 100:.1f}% {legitimate[band] * 100:.2f}%" in text, (name, band)
        m = entry["metrics"]
        assert (f"Review or Block: {m['review_recall']['value'] * 100:.1f}% of phishing, "
                f"{m['review_fpr']['value'] * 100:.2f}% of legitimate.") in text
    assert "selected after the test results were seen (post hoc)" in text
    assert f"{metrics['datasets']['phreshphish_eval']['rows']:,} URLs" in text


@needs_models
def test_every_note_feature_and_error_has_plain_wording(app):
    assert set(app.PLAIN) == set(F.FEATURE_NAMES)
    for name in F.FEATURE_NAMES:
        for value in (0, 0.25, 1, 2, 3.5, 40):
            sentence = app.PLAIN[name](value)
            assert sentence.endswith(".") and "  " not in sentence, (name, value)
    notes = {note for url in ("https://shop.example.com/a?b=1", "example.org/", "https://example.com/" + "a" * 300,
                              "<hxxps://example[.]com/a>")
             for note in app.score_url(url)["notes"]}
    assert len(notes) == 6 and notes == set(app.NOTE_PLAIN)
    text = _visible_text(app.analyse("example.org/?x=1"))
    assert all(app.NOTE_PLAIN[note] in text for note in app.score_url("example.org/?x=1")["notes"])
    assert app._plain_error("The limit is unknown") == "The limit is unknown"


@needs_models
def test_report_links_only_to_its_own_sections_and_counts_them(app):
    rates = app.get_scorer().rates
    hostile = 'https://example.com/"><a href=//evil.example>x</a>'
    for page in (app.analyse("http://secure-login.account-verify.example.net/signin/"), app.analyse(hostile),
                 app.render_idle(rates)):
        tags = re.findall(r"<a\b[^>]*>", page)
        assert tags and all(re.fullmatch(r'<a href="#pud-[a-z]+">', t) for t in tags), tags
        for anchor in set(re.findall(r'href="#(pud-[a-z]+)"', page)):
            assert f'id="{anchor}"' in page
    page = app.analyse("https://shop.example.com/products/item-48213?ref=newsletter")
    counts = dict(re.findall(r"<span>([^<]+)</span><span class=pud-count>(\d+)</span>", page))
    assert counts == {"Notes": "1", "Contributions": "5", "Performance": "2", "Limitations": "6"}


@needs_models
@pytest.mark.parametrize("url", INVENTED + ["https://shop.example.co.uk/basket", "http://192.0.2.44:8080/x"])
def test_address_breakdown_is_what_the_feature_module_parsed(app, url):
    result = app.score_url(url)
    scored = F.normalize_url(url, strip_query=F.MAIN_STRIP_QUERY)
    described, parts = F.describe_url(scored), F.split_url(scored)
    features = F.extract_features(url, strip_query=F.MAIN_STRIP_QUERY)
    a = result["address"]
    assert {k: a[k] for k in ("scheme", "host", "registered_domain", "tld")} == {
        k: described[k] for k in ("scheme", "host", "registered_domain", "tld")}
    assert (a["port"], a["userinfo"], a["url_path"]) == (parts.port, parts.userinfo, parts.path or "/")
    assert a["subdomain_levels"] == features["subdomain_depth"] and a["host_is_ip"] == bool(features["host_is_ip"])
    assert a["punycode"] == bool(features["has_punycode"]) and a["brand_words"] == features["n_brand_words"]
    assert a["query_removed"] == F.normalization_changes(url, strip_query=True)["query_removed"]
    json.dumps(result)                                   # the /score API must still serialize

    def keys(obj):
        return (set(obj) | set().union(*(keys(v) for v in obj.values()))) if isinstance(obj, dict) else \
            set().union(*(keys(v) for v in obj)) if isinstance(obj, list) else set()
    # gradio_client treats any dict with a "path" key as a file and tries to download it
    assert "path" not in keys(result)


@needs_models
def test_analyst_section_shows_the_parts_escaped_and_claims_no_lookups(app):
    page = app.analyse("http://paypal.com.account-check.example.net/verify/login.php")
    text = _visible_text(page)
    assert "Registered domain (approximate) example.net" in text and "Subdomain levels 3" in text
    assert "A well-known brand name appears outside the registered domain." in text
    assert "No lookups performed" in text and "no WHOIS, DNS, hosting or reputation data" in text
    hostile = app.analyse('https://user"><script>x</script>@login.example.com/<b>p</b>')
    assert "<script>" not in hostile and "<b>p</b>" not in hostile
    idle = app.render_idle(app.get_scorer().rates)
    assert 'id="pud-analyst"' in idle and "Registered domain" not in idle


def test_defang_makes_every_scheme_and_dot_unclickable(app):
    assert app.defang("https://a.example.com/x.php?u=HTTP://b.example") == \
        "hxxps://a[.]example[.]com/x[.]php?u=HXXP://b[.]example"
    assert app.defang("no dots or schemes") == "no dots or schemes"


@needs_models
def test_triage_note_is_plain_defanged_text_with_both_results_and_model_hashes(app):
    from datetime import datetime, timezone
    scorer = app.get_scorer()
    for _, url in app.EXAMPLES + (("hostile", 'https://example.com/"><img src=x onerror=alert(1)>'),):
        result = scorer.score(url)
        note = app.triage_note(result, scorer.model_sha256, datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc))
        assert "Checked: 2026-01-02 03:04:05 UTC" in note
        assert not re.search(r"(?i)https?://", note) and not re.search(r"[A-Za-z0-9]\.[A-Za-z]", note.split(
            "Top factors")[0].split("Checker A")[0]), note        # the URL, host and domain lines are defanged
        assert app.defang(result["scored_url"]) in note
        for name, letter in (("lgbm", "A"), ("cnn", "B")):
            m = result["models"][name]
            assert f"Model {letter} (" in note and f"risk score {m['score'] * 100:.1f} of 100, result {m['band']}" in note
        assert note.count("towards phishing") + note.count("towards legitimate") == 5
        recorded = M.load_json(ROOT / "models" / "hashes.json")
        assert f"lgbm.txt sha256 {recorded['models/lgbm.txt']['sha256'][:12]}" in note
        assert f"cnn.onnx sha256 {recorded['models/cnn.onnx']['sha256'][:12]}" in note
        assert "not threat intelligence and not a verdict" in note and "not probabilities" in note
        assert "clean" not in note.lower() and not re.search(r"\bsafe\b", note, re.I)
    for bad in ("", "x" * 2001, "hello world", None):
        page, note = app.analyse_page(bad)
        assert note == "" and "pud-error" in page


@needs_models
def test_the_interface_gives_the_note_a_copy_button_and_keeps_the_note_out_of_the_api(app):
    demo = app.build_demo()
    boxes = [b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "pud-note"]
    assert len(boxes) == 1 and boxes[0].interactive is False and "copy" in boxes[0].buttons
    public = {d.api_name for d in demo.fns.values() if d.api_visibility == "public"}
    assert public == {"analyse", "score", "score_batch"}


@needs_models
def test_before_analysis_the_page_is_only_the_input_boxes(app):
    import gradio as gr
    demo = app.build_demo()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    for hidden in ("pud-key", "pud-output", "pud-note"):
        assert by_id[hidden].visible is False, hidden
    for shown in ("pud-url", "pud-score", "pud-urls", "pud-score-many"):
        assert by_id[shown].visible is not False, shown
    assert {b.elem_id for b in demo.blocks.values() if isinstance(b, gr.Button)} == {"pud-score", "pud-score-many"}
    assert [b.label for b in demo.blocks.values() if isinstance(b, gr.Tab)] == ["One URL", "Several URLs"]
    assert "pud-example" not in app.CSS


def test_background_script_is_attributed_inert_and_safe_to_inline(app):
    js = (APP_DIR / "background.js").read_text(encoding="utf-8")
    assert "MIT + Commons Clause" in js and "Copyright (c) 2026 David Haz" in js
    assert "</script" not in js.lower() and app.HEAD == f"<script>{js}</script>"
    # decorative only: no network, no storage, no reading of the page's text or the URL box
    for banned in ("fetch(", "XMLHttpRequest", "WebSocket", "sendBeacon", "localStorage", "sessionStorage",
                   "document.cookie", "innerHTML", "textContent", "querySelector", "import(", "http://"):
        assert banned not in js, banned
    assert "prefers-reduced-motion" in js and "pointer-events: none" in app.CSS


@needs_models
def test_page_before_a_link_is_checked_shows_the_rates_and_no_result(app):
    page = app.render_idle(app.get_scorer().rates)
    assert "Enter a URL to analyze" in page and "Measured performance" in page and "33.5%" in page
    assert not any(word in page for word in ("pud-meter", "<span class=pud-tag>This URL</span>", "pud-error",
                                             "pud-detect"))


def test_first_mentions_are_explained_and_the_link_is_said_to_be_never_opened(app):
    assert "machine-learning models" in app.HEADER and "never opened" in app.HEADER
    assert "course project" in app.HEADER and "not a security product" in app.HEADER
    for band in app.BANDS:
        assert f"<strong>{band}</strong></span>: {app.ACTION[band]}" in app.KEY
    assert "never opened" in app.PRIVACY_SHORT and "not stored or logged" in app.PRIVACY_SHORT
    assert app.LAUNCH_OPTIONS["run_history"] is False


def test_theme_gives_light_and_dark_mode_the_same_colours(app):
    theme = app.build_theme()
    for name in ("body_background_fill", "body_text_color", "input_background_fill", "border_color_primary",
                 "button_primary_background_fill", "button_primary_text_color"):
        assert getattr(theme, name) == getattr(theme, name + "_dark"), name
    assert theme.body_background_fill == app.CANVAS and theme.button_primary_background_fill == app.ACCENT
    assert theme.button_transition == "none" and "Trebuchet MS" in theme.font and "Consolas" in theme.font_mono


def test_status_colours_are_readable_on_both_surfaces(app):
    def luminance(colour):
        channels = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        channels = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    def contrast(a, b):
        high, low = sorted((luminance(a), luminance(b)), reverse=True)
        return (high + 0.05) / (low + 0.05)

    for band, colour in app.STATUS_COLOUR.items():
        assert f".pud-s-{band} {{ color: {colour}; }}" in app.CSS
        for surface in (app.CANVAS, app.PANEL):
            assert contrast(colour, surface) >= 4.5, (band, surface)


# ------------------------------------------------------------------ packaging

FORBIDDEN = ("torch", "scikit-learn", "sklearn", "shap", "matplotlib", "optuna", "onnx")
DIRECT = ("numpy", "lightgbm", "onnxruntime", "gradio")
def test_app_imports_only_what_the_image_installs():
    tree = ast.parse((APP_DIR / "app.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add(node.module.split(".")[0])
    third_party = imported - set(sys.stdlib_module_names) - {"src"}
    assert third_party == {"numpy", "lightgbm", "onnxruntime", "gradio"}
    source = (APP_DIR / "app.py").read_text(encoding="utf-8")
    assert "from src import features as F" in source and "MAIN_STRIP_QUERY" in source
    assert not re.search(r"\b(requests|urllib|httpx|http\.client|aiohttp)\b", source)
    # src/features.py itself needs only numpy and the standard library
    features = ast.parse((ROOT / "src" / "features.py").read_text(encoding="utf-8"))
    used = {a.name.split(".")[0] for n in ast.walk(features) if isinstance(n, ast.Import) for a in n.names} | {
        n.module.split(".")[0] for n in ast.walk(features) if isinstance(n, ast.ImportFrom) and n.level == 0}
    assert used - set(sys.stdlib_module_names) == {"numpy"}


def test_requirements_are_fully_pinned_and_small():
    lines = [line.strip() for line in (APP_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    names = [line.split("==")[0].lower() for line in lines]
    assert all(re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9][0-9A-Za-z.]*", line) for line in lines), "every line must be pinned"
    assert len(names) == len(set(names))
    assert names[:4] == list(DIRECT), "the four packages the app imports come first"
    assert not set(names) & set(FORBIDDEN)
    development = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert all(line in development for line in lines[:4]), "app versions must be the ones the models were tested with"


def test_dockerfile_uses_the_slim_image_port_7860_and_a_non_root_user():
    text = (APP_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^FROM python:3\.11-slim\s*$", text, re.M)
    assert "EXPOSE 7860" in text and "GRADIO_SERVER_PORT=7860" in text
    assert re.search(r"^USER user\s*$", text, re.M) and "useradd -m -u 1000 user" in text
    assert text.index("USER user") < text.index("pip install") < text.index("CMD")
    assert not any(word in text for word in ("torch", "scikit-learn", "shap"))


def test_space_card_header_declares_a_docker_space_on_port_7860():
    text = (APP_DIR / "README.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    header = text.split("---\n")[1]
    fields = dict(line.split(": ", 1) for line in header.splitlines() if ": " in line and not line.startswith(" "))
    assert fields["sdk"] == "docker" and fields["app_port"] == "7860"
    assert fields["colorFrom"] in ("red", "yellow", "green", "blue", "indigo", "purple", "pink", "gray")
    assert fields["colorTo"] in ("red", "yellow", "green", "blue", "indigo", "purple", "pink", "gray")
    assert {"title", "emoji", "short_description"} <= set(fields)


@needs_models
def test_numbers_in_the_space_card_are_the_saved_ones():
    card = (APP_DIR / "README.md").read_text(encoding="utf-8")
    metrics = M.load_json(ROOT / "reports" / "metrics.json")
    post = metrics["post_hoc"]
    for name, label in (("lgbm", "Model A"), ("cnn", "Model B")):
        entry = post["review"]["results"]["phreshphish_eval"][name]["all"]
        for band in ("Block", "Review", "Allow"):
            row = (f"| {label} | {band} | {entry['bands']['phishing'][band] / entry['phishing']:.1%} | "
                   f"{entry['bands']['legitimate'][band] / entry['legitimate']:.2%} |")
            assert row in card, row
        block = post["targeted_robustness"]["models"][name]["block"]
        assert f"from {block['before']['value']:.1%} to {block['after']['all']['value']:.1%} for {label}" in " ".join(card.split())
        wangchuk = post["review"]["results"]["wangchuk"][name]["all"]["metrics"]["review_fpr"]["value"]
        assert f"{wangchuk:.2%} ({label})" in card


@needs_models
def test_space_bundle_holds_what_the_image_needs_and_scores_the_same(tmp_path, app):
    builder = _load(APP_DIR / "build_space.py", "build_space")
    out = builder.build(tmp_path / "space")
    files = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert files == set(builder.FILES.values())
    # a clone of the Space keeps git's own files, loses strays, and a foreign folder is refused
    (out / ".gitattributes").write_text("*.onnx filter=lfs", encoding="utf-8")
    (out / "stray.txt").write_text("x", encoding="utf-8")
    builder.build(out)
    assert (out / ".gitattributes").exists() and not (out / "stray.txt").exists()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        builder.build(foreign)
    assert (foreign / "keep.txt").exists()
    (out / ".gitattributes").unlink()
    assert {"Dockerfile", "README.md", "app.py", "requirements.txt", "src/features.py", "models/lgbm.txt",
            "models/cnn.onnx", "reports/metrics.json"} <= files
    assert not any(name.endswith((".pt", ".parquet", ".ipynb")) for name in files)
    assert sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) < 15 * 1024 * 1024
    url = "http://secure-login.account-verify.example.net/signin/"
    done = subprocess.run(
        [sys.executable, "-c", "import json, app; print(json.dumps(app.score_url(%r)['models']))" % url],
        cwd=out, capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stderr[-2000:]
    bundled = json.loads(done.stdout.strip().splitlines()[-1])
    assert bundled == app.score_url(url)["models"]


# ------------------------------------------------------------------ the running container

CONTAINER = os.environ.get("PHISHING_APP_URL")


@needs_models
@pytest.mark.skipif(not CONTAINER, reason="set PHISHING_APP_URL to a running copy of the app, e.g. http://localhost:7860")
def test_running_container_gives_the_notebook_scores_and_bands():
    """The built image, called over HTTP, must give the scores and bands the notebooks saved."""
    import httpx

    api = CONTAINER.rstrip("/") + "/gradio_api/call/score"
    bands = {row["url"]: row for row in M.load_json(REFERENCE_BANDS)["urls"]}
    with httpx.Client(timeout=60) as client:
        for entry in M.load_json(REFERENCE)["urls"]:
            event = client.post(api, json={"data": [entry["url"]]}).json()["event_id"]
            body = client.get(f"{api}/{event}").text
            data = next(line[len("data: "):] for line in reversed(body.splitlines()) if line.startswith("data: "))
            got = json.loads(data)[0]
            assert got["ok"] and got["scored_url"] == entry["main_track_url"]
            for name in ("lgbm", "cnn"):
                assert got["models"][name]["raw_score"] == pytest.approx(entry[name]["raw_score"], abs=1e-6), entry["url"]
                assert got["models"][name]["score"] == pytest.approx(entry[name]["calibrated_score"], abs=1e-6)
                assert got["models"][name]["band"] == bands[entry["url"]][name]["band"]


@needs_models
@pytest.mark.skipif(not CONTAINER, reason="set PHISHING_APP_URL to a running copy of the app, e.g. http://localhost:7860")
def test_running_container_scores_a_batch_like_the_notebooks():
    import httpx

    api = CONTAINER.rstrip("/") + "/gradio_api/call/score_batch"
    reference = M.load_json(REFERENCE)["urls"][:5]
    with httpx.Client(timeout=60) as client:
        event = client.post(api, json={"data": ["\n".join(e["url"] for e in reference)]}).json()["event_id"]
        body = client.get(f"{api}/{event}").text
        data = next(line[len("data: "):] for line in reversed(body.splitlines()) if line.startswith("data: "))
    batch = json.loads(data)[0]
    assert batch["ok"] and batch["count"] == 5
    for row, entry in zip(batch["rows"], reference):
        assert row["scored_url"] == entry["main_track_url"]
        for name in ("lgbm", "cnn"):
            assert row["models"][name]["raw_score"] == pytest.approx(entry[name]["raw_score"], abs=1e-6)


# ------------------------------------------------------------------ defanged and wrapped input

#: Measured before the fix (2026-11-23): each was scored as a different string (hosts such as "hxxps[:]" or
#: "<https"), and the first one's Model A result moved from Block to Allow.
PASTED = [
    ("hxxps[:]//paypal.com.account-check[.]example[.]net/verify/login.php",
     "https://paypal.com.account-check.example.net/verify/login.php"),
    ("hxxp://secure-login[.]account-verify[.]example[.]net/signin/",
     "http://secure-login.account-verify.example.net/signin/"),
    ("<https://www.example.com/>", "https://www.example.com/"),
    ('"http://192.0.2.44/wp-admin/update/index.html"', "http://192.0.2.44/wp-admin/update/index.html"),
]


@pytest.mark.parametrize("pasted, expected", [
    ("hxxp://secure-login[.]account-verify[.]example[.]net/signin/",
     "http://secure-login.account-verify.example.net/signin/"),
    ("hxxps[:]//paypal.com.account-check[.]example[.]net/verify/login.php",
     "https://paypal.com.account-check.example.net/verify/login.php"),
    ("hXXps[://]a(dot)example{.}com/x", "hTTps://a.example.com/x"),
    ("HXXP://A[DOT]EXAMPLE(.)COM/", "HTTP://A.EXAMPLE.COM/"),
    ("example[.]com/login", "example.com/login"),
    ("https://a.example/go/hxxp://b[.]example/", "https://a.example/go/http://b.example/"),
    ("hxxps://a[.]example/x?u=b[.]example", "https://a.example/x?u=b[.]example"),
])
def test_undefang_restores_common_ioc_notation(app, pasted, expected):
    assert app.undefang(pasted) == (expected, {"unwrapped": False, "defanged": True})


@pytest.mark.parametrize("pasted, expected", [
    ("<https://www.example.com/>", "https://www.example.com/"),
    ('  "https://example.org/a"  ', "https://example.org/a"),
    ("`https://example.org/a`", "https://example.org/a"),
    ("'<https://example.org/a>'", "https://example.org/a"),
])
def test_undefang_removes_surrounding_brackets_and_quotes(app, pasted, expected):
    assert app.undefang(pasted) == (expected, {"unwrapped": True, "defanged": False})


@pytest.mark.parametrize("url", [
    "https://www.example.com/", "  https://docs.example.net/a.html  ", "https://hxxpsbank.example.com/404.html",
    "https://ipfs.example/ipfs/QmPPHXXPMLJduf3", "https://a.example/x.php?u=hXXps://b[.]example#c[.]d",
    "http://[2001:db8::1]:8080/a", "<https://example.com/", "https://example.com/'", "", "   ",
])
def test_undefang_leaves_ordinary_urls_exactly_as_given(app, url):
    # "hxxp" inside a host or an id is not a scheme, and the query and fragment are never touched
    assert app.undefang(url) == (url, {"unwrapped": False, "defanged": False})


def test_undefang_is_idempotent_and_never_raises(app):
    import random

    rng = random.Random(42)
    alphabet = "hxXp[].():/<>\"'`{}dot?#abc "
    for _ in range(3000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))
        once = app.undefang(text)[0]
        assert app.undefang(once)[0] == once, text
    assert app.undefang(None) == ("", {"unwrapped": False, "defanged": False})


PROCESSED = ROOT / "data" / "processed"


@pytest.mark.skipif(not (PROCESSED / "manifest.json").exists(), reason="data/processed/ is built by notebook 01")
def test_undefang_changes_no_url_in_any_processed_dataset(app):
    # The app undoes defanged notation before scoring. It must be inert on every URL the models were trained or
    # tested on, or the app's input would be handled differently from training's.
    for path in sorted(PROCESSED.glob("*.parquet")):
        frame = pd.read_parquet(path)
        for column in [c for c in frame.columns if c.startswith("url")]:
            changed = [u for u in frame[column].dropna() if app.undefang(u)[0] != u]
            assert not changed, (path.name, column, changed[:3])


def _keys(value) -> set:
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys(v) for v in value))
    return set()


@needs_models
@pytest.mark.parametrize("pasted, clean", PASTED)
def test_defanged_or_wrapped_input_scores_like_the_url_it_stands_for(app, pasted, clean):
    got, want = app.score_url(pasted), app.score_url(clean)
    assert got["ok"] and got["scored_url"] == want["scored_url"]
    for name in ("lgbm", "cnn"):
        assert got["models"][name] == want["models"][name]
    assert got["input_restored"] and not want["input_restored"]
    added = set(got["notes"]) - set(want["notes"])
    assert added and added <= {app.DEFANG_NOTE, app.UNWRAP_NOTE}


@needs_models
@pytest.mark.parametrize("url", INVENTED[:5] + [
    "https://docs.example.org/guides/getting-started/installation.html", "http://192.0.2.44/wp-admin/update/index.html"])
def test_the_triage_note_url_pasted_back_in_gives_the_same_result(app, url):
    scorer = app.get_scorer()
    note = app.triage_note(scorer.score(url), scorer.model_sha256)
    line = next(line for line in note.splitlines() if line.startswith("URL as scored (defanged): "))
    assert app.score_url(line.split(": ", 1)[1])["models"] == app.score_url(url)["models"]


@needs_models
def test_clean_reference_urls_are_not_marked_as_restored(app):
    for entry in M.load_json(REFERENCE)["urls"]:
        assert app.score_url(entry["url"])["input_restored"] is False, entry["url"]


@needs_models
@pytest.mark.parametrize("bad", ['https://exa"mple.com/', "https://<example>.com/", "https://ex`ample.com/",
                                 "https://a]b.example/", "https://[::1"])
def test_hosts_with_impossible_characters_are_rejected(app, bad):
    result = app.score_url(bad)
    assert result == {"ok": False, "error": result["error"]} and "host name contains characters" in result["error"]
    assert "the host name contains characters a host name cannot have" in _visible_text(app.analyse(bad))


@needs_models
def test_ipv6_hosts_are_still_scored(app):
    assert app.score_url("http://[2001:db8::1]:8080/a")["ok"]


# ------------------------------------------------------------------ several URLs at once

@needs_models
def test_score_many_keeps_order_skips_blanks_marks_duplicates_and_survives_bad_lines(app):
    text = "https://www.example.com/\n\n  \nhello world\nhxxps://www[.]example[.]com\nhttp://192.0.2.44/a/\n"
    batch = app.score_many(text)
    assert batch["ok"] and batch["count"] == 4
    assert [row["line"] for row in batch["rows"]] == [1, 4, 5, 6]
    first, bad, again, ip = batch["rows"]
    assert not bad["ok"] and "no host name" in bad["error"] and bad["input"] == "hello world"
    assert first["same_as_line"] is None and again["same_as_line"] == 1 and ip["same_as_line"] is None
    for row, url in ((first, "https://www.example.com/"), (again, "https://www.example.com/"),
                     (ip, "http://192.0.2.44/a/")):
        single = app.score_url(url)
        assert row["scored_url"] == single["scored_url"]
        for name in ("lgbm", "cnn"):
            assert row["models"][name] == {k: single["models"][name][k] for k in ("band", "score", "raw_score")}
        assert row["result"] == (app.STATUS[single["models"]["lgbm"]["band"]] if single["agree"] else app.DISAGREE)
    assert sum(batch["summary"].values()) == batch["count"] and batch["summary"][app.NOT_SCORED] == 1
    json.dumps(batch)                                      # the /score_batch API must serialize
    assert "path" not in _keys(batch)                      # gradio_client reads a "path" key as a file


@needs_models
def test_score_many_limits_and_empty_input(app):
    assert app.score_many("") == {"ok": False, "error": "Enter one URL per line."}
    assert app.score_many(None)["ok"] is False
    over = "\n".join(f"https://example.com/{i}" for i in range(app.MAX_BATCH_URLS + 1))
    assert "limit is 50 per batch" in app.score_many(over)["error"]
    at = "\n".join(f"https://example.com/{i}" for i in range(app.MAX_BATCH_URLS))
    assert app.score_many(at)["count"] == app.MAX_BATCH_URLS
    assert "characters long" in app.score_many("x" * (app.MAX_BATCH_URLS * 2002 + 1))["error"]


@needs_models
def test_batch_csv_parses_is_defanged_and_cannot_run_as_a_formula(app):
    import csv
    import io

    batch = app.score_many('https://www.example.com/a.html\n=HYPERLINK("http://x")\nhttp://192.0.2.44/b')
    text = app.batch_csv(batch)
    rows = list(csv.reader(io.StringIO(text)))
    assert tuple(rows[0]) == app.BATCH_CSV_COLUMNS and len(rows) == 4
    assert rows[1][1] == "hxxps://www[.]example[.]com/a[.]html" and rows[3][1] == "hxxp://192[.]0[.]2[.]44/b"
    assert rows[2][1].startswith("'=") and rows[2][2] == app.NOT_SCORED
    assert "http://" not in text and "https://" not in text
    assert app.batch_csv({"ok": False, "error": "x"}) == ""


@needs_models
def test_batch_page_escapes_input_states_the_caveats_and_never_combines(app):
    page = app.render_batch(app.score_many("https://example.com/<script>alert(1)</script>\n<img src=x>"))
    assert "<script>" not in page and "<img" not in page and "&lt;script&gt;" in page
    text = _visible_text(page)
    assert "Not a probability" in text and "Allow is not proof" in text and "(not opened)" in text
    assert "Limitations" in text and 'id="pud-status"' in page and "pud-summary" in page
    error = app.render_batch(app.score_many(""))
    assert 'role="alert"' in error and "Enter one URL per line." in error


@needs_models
def test_batch_skips_the_explanation_and_is_faster_than_single_checks(app):
    import time

    scorer = app.get_scorer()
    assert scorer.score("https://www.example.com/a", explain=False)["contributions"] == []
    urls = "\n".join(f"http://login{i}.account-verify.example.net/signin/" for i in range(20))
    started = time.perf_counter()
    app.score_many(urls)
    batch = time.perf_counter() - started
    started = time.perf_counter()
    for line in urls.splitlines():
        scorer.score(line)
    assert batch < time.perf_counter() - started


@needs_models
def test_the_batch_wording_is_neutral(app):
    examples = "\n".join(url for _, url in app.EXAMPLES)
    for page in (app.render_batch(app.score_many(examples)), app.render_batch(app.score_many("")),
                 app.batch_csv(app.score_many(examples))):
        text = _visible_text(page)
        assert not re.search(r"\b(we|our|ours|us|you|your|yours)\b", text, re.I), text
        assert not re.search(r"\b(I|I'm|I've|me|my)\b", text), text
