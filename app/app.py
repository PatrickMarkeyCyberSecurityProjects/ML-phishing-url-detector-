"""Phishing URL scorer: the Gradio app.

Scores a URL string with the two frozen models and shows, for each, a score
and one of three bands: Block, Review or Allow.

    Model A   LightGBM on 31 lexical features          models/lgbm.txt
    Model B   character-level CNN, ONNX Runtime on CPU  models/cnn.onnx

The app never fetches the URL. It analyses the string only, through
`src/features.py`, the same module the models were trained with, and it
removes the query string and fragment first (`MAIN_STRIP_QUERY`).

Thresholds
    Block    the frozen threshold set on validation for a 0.1% false-positive rate
    Review   the threshold set on validation for a 1% false-positive rate. It was
             added after the test results had been seen (docs/design_decisions.md, decision 22)
    Allow    everything below Review

The page is a neutral, utilitarian report: the result and a per-model detection
table, then notes, URL details, Model A's feature contributions, the measured
rates of each result (read from reports/metrics.json: what each band did on
the later-in-time test), limitations and technical details.

A "URL details" section shows the address as `src/features.py` parsed it, and
a text box below the report holds a plain-text triage note for a ticket: the
URL defanged, both results, Model A's top factors, the SHA-256 of the two model
files and the caveats. Nothing in either is looked up anywhere.

Pasted input in defanged notation (hxxp, [.]) or wrapped in <> or quotes is
restored first (`undefang`), with a note, so an indicator copied from a
ticket scores like the URL it stands for.

A second tab, "Several URLs", scores up to 50 lines at once (`score_many`):
a table with each model's result per line, and the same table as CSV text
with the URLs defanged. Nothing is written to disk.

Needs numpy, lightgbm, onnxruntime and gradio. It does not import torch,
scikit-learn, shap or anything else from `src/`.

Run locally:  python app/app.py     then open http://localhost:7860
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# In the repository this file is app/app.py; in the Space bundle it sits at the
# root, next to src/, models/ and reports/.
_HERE = Path(__file__).resolve().parent
ROOT = _HERE if (_HERE / "src" / "features.py").exists() else _HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import features as F  # noqa: E402  (the path is set just above)

MAX_URL_CHARS = 2000
MODELS = {"lgbm": "Model A: LightGBM on 31 URL features", "cnn": "Model B: character-level CNN"}
BANDS = ("Block", "Review", "Allow")

#: Invented URLs on names reserved for documentation (RFC 2606, RFC 5737). None is a real site. Not shown on
#: the page (it has only the URL box); the tests and the demo script's fallback use them.
EXAMPLES = (
    ("Bare homepage", "https://www.example.com/"),
    ("Documentation page", "https://docs.example.org/guides/getting-started/installation.html"),
    ("Shop link with a query string", "https://shop.example.com/products/item-48213?ref=newsletter&utm_source=mail"),
    ("Login-style host", "http://secure-login.account-verify.example.net/signin/"),
    ("Brand name in a subdomain", "http://paypal.com.account-check.example.net/verify/login.php"),
    ("IP address host", "http://192.0.2.44/wp-admin/update/index.html"),
)

FEATURE_TEXT = {
    "url_len": "length of the URL", "host_len": "length of the host name", "path_len": "length of the path",
    "query_len": "length of the query string", "n_digits": "number of digits", "n_letters": "number of letters",
    "n_special": "number of other characters", "digit_ratio": "share of digits", "letter_ratio": "share of letters",
    "special_ratio": "share of other characters", "host_digit_ratio": "share of digits in the host name",
    "url_entropy": "randomness of the URL's characters", "host_entropy": "randomness of the host name's characters",
    "subdomain_depth": "number of subdomain levels", "path_depth": "number of path segments",
    "n_query_params": "number of query parameters", "n_dots": "number of dots", "n_hyphens": "number of hyphens",
    "n_host_hyphens": "number of hyphens in the host name", "n_pct_encoded": "number of %-encoded characters",
    "tld_len": "length of the top-level domain", "tld_is_common": "top-level domain is a long-established one",
    "tld_is_cc": "top-level domain is a country code", "host_is_ip": "host is an IP address",
    "has_at": "contains an @", "has_double_slash": "contains // after the host", "has_port": "names a port",
    "has_punycode": "host uses non-ASCII or punycode", "is_https": "scheme is https",
    "n_brand_words": "brand names outside the registered domain",
    "n_cred_words": "credential words outside the registered domain",
}


DEFANG_NOTE = "Defanged notation (hxxp, [.]) was restored before scoring."
UNWRAP_NOTE = "Surrounding brackets or quotes were removed before scoring."
#: Characters no host name can contain. "[" and "]" are allowed only around an IPv6 literal.
_NOT_IN_HOST = frozenset('<>"\'`{}|\\^[]')


def _plausible_host(host: str) -> bool:
    if host.startswith("[") and host.endswith("]") and host.count("[") == host.count("]") == 1:
        return True
    return not any(ch.isspace() or ch in _NOT_IN_HOST for ch in host)


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Scorer:
    """The two frozen models, their thresholds and calibrators, and the measured band rates."""

    def __init__(self, root: Path = ROOT) -> None:
        import lightgbm as lgb
        import onnxruntime as ort

        models = Path(root) / "models"
        #: SHA-256 of the two model files as loaded, quoted in the triage note so a ticket names the exact models
        self.model_sha256 = {name: hashlib.sha256((models / file).read_bytes()).hexdigest()
                             for name, file in (("lgbm", "lgbm.txt"), ("cnn", "cnn.onnx"))}
        self.booster = lgb.Booster(model_file=str(models / "lgbm.txt"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(models / "cnn.onnx"), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        review = _load_json(models / "review_operating.json")
        self.block, self.review, self.calibration = {}, {}, {}
        for name in MODELS:
            operating = _load_json(models / f"{name}_operating.json")
            if not operating["strip_query"]:
                raise ValueError(f"{name} is not a main-track model")
            self.block[name] = float(operating["thresholds"]["block"]["raw_score"])
            self.review[name] = float(review["models"][name]["raw_score"])
            self.calibration[name] = (np.asarray(operating["calibration"]["x"], dtype=np.float64),
                                      np.asarray(operating["calibration"]["y"], dtype=np.float64))
            if not self.review[name] < self.block[name]:
                raise ValueError(f"{name}: the review threshold must lie below the block threshold")
        metrics = _load_json(Path(root) / "reports" / "metrics.json")
        self.rates = self._band_rates(metrics)
        self.evasion = self._evasion(metrics)

    # ------------------------------------------------------------------ measured rates

    @staticmethod
    def _band_rates(metrics: dict) -> dict:
        """What each band did on the later-in-time test, read from reports/metrics.json."""
        dataset = metrics["datasets"]["phreshphish_eval"]
        results = metrics["post_hoc"]["review"]["results"]["phreshphish_eval"]
        out = {"test_set": dataset["title"], "rows": dataset["rows"], "models": {}}
        for name in MODELS:
            entry = results[name]["all"]
            m = entry["metrics"]
            out["models"][name] = {
                "block_detection": m["block_recall"]["value"], "block_fpr": m["block_fpr"]["value"],
                "review_or_block_detection": m["review_recall"]["value"],
                "review_or_block_fpr": m["review_fpr"]["value"],
                "phishing_share": {b: entry["bands"]["phishing"][b] / entry["phishing"] for b in BANDS},
                "legitimate_share": {b: entry["bands"]["legitimate"][b] / entry["legitimate"] for b in BANDS},
            }
        return out

    @staticmethod
    def _evasion(metrics: dict) -> dict:
        """Share of phishing blocked before and after three legitimate-looking edits (reports/metrics.json)."""
        test = metrics["post_hoc"]["targeted_robustness"]
        return {name: {"before": test["models"][name]["block"]["before"]["value"],
                       "after": test["models"][name]["block"]["after"]["all"]["value"]} for name in MODELS}

    # ------------------------------------------------------------------ scoring

    def band(self, name: str, raw: float) -> str:
        if raw >= self.block[name]:
            return "Block"
        if raw >= self.review[name]:
            return "Review"
        return "Allow"

    def raw_scores(self, url: str) -> tuple[dict, np.ndarray]:
        """Raw scores (log-odds) of both models, and the feature row Model A read."""
        X = F.feature_matrix([url], strip_query=F.MAIN_STRIP_QUERY)
        lgbm = float(self.booster.predict(X, raw_score=True, num_threads=1)[0])
        ids = F.encode_batch([url], F.DEFAULT_MAX_LEN, F.MAIN_STRIP_QUERY)
        cnn = float(self.session.run(["raw_score"], {"ids": ids})[0][0])
        return {"lgbm": lgbm, "cnn": cnn}, X

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Model A's 31 feature contributions and, last, its base value. They add up to its raw score."""
        return np.asarray(self.booster.predict(X, pred_contrib=True, num_threads=1), dtype=np.float64)[0]

    def score(self, url: object, explain: bool = True) -> dict:
        """Score one URL string. Never raises and never fetches anything.

        With `explain=False` (the batch tab) Model A's feature contributions are skipped: they take most of the
        time and the batch table does not show them. `contributions` is then empty and `base_value` None.
        """
        started = time.perf_counter()
        if not isinstance(url, str) or not url.strip():
            return {"ok": False, "error": "Enter a URL to score."}
        if len(url) > MAX_URL_CHARS:
            return {"ok": False, "error": f"That input is {len(url):,} characters long. The limit is "
                                          f"{MAX_URL_CHARS:,} characters."}
        try:
            # Defanged notation (hxxp, [.]) and surrounding <> or quotes are undone first, so a pasted indicator
            # scores like the URL it stands for. See undefang.
            url, restored = undefang(url)
            scored = F.normalize_url(url, strip_query=F.MAIN_STRIP_QUERY)
            parts = F.split_url(scored)
            if not parts.host or any(ch.isspace() for ch in parts.host):
                return {"ok": False, "error": "That does not look like a URL: no host name was found in it."}
            if not _plausible_host(parts.host):
                return {"ok": False, "error": "That does not look like a URL: the host name contains characters a "
                                              "host name cannot have."}
            changes = F.normalization_changes(url, strip_query=F.MAIN_STRIP_QUERY)
            raw, X = self.raw_scores(url)
            scored_at = time.perf_counter()
            contributions = self.contributions(X) if explain else None
            explained_at = time.perf_counter()
        except Exception:      # a scoring failure must never reach the user as a stack trace
            return {"ok": False, "error": "That input could not be scored."}

        models = {}
        for name in MODELS:
            x, y = self.calibration[name]
            models[name] = {"label": MODELS[name], "raw_score": raw[name],
                            "score": float(np.interp(raw[name], x, y)), "band": self.band(name, raw[name])}
        order = np.argsort(-np.abs(contributions[:-1]))[:5] if explain else []
        top = [{"feature": F.FEATURE_NAMES[i], "meaning": FEATURE_TEXT[F.FEATURE_NAMES[i]],
                "value": float(X[0, i]), "contribution": float(contributions[i]),
                "pushes_towards": "phishing" if contributions[i] > 0 else "legitimate"} for i in order]
        notes = []
        if restored["defanged"]:
            notes.append(DEFANG_NOTE)
        if restored["unwrapped"]:
            notes.append(UNWRAP_NOTE)
        if changes["query_removed"]:
            notes.append("The query string or fragment (everything from the first ? or #) was removed before "
                         "scoring. The models never see it.")
        if parts.path in ("", "/"):
            notes.append("This is a bare homepage: there is no path. The models have little to go on, and their "
                         "scores for homepages are less reliable.")
        if changes["scheme_added"]:
            notes.append("No scheme was given, so https:// was assumed.")
        if len(scored) > F.DEFAULT_MAX_LEN:
            notes.append(f"Model B reads only the first {F.DEFAULT_MAX_LEN} characters.")
        # The address as the feature module parsed it, and some of Model A's own feature values. Nothing here is
        # parsed again in the app, and nothing is looked up: these are facts about the string, not intelligence.
        described = F.describe_url(scored)
        value = dict(zip(F.FEATURE_NAMES, (float(v) for v in X[0])))
        # No key may be called "path": gradio_client reads a dict with one as a file to download.
        address = {
            "scheme": described["scheme"], "host": described["host"],
            "registered_domain": described["registered_domain"], "tld": described["tld"],
            "port": parts.port, "userinfo": parts.userinfo, "url_path": parts.path or "/",
            "query_removed": changes["query_removed"],
            "subdomain_levels": int(value["subdomain_depth"]), "host_is_ip": value["host_is_ip"] >= 0.5,
            "punycode": value["has_punycode"] >= 0.5, "has_at": value["has_at"] >= 0.5,
            "is_https": value["is_https"] >= 0.5, "brand_words": int(value["n_brand_words"]),
            "credential_words": int(value["n_cred_words"]),
        }
        return {
            "ok": True, "scored_url": scored, "notes": notes, "models": models, "address": address,
            "input_restored": restored["defanged"] or restored["unwrapped"],
            "agree": models["lgbm"]["band"] == models["cnn"]["band"],
            "contributions": top, "base_value": float(contributions[-1]) if explain else None,
            "scoring_ms": (scored_at - started) * 1000.0,
            "explanation_ms": (explained_at - scored_at) * 1000.0,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }


_SCORER: Scorer | None = None


def get_scorer() -> Scorer:
    global _SCORER
    if _SCORER is None:
        _SCORER = Scorer()
    return _SCORER


def score_url(url: str) -> dict:
    """Score a URL string with both models. The URL is never fetched.

    Returns each model's raw score, calibrated score (not a probability) and
    band (Block, Review or Allow), whether the models agree, Model A's five
    largest feature contributions, the address as `src/features.py` parsed it
    (`address`), and the time taken in milliseconds
    (`scoring_ms` for both models, `explanation_ms` for the contributions,
    `latency_ms` for everything).
    """
    return get_scorer().score(url)


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #
#
# A utilitarian analysis report in a neutral voice, with no first person anywhere on the page: the result and
# the per-model detection table first, then notes, URL details, Model A's feature contributions, measured
# performance, limitations and technical details. The two results are never combined, and Allow is never
# presented as proof of safety. Nothing built from the submitted text becomes a link, an image or a script:
# it is escaped text only.

# Graphite surfaces, off-white text, one indigo for actions, focus and meters.
CANVAS, PANEL = "#242629", "#303236"        # the page; cards and inputs
TEXT, MUTED = "#F3F3F1", "#C3C5C9"          # primary and secondary text
LINE, EDGE = "#595C62", "#85888F"           # rules; outlines of inputs and meter tracks
ACCENT, INK = "#B1B8ED", "#232539"          # the indigo; text on the indigo
#: Status colours, always shown with the word. Contrast on CANVAS / PANEL: 7.05 / 5.97, 8.80 / 7.45, 8.38 / 7.09.
STATUS_COLOUR = {"Block": "#F19A8F", "Review": "#E6C07A", "Allow": "#8FCF9F"}

#: Short names of the two models, and their method.
SHORT = {"lgbm": "Model A", "cnn": "Model B"}
METHOD = {"lgbm": "LightGBM, 31 lexical features", "cnn": "Character-level CNN"}
#: What each result means.
ACTION = {"Block": "score above the threshold set for a 0.1% false-positive rate on validation",
          "Review": "score above the threshold set for a 1% false-positive rate on validation",
          "Allow": "score below both thresholds; not proof that the URL is safe"}
#: The status line when both models give the same result. A split has a line of its own; results are never combined.
STATUS = {band: f"{band} (2/2 models)" for band in BANDS}
DISAGREE = "Split result (no combined verdict)"
#: Shown directly under an Allow status, every time.
NOT_PROOF = "Allow is not proof that the URL is safe. Only the URL string was analyzed."
NEXT_STEP = {
    "Block": "Do not open. Contact the sender through a known channel if needed, and handle per local policy.",
    "Review": "Hold. Verify by another method (sandbox, reputation lookup or sender confirmation) before opening.",
    "Allow": "No indicator from this check. Verify by another method if the URL was unexpected or asks for "
             "credentials, payment or a download.",
    "split": "Hold. Verify by another method before opening. The two results are not combined.",
}

#: The page's wording for the notes `Scorer.score` returns. The API keeps the original sentences.
NOTE_PLAIN = {
    "The query string or fragment (everything from the first ? or #) was removed before scoring. The models never "
    "see it.": "Query string and fragment (from the first ? or #) removed before scoring.",
    "This is a bare homepage: there is no path. The models have little to go on, and their scores for homepages "
    "are less reliable.": "Bare homepage (no path): little lexical signal, so results are less reliable.",
    "No scheme was given, so https:// was assumed.": "No scheme given; https:// assumed.",
    f"Model B reads only the first {F.DEFAULT_MAX_LEN} characters.":
        f"Model B reads only the first {F.DEFAULT_MAX_LEN} characters; the rest was not scored.",
    DEFANG_NOTE: "Defanged input (hxxp, [.]) restored before scoring.",
    UNWRAP_NOTE: "Surrounding brackets or quotes removed before scoring.",
}

#: The limits of the tool. {a_before} etc. are filled from reports/metrics.json.
LIMITS = (
    ("URL string only.", "The URL is not opened, so page content, redirects, link shorteners, domain age and "
     "reputation are not assessed."),
    ("Query string and fragment ignored.", "Everything from the first ? or # is removed before scoring."),
    ("Bare homepages carry little signal.", "A URL with no path gives the models little to analyze."),
    ("Risk score is not a probability.", "A risk score of 90 does not mean a 90% chance of phishing."),
    ("Evadable with benign-looking edits.", "In testing, adding <code>www.</code>, switching to <code>https</code> "
     "and appending an ordinary path reduced phishing blocked from {a_before} to {a_after} (Model A) and from "
     "{b_before} to {b_after} (Model B)."),
    ("Decision support, not a standalone control.", "Use with other controls. The two models can disagree, and "
     "their results are never combined."),
)

PRIVACY_SHORT = ("Analyzed as text only: the URL is never opened, fetched or resolved. Submissions are not stored or "
                 "logged.")
PRIVACY_FULL = (
    "Data handling: a submitted URL is sent to the server running this page and processed as text. The server "
    "never connects to it, holds it in memory only during scoring, does not write it to disk or logs, and keeps no "
    "history of checks in the browser. A link of the form <code>/?url=…</code> places the URL in the browser's own "
    "history. The hosting service may keep its own request logs, outside this project's control.")


def _pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def _risk(score: float) -> str:
    """A 0 to 1 score as the risk score shown on the page: out of 100, one decimal."""
    return f"{score * 100:.1f}"


def _per(shares: dict, n: int) -> dict:
    """Shares as whole counts out of `n` that add up to `n` (largest remainder), for "about 33 in 100"."""
    raw = {band: shares[band] * n for band in BANDS}
    counts = {band: int(raw[band]) for band in BANDS}
    for band in sorted(BANDS, key=lambda b: raw[b] - counts[b], reverse=True)[: n - sum(counts.values())]:
        counts[band] += 1
    return counts


def _n(value: float, one: str, many: str, none: str | None = None) -> str:
    count = int(round(value))
    if count == 0 and none:
        return none
    return f"{count:,} {one if count == 1 else many}"


def _yes_no(yes: str, no: str):
    return lambda v: yes if v >= 0.5 else no


#: One plain sentence for each of the 31 features, given its value in this URL.
PLAIN = {
    "url_len": lambda v: f"The URL is {_n(v, 'character', 'characters')} long.",
    "host_len": lambda v: f"The host is {_n(v, 'character', 'characters')} long.",
    "path_len": lambda v: ("There is nothing after the host, or only a /." if v <= 1 else
                           f"The part after the host is {_n(v, 'character', 'characters')} long."),
    "query_len": lambda v: ("There is no query (the part after ?)." if v == 0 else
                            f"The query (the part after ?) is {_n(v, 'character', 'characters')} long."),
    "n_digits": lambda v: f"The URL contains {_n(v, 'digit', 'digits', 'no digits')}.",
    "n_letters": lambda v: f"The URL contains {_n(v, 'letter', 'letters', 'no letters')}.",
    "n_special": lambda v: f"The URL contains {_n(v, 'punctuation mark or symbol', 'punctuation marks or symbols')}.",
    "digit_ratio": lambda v: f"{v * 100:.0f}% of the URL is digits.",
    "letter_ratio": lambda v: f"{v * 100:.0f}% of the URL is letters.",
    "special_ratio": lambda v: f"{v * 100:.0f}% of the URL is punctuation or symbols.",
    "host_digit_ratio": lambda v: f"{v * 100:.0f}% of the host is digits.",
    "url_entropy": lambda v: f"The URL's mix of characters scores {v:.2f} for randomness (higher looks more "
                             "random).",
    "host_entropy": lambda v: f"The host's mix of characters scores {v:.2f} for randomness (higher looks more "
                              "random).",
    "subdomain_depth": lambda v: ("The host has no extra parts in front of the main name." if v == 0 else
                                  f"The host has {_n(v, 'extra part', 'extra parts')} in front of the main "
                                  "name, such as www. or login."),
    "path_depth": lambda v: ("The URL has no folders after the host." if v == 0 else
                             f"The URL has {_n(v, 'folder level', 'folder levels')} after the host."),
    "n_query_params": lambda v: f"The query has {_n(v, 'setting', 'settings', 'no settings')} (name=value pairs).",
    "n_dots": lambda v: f"The URL contains {_n(v, 'dot', 'dots', 'no dots')}.",
    "n_hyphens": lambda v: f"The URL contains {_n(v, 'hyphen', 'hyphens', 'no hyphens')}.",
    "n_host_hyphens": lambda v: f"The host contains {_n(v, 'hyphen', 'hyphens', 'no hyphens')}.",
    "n_pct_encoded": lambda v: f"The URL contains {_n(v, '%-encoded character', '%-encoded characters', 'no %-encoded characters')}.",
    "tld_len": lambda v: f"The ending of the host (such as .com) is {_n(v, 'letter', 'letters')} long.",
    "tld_is_common": _yes_no("The host ends in a long-established ending: .com, .org, .net, .edu, .gov, .mil "
                             "or .int.", "The host does not end in a long-established ending such as .com, "
                             ".org or .net."),
    "tld_is_cc": _yes_no("The host ends in a two-letter country ending, such as .uk or .de.",
                         "The host does not end in a two-letter country ending."),
    "host_is_ip": _yes_no("The host is an IP address, not a name.",
                          "The host is a name, not an IP address."),
    "has_at": _yes_no("The URL contains an @ sign.", "The URL contains no @ sign."),
    "has_double_slash": _yes_no("The URL contains // after the host.",
                                "The URL contains no // after the host."),
    "has_port": _yes_no("The URL names a port: a number after a colon in the host.",
                        "The URL names no port."),
    "has_punycode": _yes_no("The host contains non-English characters or their encoded form (xn--).",
                            "The host uses only plain letters, digits and hyphens."),
    "is_https": _yes_no("The URL starts with https.", "The URL does not start with https."),
    "n_brand_words": lambda v: ("No well-known brand name (such as paypal or microsoft) outside the registered "
                                "domain." if v == 0 else
                                f"{_n(v, 'well-known brand name', 'well-known brand names')} (such as paypal or "
                                "microsoft) outside the registered domain, for example in a subdomain or the path."),
    "n_cred_words": lambda v: ("No login-type word (login, verify, secure, account, update) outside the registered "
                               "domain." if v == 0 else
                               f"{_n(v, 'login-type word', 'login-type words')} (login, verify, secure, account, "
                               "update) outside the registered domain."),
}


def _badge(band: str) -> str:
    """A result word with its colour bar. The word is always there; colour is never alone."""
    return (f'<span class="pud-res pud-s-{band}"><span class="pud-bar" aria-hidden="true"></span>'
            f"<strong>{band}</strong></span>")


def _meter(score: float) -> str:
    """A horizontal bar: full width is a risk score of 100. Built from the number only."""
    share = min(max(score, 0.0), 1.0) * 100.0
    return f'<span class="pud-meter" aria-hidden="true"><span style="width: {share:.1f}%"></span></span>'


def _verdict(result: dict, rates: dict) -> str:
    models = result["models"]
    a, b = models["lgbm"]["band"], models["cnn"]["band"]
    status, key = (STATUS[a], a) if result["agree"] else (DISAGREE, "split")
    summary = (f'Model A: {a}, risk score {_risk(models["lgbm"]["score"])}. '
               f'Model B: {b}, risk score {_risk(models["cnn"]["score"])}.')
    rows = "".join(
        f'<tr><th scope="row">{SHORT[name]}</th><td>{METHOD[name]}</td><td>{_badge(m["band"])}</td>'
        f'<td><span class="pud-scorecell"><span class="pud-num">{_risk(m["score"])}</span>{_meter(m["score"])}</span>'
        f'</td><td class="pud-num">{m["raw_score"]:+.2f}</td></tr>' for name, m in models.items())
    missed = {name: rates["models"][name]["phishing_share"]["Allow"] for name in MODELS}
    return (
        '<section id="pud-result" class="pud-card pud-verdict" aria-labelledby="pud-status">'
        '<p class="pud-kicker">Result</p>'
        f'<h2 id="pud-status" tabindex="-1" class="pud-status pud-s-{key}"><span class="pud-bar" aria-hidden="true">'
        f"</span><span>{status}</span></h2>"
        f'{f"<p class=pud-notproof>{NOT_PROOF}</p>" if key == "Allow" else ""}'
        f'<p class="pud-summary">{summary}</p>'
        '<p class="pud-micro">URL analyzed (not opened)</p>'
        f'<div class="pud-url">{html.escape(result["scored_url"])}</div>'
        '<div class="pud-scroll" tabindex="0" role="region" aria-label="Detection by model">'
        '<table class="pud-detect"><thead><tr><th scope="col">Model</th><th scope="col">Method</th>'
        '<th scope="col">Result</th><th scope="col">Risk score (0 to 100)</th>'
        f'<th scope="col" class="pud-num">Raw (log-odds)</th></tr></thead><tbody>{rows}</tbody></table></div>'
        "<p class=\"pud-define\">Risk score: calibrated model output, 0 to 100. Not a probability. Each result is set "
        "from the model's raw score, so equal rounded risk scores can map to different results.</p>"
        f'<p class="pud-next"><strong>Handling:</strong> {NEXT_STEP[key]}</p>'
        f'<p class="pud-trust"><strong>Miss rate in testing:</strong> Model A returned Allow for '
        f'{_pct(missed["lgbm"])} of phishing URLs, Model B for {_pct(missed["cnn"])}. '
        '<a href="#pud-reliable">Measured performance</a> · <a href="#pud-limits">Limitations</a></p></section>')


#: The page's wording for the messages `Scorer.score` returns for unusable input, matched on how they start.
ERROR_PLAIN = (
    ("Enter a URL", "No input. Enter a URL to analyze."),
    ("That does not look like a URL: the host name contains",
     "Not a URL: the host name contains characters a host name cannot have."),
    ("That does not look like a URL", "Not a URL: no host name found."),
    ("That input could not be scored", "Input could not be scored."),
)


def _plain_error(message: str) -> str:
    """The error message in the page's words. Unknown messages, such as the length limit, are shown as they are."""
    return next((plain for start, plain in ERROR_PLAIN if message.startswith(start)), message)


def _waiting(message: str, alert: bool = False) -> str:
    role = ' role="alert"' if alert else ""
    title = "Input rejected" if alert else "Result"
    return ('<section id="pud-result" class="pud-card pud-verdict" aria-labelledby="pud-status">'
            f'<p class="pud-kicker">{title}</p><p id="pud-status" tabindex="-1" class="pud-wait{" pud-error" if alert else ""}"'
            f"{role}>{html.escape(message)}</p></section>")


IDLE = '<p class="pud-muted">Shown after analysis.</p>'


def _notes(result: dict | None) -> str:
    if result is None:
        body = IDLE
    elif not result["notes"]:
        body = "<p>None.</p>"
    else:
        body = ('<ul class="pud-warnings">' + "".join(
            f"<li>{html.escape(NOTE_PLAIN.get(note, note))}</li>" for note in result["notes"]) + "</ul>")
    return ('<section id="pud-notes" class="pud-card" aria-labelledby="pud-notes-title">'
            f'<h2 id="pud-notes-title">Notes</h2>{body}</section>')


def _reasons(result: dict | None) -> str:
    intro = ("<p>Top 5 of Model A's 31 features by absolute contribution to its raw score (log-odds). Positive "
             "values push towards phishing. Model B (character-level CNN) provides no per-feature attribution.</p>")
    if result is None:
        body = IDLE
    else:
        rows = "".join(
            f'<tr><td class="pud-num">{c["contribution"]:+.2f}</td><td>{c["pushes_towards"]}</td>'
            f'<td><code>{html.escape(c["feature"])}</code></td><td class="pud-num">{c["value"]:.3g}</td>'
            f'<td>{html.escape(PLAIN[c["feature"]](c["value"]))}</td></tr>' for c in result["contributions"])
        body = ('<div class="pud-scroll" tabindex="0" role="region" aria-label="Model A feature contributions">'
                '<table class="pud-contrib"><thead><tr><th scope="col" class="pud-num">Contribution</th>'
                '<th scope="col">Towards</th><th scope="col">Feature</th><th scope="col" class="pud-num">Value</th>'
                f'<th scope="col">Observation</th></tr></thead><tbody>{rows}</tbody></table></div>')
    return ('<section id="pud-why" class="pud-card" aria-labelledby="pud-why-title">'
            f'<h2 id="pud-why-title">Feature contributions (Model A)</h2>{intro}{body}</section>')


def _reliability(here: dict, rates: dict) -> str:
    cards = []
    for name in MODELS:
        r = rates["models"][name]
        p, legit = r["phishing_share"], r["legitimate_share"]
        rows = "".join(
            f'<tr class="{"pud-here" if band == here.get(name) else ""}"><th scope="row">{_badge(band)}'
            f'{"<span class=pud-tag>This URL</span>" if band == here.get(name) else ""}</th>'
            f'<td class="pud-num">{_pct(p[band])}</td><td class="pud-num">{_pct(legit[band], 2)}</td></tr>'
            for band in BANDS)
        cards.append(
            f'<div class="pud-rely"><h3>{SHORT[name]} <span class="pud-h3-sub">{METHOD[name]}</span></h3>'
            f'<div class="pud-scroll" tabindex="0" role="region" aria-label="{SHORT[name]}: measured rates">'
            '<table><thead><tr><th scope="col">Result</th><th scope="col" class="pud-num">Phishing URLs</th>'
            f'<th scope="col" class="pud-num">Legitimate URLs</th></tr></thead><tbody>{rows}</tbody></table></div>'
            f'<p class="pud-define">Review or Block: {_pct(r["review_or_block_detection"])} of phishing, '
            f'{_pct(r["review_or_block_fpr"], 2)} of legitimate.</p></div>')
    return (
        '<section id="pud-reliable" class="pud-card" aria-labelledby="pud-reliable-title">'
        '<h2 id="pud-reliable-title">Measured performance</h2>'
        f'<p>Test set: {html.escape(rates["test_set"])}, {rates["rows"]:,} URLs, all collected after the training '
        "data. Share of each class assigned each result.</p>"
        f'<div class="pud-rely-grid">{"".join(cards)}</div>'
        '<p class="pud-caveat">Block threshold: 0.1% false-positive rate on validation. Review threshold: 1% on '
        "validation, selected after the test results were seen (post hoc), so its rates are less reliable than "
        "Block's. Rates on other data sources and on real traffic differ.</p></section>")


def _limits(evasion: dict) -> str:
    values = {"a_before": _pct(evasion["lgbm"]["before"]), "a_after": _pct(evasion["lgbm"]["after"]),
              "b_before": _pct(evasion["cnn"]["before"]), "b_after": _pct(evasion["cnn"]["after"])}
    items = "".join(f"<li><strong>{lead}</strong> {rest.format(**values)}</li>" for lead, rest in LIMITS)
    return ('<section id="pud-limits" class="pud-card" aria-labelledby="pud-limits-title">'
            f'<h2 id="pud-limits-title">Limitations</h2><ul class="pud-warnings">{items}</ul></section>')


def defang(text: str) -> str:
    """The usual SOC defanging, so a pasted note cannot be clicked: http becomes hxxp, every dot becomes [.]."""
    return re.sub(r"(?i)h(tt)p", lambda m: m.group(0)[0] + ("xx" if m.group(1) == "tt" else "XX") + m.group(0)[3],
                  text).replace(".", "[.]")


_WRAPPERS = {"<": ">", '"': '"', "'": "'", "`": "`"}
#: `hxxp` or `hxxps` as a scheme: followed by ":" or a bracketed separator. Real URLs contain "hxxp" elsewhere (a
#: host name, random IPFS ids in the training data), so only this position is restored.
_DEFANGED_SCHEME_RE = re.compile(r"\b[hH]([xX]{2})([pP][sS]?)(?=:|\[:\]|\[://\])")
_DEFANGED_SEPARATOR_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*)(\[://\]|\[:\]//|\[:\])")
_DEFANGED_DOT_RE = re.compile(r"\[\.\]|\(\.\)|\{\.\}|\[dot\]|\(dot\)", re.IGNORECASE)


def undefang(text: object) -> tuple[str, dict[str, bool]]:
    """Restore pasted input that is defanged or wrapped, before it reaches `src/features.py`. Returns (url, changes).

    Restores, conservatively:
      - one or more surrounding pairs of <>, "", '' or backticks;
      - `hxxp`/`hxxps` (any case) as a scheme, i.e. directly followed by ":", "[:]" or "[://]": hXXps -> hTTps;
      - "[:]//", "[://]" and "[:]" right after the scheme;
      - "[.]", "(.)", "{.}", "[dot]" and "(dot)" as ".".
    The last three apply only before the first "?" or "#", the part the models score: real query strings carry
    defanged-looking text. Anything else is left exactly as given, so input with none of these is returned unchanged
    (not even stripped). It is input cleanup, not feature logic: it changes no URL in any training or test set (a test
    checks every processed dataset when it is present). Idempotent; never raises.
    """
    changes = {"unwrapped": False, "defanged": False}
    if not isinstance(text, str):
        return "", changes
    s = text
    while True:
        inner = s.strip()
        if len(inner) >= 2 and _WRAPPERS.get(inner[0]) == inner[-1]:
            s = inner[1:-1]
            changes["unwrapped"] = True
        else:
            break
    if changes["unwrapped"]:
        s = s.strip()
    cut = min((i for i in (s.find("?"), s.find("#")) if i != -1), default=len(s))
    head, tail = s[:cut].lstrip(), s[cut:]
    restored = _DEFANGED_SCHEME_RE.sub(
        lambda m: m.group(0)[0] + "".join("T" if c == "X" else "t" for c in m.group(1)) + m.group(2), head)
    restored = _DEFANGED_SEPARATOR_RE.sub(lambda m: m.group(1) + ("://" if m.group(2) != "[:]" else ":"), restored)
    restored = _DEFANGED_DOT_RE.sub(".", restored)
    if restored != head:
        changes["defanged"] = True
        s = restored + tail
    return s, changes


#: Patterns observed in the URL, shown under URL details: a label and a test on `result["address"]`. Facts, not verdicts.
OBSERVED = (
    ("The host is an IP address, not a name", lambda a: a["host_is_ip"]),
    ("The host contains punycode (xn--) or non-ASCII characters", lambda a: a["punycode"]),
    ("Text before an @ in the host part (userinfo)", lambda a: bool(a["userinfo"])),
    ("A port is named", lambda a: bool(a["port"])),
    ("The scheme is not https", lambda a: not a["is_https"]),
    ("Two or more subdomain levels in front of the registered domain", lambda a: a["subdomain_levels"] >= 2),
    ("A well-known brand name appears outside the registered domain", lambda a: a["brand_words"] > 0),
    ("A login-type word appears outside the registered domain", lambda a: a["credential_words"] > 0),
)

TRIAGE_CAVEAT = ("Caveats: lexical machine-learning scores from the URL string only. This is not threat intelligence "
                 "and not a verdict. The URL was not fetched, resolved or looked up. Risk scores are not "
                 "probabilities. Everything from the first ? or # was removed before scoring.")


def triage_note(result: dict, model_sha256: dict, checked_at: datetime | None = None) -> str:
    """A plain-text note for a ticket: defanged URL, both results, the top reasons, the models used and the caveats.

    Plain text only: it is shown in a text box and never becomes HTML. Empty for input that could not be scored.
    """
    if not result.get("ok"):
        return ""
    checked_at = checked_at or datetime.now(timezone.utc)
    a, m = result["address"], result["models"]
    lines = [
        "Phishing URL scorer: triage note (decision support, not a verdict)",
        f"Checked: {checked_at.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"URL as scored (defanged): {defang(result['scored_url'])}",
        f"Host: {defang(a['host'])}",
        f"Registered domain (approximate): {defang(a['registered_domain'])}",
    ]
    for name, title in SHORT.items():
        lines.append(f"{title} ({METHOD[name]}): risk score {_risk(m[name]['score'])} of 100, "
                     f"result {m[name]['band']}")
    lines.append("Agreement: both models gave the same result" if result["agree"] else
                 "Agreement: split result; no combined verdict")
    observed = [label for label, test in OBSERVED if test(a)]
    lines.append("Observed patterns: " + ("; ".join(observed) if observed else "none of the listed patterns"))
    lines.append("Top factors for Model A (feature = value, push on its log-odds):")
    lines += [f"  {c['contribution']:+.2f} towards {c['pushes_towards']}: {c['feature']} = {c['value']:.3g} "
              f"(feature: {c['meaning']})" for c in result["contributions"]]
    if result["notes"]:
        lines.append("Notes: " + " ".join(result["notes"]))
    lines.append(f"Models: lgbm.txt sha256 {model_sha256['lgbm'][:12]}, cnn.onnx sha256 {model_sha256['cnn'][:12]}")
    lines.append(TRIAGE_CAVEAT)
    return "\n".join(lines)


def _analyst(result: dict | None) -> str:
    """The URL broken into its parts, and patterns observed in it. Everything from the user is escaped."""
    intro = ("<p>Parsed by the same module used in training. No lookups performed: no WHOIS, DNS, hosting or "
             "reputation data.</p>")
    if result is None:
        body = IDLE
    else:
        a = result["address"]
        rows = (("Scheme", a["scheme"]), ("Host", a["host"]),
                ("Registered domain (approximate)", a["registered_domain"]), ("Top-level domain", a["tld"] or "none"),
                ("Subdomain levels", str(a["subdomain_levels"])), ("Port", a["port"] or "none"),
                ("Userinfo (before @)", a["userinfo"] or "none"), ("Path", a["url_path"]),
                ("Query string or fragment", "removed before scoring" if a["query_removed"] else "none"))
        table = "".join(f'<tr><th scope="row">{label}</th><td><code>{html.escape(value)}</code></td></tr>'
                        for label, value in rows)
        observed = [label for label, test in OBSERVED if test(a)]
        facts = ('<ul class="pud-warnings">' + "".join(f"<li>{label}.</li>" for label in observed) + "</ul>"
                 if observed else "<p>None of the listed patterns.</p>")
        body = ('<div class="pud-scroll" tabindex="0" role="region" aria-label="URL, part by part">'
                f'<table class="pud-addr"><tbody>{table}</tbody></table></div>'
                '<p class="pud-define">Registered domain is approximate: without a public-suffix list, hosting names '
                "such as github.io count as one domain.</p>"
                f"<h3>Observed patterns</h3>{facts}"
                "<p>Triage note (plain text, URL defanged): in the text box below this report.</p>")
    return ('<section id="pud-analyst" class="pud-card" aria-labelledby="pud-analyst-title">'
            f'<h2 id="pud-analyst-title">URL details</h2>{intro}{body}</section>')


def _technical(result: dict | None, rates: dict) -> str:
    parts = []
    if result is not None:
        m = result["models"]
        parts.append(
            f'<p>Calibrated scores (0 to 1): Model A {m["lgbm"]["score"]:.3f}, Model B {m["cnn"]["score"]:.3f}. '
            "Risk score is the same number times 100. Neither is a probability. Raw is the model output before "
            "calibration, in log-odds.</p>"
            f'<p>Scoring time: {result["scoring_ms"]:.1f} ms for both models on CPU; feature attribution '
            f'{result["explanation_ms"]:.1f} ms; {result["latency_ms"]:.1f} ms in total.</p>')
    parts.append(
        f'<p>Performance rates: measured on the {html.escape(rates["test_set"])} test set ({rates["rows"]:,} URLs '
        "collected after the training data). Block: 0.1% false-positive rate on validation. Review: 1% on "
        "validation, added after the test results were seen. A false positive is a legitimate URL flagged. Real "
        "traffic will differ.</p>"
        "<p>Model A: LightGBM on 31 lexical features of the URL string. Model B: character-level convolutional "
        "neural network, run with ONNX Runtime on CPU. Both trained on the PhreshPhish train split and frozen before "
        "testing. Feature contributions are LightGBM's own per-feature contributions to its raw score.</p>"
        f"<p>{PRIVACY_FULL}</p>")
    return ('<details id="pud-tech" class="pud-card pud-tech"><summary><h2>Technical details</h2></summary>'
            f'{"".join(parts)}</details>')


def _contents(result: dict | None) -> str:
    counts = {"notes": len(result["notes"]) if result else None,
              "why": len(result["contributions"]) if result else None,
              "reliable": len(MODELS), "limits": len(LIMITS)}
    links = [("pud-result", "Result", None), ("pud-notes", "Notes", counts["notes"]),
             ("pud-analyst", "URL details", None), ("pud-why", "Contributions", counts["why"]),
             ("pud-reliable", "Performance", counts["reliable"]),
             ("pud-limits", "Limitations", counts["limits"]), ("pud-tech", "Technical details", None)]
    items = "".join(
        f'<li><a href="#{anchor}"><span>{label}</span>'
        f'{f"<span class=pud-count>{count}</span>" if count is not None else ""}</a></li>'
        for anchor, label, count in links)
    return f'<nav class="pud-toc" aria-label="On this page"><p class="pud-kicker">On this page</p><ul>{items}</ul></nav>'


def _report(verdict: str, result: dict | None, rates: dict, evasion: dict | None) -> str:
    evasion = evasion if evasion is not None else get_scorer().evasion
    here = {name: m["band"] for name, m in result["models"].items()} if result else {}
    return (f'<div class="pud-wrap"><div class="pud-layout">{verdict}{_contents(result)}{_notes(result)}'
            f"{_analyst(result)}{_reasons(result)}{_reliability(here, rates)}{_limits(evasion)}"
            f"{_technical(result, rates)}</div></div>")


def render(result: dict, rates: dict, evasion: dict | None = None) -> str:
    """The result of `score_url` as the page's report. Everything that came from the user is escaped."""
    if not result.get("ok"):
        return _report(_waiting(_plain_error(str(result.get("error", ""))), alert=True), None, rates, evasion)
    return _report(_verdict(result, rates), result, rates, evasion)


def render_idle(rates: dict, evasion: dict | None = None) -> str:
    """The report before any URL has been analyzed."""
    return _report(_waiting("Enter a URL to analyze."), None, rates, evasion)


def analyse(url: str) -> str:
    """Score a URL and return the report as HTML."""
    return analyse_page(url)[0]


def analyse_page(url: str) -> tuple[str, str]:
    """Score a URL once and return the report as HTML and the plain-text triage note. Used by the interface."""
    scorer = get_scorer()
    result = scorer.score(url)
    return render(result, scorer.rates, scorer.evasion), triage_note(result, scorer.model_sha256)


# ---------------------------------------------------------------------- several URLs at once

MAX_BATCH_URLS = 50
NOT_SCORED = "Not scored"
#: An input line that could not be scored is echoed back cut to this many characters.
MAX_ECHO_CHARS = 200
BATCH_CSV_COLUMNS = ("line", "url_defanged", "result", "model_a_result", "model_a_risk_score", "model_b_result",
                     "model_b_risk_score", "same_as_line", "notes")
BATCH_NOTE_LABEL = "CSV (plain text, URLs defanged)"
TRIAGE_NOTE_LABEL = "Triage note (plain text, URL defanged)"


def score_many(text: str) -> dict:
    """Score several URL strings, one per line, with both models. No URL is fetched.

    Blank lines are skipped; at most MAX_BATCH_URLS lines. Each line is scored exactly as `score_url` scores it.
    Returns `rows` in input order: the line number and either `scored_url`, each model's band, calibrated score
    (not a probability) and raw score, `result` (the shared band, or a split; results are never combined),
    `notes` and `same_as_line` (the first line with the same URL as scored, or null), or `error` and `input`.
    `summary` counts the rows per result.
    """
    started = time.perf_counter()
    if not isinstance(text, str) or not text.strip():
        return {"ok": False, "error": "Enter one URL per line."}
    if len(text) > MAX_BATCH_URLS * (MAX_URL_CHARS + 2):
        return {"ok": False, "error": f"That input is {len(text):,} characters long. The limit is "
                                      f"{MAX_BATCH_URLS} URLs of up to {MAX_URL_CHARS:,} characters each."}
    lines = [(number, line) for number, line in enumerate(text.splitlines(), 1) if line.strip()]
    if len(lines) > MAX_BATCH_URLS:
        return {"ok": False, "error": f"{len(lines)} URLs entered. The limit is {MAX_BATCH_URLS} per batch."}
    scorer = get_scorer()
    rows, first_line = [], {}
    summary = {label: 0 for label in (*STATUS.values(), DISAGREE, NOT_SCORED)}
    for number, line in lines:
        result = scorer.score(line, explain=False)
        if not result["ok"]:
            summary[NOT_SCORED] += 1
            echo = line.strip()
            rows.append({"line": number, "ok": False, "result": NOT_SCORED, "error": result["error"],
                         "input": echo if len(echo) <= MAX_ECHO_CHARS else echo[:MAX_ECHO_CHARS] + "…"})
            continue
        models = result["models"]
        combined = STATUS[models["lgbm"]["band"]] if result["agree"] else DISAGREE
        summary[combined] += 1
        url = result["scored_url"]
        rows.append({"line": number, "ok": True, "scored_url": url, "result": combined, "agree": result["agree"],
                     "models": {name: {key: m[key] for key in ("band", "score", "raw_score")}
                                for name, m in models.items()},
                     "notes": result["notes"], "same_as_line": first_line.get(url)})
        first_line.setdefault(url, number)
    return {"ok": True, "count": len(rows), "summary": summary, "rows": rows,
            "latency_ms": (time.perf_counter() - started) * 1000.0}


def _summary_line(summary: dict) -> str:
    return " · ".join(f"{label}: {count}" for label, count in summary.items() if count)


def _batch_notes(row: dict) -> list[str]:
    """The notes for one batch row, in the page's wording."""
    if not row["ok"]:
        return [_plain_error(row["error"])]
    notes = [NOTE_PLAIN.get(note, note) for note in row["notes"]]
    if row["same_as_line"] is not None:
        notes.insert(0, f"Same URL as line {row['same_as_line']}.")
    return notes


def render_batch(batch: dict, evasion: dict | None = None) -> str:
    """The result of `score_many` as the page's report. Everything that came from the user is escaped."""
    evasion = evasion if evasion is not None else get_scorer().evasion
    if not batch.get("ok"):
        verdict = _waiting(str(batch.get("error", "")), alert=True)
    else:
        body = []
        for row in batch["rows"]:
            if row["ok"]:
                url = f'<code>{html.escape(row["scored_url"])}</code>'
                cells = "".join(
                    f'<td data-label="{SHORT[name]}">{_badge(m["band"])}'
                    f'<span class="pud-num pud-risk">{_risk(m["score"])}</span></td>' for name, m in row["models"].items())
                result = html.escape(row["result"])
            else:
                url = f'<code>{html.escape(row["input"])}</code>'
                cells, result = '<td colspan="2" data-label="Models">–</td>', NOT_SCORED
            notes = " ".join(html.escape(note) for note in _batch_notes(row))
            body.append(f'<tr><td class="pud-num" data-label="Line">{row["line"]}</td>'
                        f'<td class="pud-u" data-label="URL">{url}</td>{cells}'
                        f'<td data-label="Result">{result}</td><td data-label="Notes">{notes or "none"}</td></tr>')
        verdict = (
            '<section id="pud-result" class="pud-card pud-verdict" aria-labelledby="pud-status">'
            '<p class="pud-kicker">Batch result</p>'
            f'<h2 id="pud-status" tabindex="-1" class="pud-status"><span>{batch["count"]} '
            f'{"URL" if batch["count"] == 1 else "URLs"} analyzed (not opened)</span></h2>'
            f'<p class="pud-summary">{html.escape(_summary_line(batch["summary"]))}</p>'
            '<p class="pud-define">Each line is scored on its own, exactly as in the One URL tab. Risk score: '
            "calibrated model output, 0 to 100. Not a probability. Allow is not proof that a URL is safe.</p>"
            '<div class="pud-scroll" tabindex="0" role="region" aria-label="Results by line">'
            '<table class="pud-batch"><thead><tr><th scope="col" class="pud-num">Line</th>'
            '<th scope="col">URL analyzed</th><th scope="col">Model A (risk score)</th>'
            '<th scope="col">Model B (risk score)</th><th scope="col">Result</th><th scope="col">Notes</th>'
            f'</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
            "<p>CSV of this table (plain text, URLs defanged): in the text box below this report. For feature "
            "contributions and URL details, analyze one URL in the One URL tab.</p></section>")
    return f'<div class="pud-wrap">{verdict}{_limits(evasion)}</div>'


def _csv_cell(value: object) -> str:
    """A CSV cell that a spreadsheet will not run as a formula."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def batch_csv(batch: dict) -> str:
    """The batch result as CSV text, URLs defanged. Plain text in a text box; never written to disk."""
    if not batch.get("ok"):
        return ""
    import csv
    import io

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(BATCH_CSV_COLUMNS)
    for row in batch["rows"]:
        if row["ok"]:
            a, b = row["models"]["lgbm"], row["models"]["cnn"]
            values = (row["line"], defang(row["scored_url"]), row["result"], a["band"], _risk(a["score"]),
                      b["band"], _risk(b["score"]), row["same_as_line"], " ".join(_batch_notes(row)))
        else:
            values = (row["line"], defang(row["input"]), NOT_SCORED, "", "", "", "", "", " ".join(_batch_notes(row)))
        writer.writerow([_csv_cell(v) for v in values])
    return out.getvalue()


def analyse_many_page(text: str) -> tuple[str, str]:
    """Score several URLs once and return the report as HTML and the CSV text. Used by the interface."""
    batch = score_many(text)
    return render_batch(batch), batch_csv(batch)


HEADER = (
    '<p class="pud-kicker">URL analysis · course project, not a security product</p>'
    "<h1>Phishing URL scorer</h1>"
    "<p>Scores a URL string with two machine-learning models trained on known phishing and legitimate URLs. "
    "<strong>The URL is never opened.</strong></p>")
KEY = ("<p>Result per model:</p><ul>" + "".join(f"<li>{_badge(band)}: {ACTION[band]}</li>" for band in BANDS)
       + "</ul>")


CSS = """
.gradio-container { color-scheme: dark; overflow: clip; }
.gradio-container a:focus-visible, .gradio-container button:focus-visible, .gradio-container summary:focus-visible,
.gradio-container [tabindex]:focus-visible { outline: 3px solid var(--color-accent); outline-offset: 3px; }

#pud-main { max-width: 1280px; margin: 0 auto; padding: 40px 48px 64px; gap: 24px; }
#pud-head .pud-kicker, #pud-output .pud-kicker { margin: 0 0 8px; color: var(--body-text-color-subdued); font-family: var(--font-mono); font-size: 13px;
              line-height: 24px; letter-spacing: 0.06em; text-transform: uppercase; }
#pud-head h1 { margin: 0 0 16px; font-weight: 700; font-size: 32px; line-height: 40px; }
#pud-head p, #pud-key p, #pud-privacy p, #pud-foot p { max-width: 72ch; margin: 0; font-size: 16px; line-height: 24px; }
#pud-entry { align-items: flex-end; gap: 8px; }
#pud-url label > span { color: var(--body-text-color); font-size: 20px; line-height: 32px; font-weight: 700; }
#pud-url input, #pud-url textarea { min-height: 56px; font-family: var(--font-mono); font-size: 18px; }
#pud-score { min-height: 56px; }
#pud-privacy p { color: var(--body-text-color-subdued); }
#pud-score:hover { text-decoration: underline; }
#pud-key ul { margin: 8px 0 0; padding: 0; list-style: none; }
#pud-key li { margin: 0 0 4px; }
#pud-foot p, #pud-foot span { color: var(--body-text-color-subdued); }

#pud-output .pud-wrap { color: var(--body-text-color); font-size: 16px; line-height: 24px; }
#pud-output h2, #pud-output h3, #pud-output p, #pud-output ul, #pud-output li, #pud-output table { margin: 0; }
#pud-output .pud-layout { display: grid; grid-template-columns: minmax(0, 1fr) 232px; column-gap: 32px;
                          align-items: start; }
#pud-output .pud-layout > * { grid-column: 1; }
#pud-output .pud-toc { grid-column: 2; grid-row: 1 / span 8; position: sticky; top: 24px; padding: 16px 0 0; }
#pud-output .pud-toc ul { padding: 0; list-style: none; border-left: 1px solid var(--border-color-primary); }
#pud-output .pud-toc a { display: flex; justify-content: space-between; gap: 16px; padding: 8px 16px;
                         color: var(--body-text-color); text-decoration: none; }
#pud-output .pud-toc a:hover { color: var(--color-accent); text-decoration: underline; }
#pud-output .pud-count { min-width: 32px; padding: 0 8px; border: 1px solid var(--input-border-color);
                         font-family: var(--font-mono); text-align: center; }
#pud-output .pud-card { margin: 0 0 24px; padding: 24px; background: var(--background-fill-secondary);
                        border: 1px solid var(--border-color-primary); }
#pud-output .pud-card > * + * { margin-top: 16px; }
#pud-output h2 { font-weight: 700; font-size: 20px; line-height: 28px; }
#pud-output h3 { font-size: 18px; line-height: 24px; font-weight: 700; }
#pud-output .pud-muted, #pud-output .pud-caveat, #pud-output .pud-define,
#pud-output .pud-h3-sub { color: var(--body-text-color-subdued); }
#pud-output .pud-h3-sub { font-weight: 400; font-size: 14px; }

#pud-output #pud-status:focus { outline: none; }
#pud-output .pud-status { display: flex; gap: 16px; align-items: stretch; font-size: 28px;
                          line-height: 36px; }
#pud-output .pud-res { display: inline-flex; gap: 8px; align-items: baseline; }
#pud-output .pud-bar { flex: 0 0 auto; width: 6px; background: currentColor; }
#pud-output .pud-status .pud-bar { width: 8px; }
.pud-s-Block { color: #F19A8F; }
.pud-s-Review { color: #E6C07A; }
.pud-s-Allow { color: #8FCF9F; }
#pud-output .pud-s-split, #pud-output .pud-s-split span { color: var(--body-text-color); }
#pud-output .pud-s-split .pud-bar { background: var(--input-border-color); }
.pud-res > span:last-child { color: var(--body-text-color); }
.pud-res strong { color: inherit; }
#pud-output .pud-res strong, #pud-key .pud-res strong { color: inherit; }
#pud-output .pud-s-Block strong, #pud-key .pud-s-Block strong { color: #F19A8F; }
#pud-output .pud-s-Review strong, #pud-key .pud-s-Review strong { color: #E6C07A; }
#pud-output .pud-s-Allow strong, #pud-key .pud-s-Allow strong { color: #8FCF9F; }
#pud-key .pud-res { display: inline-flex; gap: 8px; align-items: baseline; margin: 0; }
#pud-output .pud-res .pud-bar, #pud-key .pud-res .pud-bar { align-self: stretch; }  /* baseline from the word */
#pud-key .pud-bar { flex: 0 0 auto; width: 6px; background: currentColor; }
#pud-output .pud-summary { font-size: 20px; line-height: 32px; }
#pud-output .pud-summary + .pud-define { margin-top: 8px; }
#pud-output .pud-notproof { margin-top: 8px; font-size: 20px; line-height: 32px; font-weight: 700; }
#pud-output .pud-next { padding: 16px; border-left: 4px solid var(--color-accent); background: var(--body-background-fill); }
#pud-output .pud-micro { color: var(--body-text-color-subdued); font-family: var(--font-mono); font-size: 13px;
                         letter-spacing: 0.06em; text-transform: uppercase; }
#pud-output .pud-micro + .pud-url { margin-top: 8px; }
#pud-output .pud-url { padding: 8px 16px; border: 1px solid var(--border-color-primary);
                       background: var(--body-background-fill); font-family: var(--font-mono);
                       overflow-wrap: anywhere; user-select: text; }
#pud-output .pud-wait { font-size: 20px; line-height: 32px; }
#pud-output .pud-error { padding: 16px; border: 1px solid var(--input-border-color);
                         border-left: 8px solid var(--body-text-color); font-weight: 700; }
                        background: var(--body-background-fill); border: 1px solid var(--border-color-primary); }
#pud-output .pud-trust { padding: 16px; border-left: 4px solid var(--input-border-color);
                         background: var(--body-background-fill); }
#pud-output .pud-next + .pud-trust { margin-top: 8px; }
#pud-output a { color: var(--color-accent); text-underline-offset: 4px; }

#pud-output .pud-warnings { padding: 0; list-style: none; }
#pud-output .pud-warnings li { padding: 8px 16px; border-left: 4px solid #E6C07A; background: var(--body-background-fill); }
#pud-output .pud-warnings li + li { margin-top: 8px; }
#pud-output .pud-reasons { padding: 0; list-style: none; }
#pud-output .pud-reasons li + li { margin-top: 8px; }
#pud-output h3 + .pud-reasons, #pud-output h3 + p { margin-top: 8px; }
#pud-output details summary { cursor: pointer; }
#pud-output .pud-reasons details { padding: 8px 16px; background: var(--body-background-fill);
                                   border: 1px solid var(--border-color-primary); }
#pud-output .pud-reasons details p { margin-top: 8px; color: var(--body-text-color-subdued); }
#pud-output summary::marker { color: var(--color-accent); }
#pud-output .pud-tech summary h2 { display: inline; }
#pud-output .pud-tech[open] > * + * { margin-top: 16px; }

#pud-output .pud-rely-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; }
#pud-output .pud-rely { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
#pud-output .pud-scroll { overflow-x: auto; }
#pud-output table { width: 100%; border-collapse: collapse; }
#pud-output table, #pud-output tr, #pud-output th, #pud-output td { border: 0; }
#pud-output th, #pud-output td { padding: 8px; border-bottom: 1px solid var(--border-color-primary); text-align: left;
                                 vertical-align: top; font-weight: 400; }
#pud-output thead th { color: var(--body-text-color-subdued); font-weight: 700; }
#pud-output .pud-tech table { min-width: 560px; }
#pud-output .pud-detect, #pud-output .pud-contrib { min-width: 560px; }
#pud-output .pud-detect tbody th { font-weight: 700; }
#pud-output .pud-scorecell { display: flex; align-items: center; gap: 12px; }
#pud-output .pud-scorecell .pud-num { min-width: 3em; }
#pud-output .pud-meter { display: block; flex: 1 1 auto; min-width: 80px; max-width: 200px; height: 8px;
                         background: var(--body-background-fill); border: 1px solid var(--input-border-color); }
#pud-output .pud-meter > span { display: block; height: 100%; background: var(--color-accent); }
#pud-output .pud-addr th { width: 16em; color: var(--body-text-color-subdued); }
#pud-output .pud-addr code { overflow-wrap: anywhere; }
#pud-note textarea { font-family: var(--font-mono); font-size: 14px; line-height: 20px; }
#pud-entry-many { align-items: flex-end; gap: 8px; }
#pud-urls label > span { color: var(--body-text-color); font-size: 20px; line-height: 32px; font-weight: 700; }
#pud-urls textarea { font-family: var(--font-mono); font-size: 16px; line-height: 24px; }
#pud-score-many { min-height: 56px; }
#pud-score-many:hover { text-decoration: underline; }
#pud-tabs { gap: 16px; }
#pud-tabs .tabitem { padding: 16px 0 0; border: 0; }
#pud-output .pud-batch { min-width: 880px; }
#pud-output .pud-batch td.pud-u { min-width: 30ch; max-width: 56ch; }
#pud-output .pud-batch td.pud-u code { overflow-wrap: anywhere; }
#pud-output .pud-batch .pud-res { margin-right: 8px; }
#pud-output .pud-batch .pud-risk { display: inline-block; min-width: 3em; }
#pud-output .pud-exact { display: block; color: var(--body-text-color-subdued); font-family: var(--font-mono);
                         font-size: 13px; line-height: 20px; }
#pud-output .pud-here > th, #pud-output .pud-here > td { background: var(--body-background-fill); }
#pud-output .pud-tag { display: inline-block; margin-top: 4px; padding: 0 8px; border: 1px solid var(--input-border-color);
                       font-family: var(--font-mono); font-size: 13px; }
#pud-output .pud-num { font-family: var(--font-mono); font-variant-numeric: tabular-nums; text-align: right;
                       white-space: nowrap; }
#pud-output code { padding: 0; background: none; color: inherit; font-family: var(--font-mono); }

@media (max-width: 1080px) {
  #pud-main { padding: 32px 24px 48px; }
  #pud-output .pud-layout { grid-template-columns: minmax(0, 1fr); }
  #pud-output .pud-toc { grid-column: 1; grid-row: auto; position: static; margin: 0 0 24px; padding: 0; }
  #pud-output .pud-toc ul { display: flex; flex-wrap: wrap; border-left: 0; }
  #pud-output .pud-toc a { padding: 8px 16px 8px 0; }
  #pud-output .pud-rely-grid { grid-template-columns: minmax(0, 1fr); }
}
@media (max-width: 760px) {
  #pud-main { padding: 24px 0 48px; }
  #pud-hero { margin-top: min(12vh, 120px); padding: 16px; }
  #pud-output .pud-card { padding: 16px; }
  #pud-output .pud-addr th, #pud-output .pud-addr td { display: block; width: auto; }
  #pud-output .pud-addr th { padding-bottom: 0; border-bottom: 0; }
  /* on a phone the detection table keeps model, result and risk score; method and raw score are on wider screens */
  #pud-output .pud-detect { min-width: 0; }
  #pud-output .pud-detect tr > :nth-child(2), #pud-output .pud-detect tr > :nth-child(5) { display: none; }
  #pud-output .pud-detect tbody th { white-space: nowrap; }
  #pud-output .pud-meter { min-width: 32px; }
  /* on a phone each batch row becomes a labelled block instead of a wide table */
  #pud-output .pud-batch { min-width: 0; }
  #pud-output .pud-batch thead { display: none; }
  #pud-output .pud-batch tr { display: block; padding: 8px 0; border-bottom: 1px solid var(--border-color-primary); }
  #pud-output .pud-batch td { display: block; padding: 2px 0; border: 0; text-align: left; }
  #pud-output .pud-batch td.pud-u { min-width: 0; max-width: none; }
  #pud-output .pud-batch td::before { content: attr(data-label) ": "; color: var(--body-text-color-subdued); }
}
@media (max-width: 480px) {
  #pud-entry, #pud-entry-many { flex-direction: column; align-items: stretch; }
}

/* Decorative animated background (background.js): one fixed layer behind the page, never clickable. The page's
   own wrappers are made transparent so it shows; the report cards keep their solid panels. */
#pud-bg { position: fixed; top: 0; left: 0; right: 0; height: min(100vh, 1100px); z-index: -1; pointer-events: none; }
.gradient-ascii-container { overflow: hidden; }
.gradient-ascii-canvas { position: absolute; inset: 0; display: block; width: 100%; height: 100%; }
gradio-app, gradio-app > .main, .gradio-container, .gradio-container .main, .gradio-container .wrap,
.gradio-container .contain, #pud-main, #pud-main > *, #pud-entry, #pud-head, #pud-privacy, #pud-key, #pud-foot,
#pud-output, #pud-note { background: transparent !important; }
/* the search area sits on a translucent panel over the darker part of the background */
#pud-hero { margin-top: min(22vh, 240px); padding: 32px; gap: 24px; background: rgba(36, 38, 41, 0.82) !important;
            border: 1px solid var(--border-color-primary); backdrop-filter: blur(6px); }
#pud-hero > *, #pud-url *:not(input):not(textarea), #pud-urls *:not(input):not(textarea), #pud-tabs .tabitem,
#pud-tabs .tab-container, #pud-entry-many { background: transparent !important; }
#pud-key, #pud-foot { text-shadow: 0 1px 3px #000, 0 0 12px #000; }
#pud-note textarea { background: var(--background-fill-secondary) !important; }
"""

# Gradio styles every element inside an HTML block with the body text colour (`.prose *`), so the status
# colours are set on each part that shows them: the word, its bar, and the status heading.
CSS += "".join(
    f"#pud-output .pud-s-{band} strong, #pud-key .pud-s-{band} strong, #pud-output .pud-status.pud-s-{band} span "
    f"{{ color: {colour}; }}\n#pud-output .pud-s-{band} .pud-bar, #pud-key .pud-s-{band} .pud-bar "
    f"{{ background: {colour}; }}\n" for band, colour in STATUS_COLOUR.items())


def build_theme():
    """The page's colours and type as a Gradio theme. Light and dark mode get the same values."""
    import gradio as gr

    colours = {
        "body_background_fill": CANVAS, "background_fill_primary": CANVAS, "background_fill_secondary": PANEL,
        "body_text_color": TEXT, "body_text_color_subdued": MUTED, "border_color_primary": LINE,
        "border_color_accent": ACCENT, "color_accent_soft": PANEL, "code_background_fill": PANEL,
        "block_background_fill": CANVAS, "block_border_color": LINE, "block_label_background_fill": CANVAS,
        "block_label_text_color": MUTED, "block_title_text_color": MUTED,
        "input_background_fill": PANEL, "input_border_color": EDGE, "input_border_color_focus": ACCENT,
        "input_placeholder_color": MUTED, "input_shadow_focus": f"0 0 0 2px {ACCENT}",
        "button_primary_background_fill": ACCENT, "button_primary_background_fill_hover": ACCENT,
        "button_primary_text_color": INK, "button_primary_border_color": ACCENT,
        "button_primary_border_color_hover": TEXT,
        "button_secondary_background_fill": PANEL, "button_secondary_background_fill_hover": PANEL,
        "button_secondary_text_color": TEXT, "button_secondary_border_color": EDGE,
        "button_secondary_border_color_hover": ACCENT,
        "link_text_color": ACCENT, "link_text_color_hover": ACCENT, "link_text_color_visited": ACCENT,
        "link_text_color_active": ACCENT,
    }
    both = {**colours, **{f"{name}_dark": value for name, value in colours.items()}}
    theme = gr.themes.Base(text_size=gr.themes.sizes.text_lg, radius_size=gr.themes.sizes.radius_none,
                           font=["Trebuchet MS", "Open Sans", "sans-serif"],
                           font_mono=["Consolas", "Liberation Mono", "monospace"])
    return theme.set(
        **both, color_accent=ACCENT, block_border_width="0px", block_padding="0px", block_title_text_size="*text_md",
        input_text_size="*text_md", input_padding="8px 16px", layout_gap="24px", button_border_width="1px",
        button_transition="none", button_large_padding="8px 24px", button_large_text_size="*text_md",
        button_large_text_weight="700", button_small_padding="8px 16px", button_small_text_size="*text_md")


def build_demo():
    import gradio as gr

    scorer = get_scorer()
    idle = render_idle(scorer.rates, scorer.evasion)
    with gr.Blocks(title="Phishing URL scorer: URL string analysis", analytics_enabled=False,
                   fill_width=True) as demo:
        with gr.Column(elem_id="pud-main"):
            with gr.Column(elem_id="pud-hero"):
                gr.HTML(HEADER, elem_id="pud-head")
                with gr.Tabs(elem_id="pud-tabs"):
                    with gr.Tab("One URL", elem_id="pud-tab-one"):
                        with gr.Row(elem_id="pud-entry"):
                            box = gr.Textbox(label="URL", placeholder="https://example.com/path", lines=1,
                                             max_lines=1, scale=5, autofocus=True, elem_id="pud-url")
                            button = gr.Button("Analyze", variant="primary", scale=0, min_width=176,
                                               elem_id="pud-score")
                    with gr.Tab("Several URLs", elem_id="pud-tab-many"):
                        with gr.Row(elem_id="pud-entry-many"):
                            many = gr.Textbox(label=f"URLs, one per line (up to {MAX_BATCH_URLS})",
                                              placeholder="https://example.com/a\nhxxps://example[.]net/b", lines=6,
                                              max_lines=20, scale=5, elem_id="pud-urls")
                            many_button = gr.Button("Analyze all", variant="primary", scale=0, min_width=176,
                                                    elem_id="pud-score-many")
                gr.HTML(f"<p>{PRIVACY_SHORT} Defanged input (hxxp, [.]) is accepted.</p>", elem_id="pud-privacy")
            # Before the first analysis the page is only the URL box; the report appears below it afterwards.
            key = gr.HTML(KEY, elem_id="pud-key", visible=False)
            result = gr.HTML(idle, elem_id="pud-output", visible=False)
            # Plain text, so nothing in it can become a link or markup; the copy button is Gradio's own.
            note = gr.Textbox(label=TRIAGE_NOTE_LABEL, value="", lines=8,
                              max_lines=30, interactive=False, buttons=["copy"], elem_id="pud-note",
                              placeholder="Shown after analysis.", visible=False)
            gr.HTML("<p>Graduate course project in machine learning for cybersecurity. Models trained on the "
                    "PhreshPhish train split and frozen before testing. Not a security product.</p>", elem_id="pud-foot")

        # After each check the result is brought to the top of the window and the status heading takes the
        # keyboard focus, so the verdict is the first thing seen or heard and Tab continues into the report.
        outputs = [key, result, note]

        def show(url):
            page, text = analyse_page(url)
            return (gr.update(visible=True), gr.update(value=page, visible=True),
                    gr.update(value=text, visible=bool(text), label=TRIAGE_NOTE_LABEL))

        # Both tabs write to the same report area; the text box holds the triage note or the batch CSV.
        def show_many(text):
            page, table = analyse_many_page(text)
            return (gr.update(visible=True), gr.update(value=page, visible=True),
                    gr.update(value=table, visible=bool(table), label=BATCH_NOTE_LABEL))

        button.click(show, box, outputs, api_visibility="private").then(None, js=SHOW_RESULT)
        box.submit(show, box, outputs, api_visibility="private").then(None, js=SHOW_RESULT)
        many_button.click(show_many, many, outputs, api_visibility="private").then(None, js=SHOW_RESULT)
        gr.api(analyse, api_name="analyse")
        gr.api(score_url, api_name="score")
        gr.api(score_many, api_name="score_batch")

        # A link of the form /?url=... fills the box and scores it. Nothing is fetched either way.
        def from_link(request):
            url = request.query_params.get("url", "") if request else ""
            return (url, *show(url)) if url else (url, gr.update(visible=False), gr.update(value=idle, visible=False),
                                                  gr.update(value="", visible=False))

        # Gradio passes the request only to a parameter annotated with the class itself. This file
        # postpones annotations (they would be strings), so the annotation is set by hand.
        from_link.__annotations__ = {"request": gr.Request}
        demo.load(from_link, None, [box, *outputs], api_visibility="private").then(None, js=SHOW_RESULT)
    return demo


#: Runs in the browser after each check: brings the result into view and focuses its status heading. It does
#: nothing while the report is waiting for a first link. A fixed script; nothing in it comes from the user.
SHOW_RESULT = """() => {
  const status = document.getElementById('pud-status');
  if (!status || !document.querySelector('#pud-result .pud-summary, #pud-result .pud-error')) return;
  document.getElementById('pud-result').scrollIntoView({block: 'start'});
  status.focus({preventScroll: true});
}"""

#: The decorative background (background.js, beside this file), put in the page's <head> by launch().
HEAD = f"<script>{(_HERE / 'background.js').read_text(encoding='utf-8')}</script>"

#: Passed to launch(). Gradio's run history would keep every checked link in the visitor's browser.
LAUNCH_OPTIONS = {"show_error": False, "ssr_mode": False, "footer_links": ["api"], "run_history": False}


if __name__ == "__main__":
    get_scorer()                                   # load the models before the first request
    build_demo().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        theme=build_theme(), css=CSS, head=HEAD, **LAUNCH_OPTIONS,
    )
