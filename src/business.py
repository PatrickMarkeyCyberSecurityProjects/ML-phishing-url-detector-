"""A worked example of what the measured rates would mean for a stream of links.

This is arithmetic on measured rates, not a measurement. Every scenario states
its assumptions, and the output (reports/business_case.json) carries them with
it. The briefing quotes that file.

    python -m src.business

What is measured: the share of phishing and of legitimate URLs each band holds
on a test set (reports/metrics.json, key post_hoc), and the effect of three
ordinary-looking edits on phishing URLs.

What is assumed: the number of links in a month, the share of them that are
phishing, and that the links resemble the test set named in each scenario.
None of those is known for any real organization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from src import evaluate as E
from src import models as M

MONTHLY_LINKS = 1_000_000
PHISHING_RATES = (0.001, 0.01)           # the two base rates already used in reports/results.md
MODELS = {"lgbm": "Model A: LightGBM", "cnn": "Model B: character CNN"}
TEST_SETS = {
    "phreshphish_eval": "links resemble the later-in-time PhreshPhish test (same collector as the training data)",
    "wangchuk": "links resemble the Wangchuk dataset (a different collector; presumed-benign class of 547 domains)",
    "ariyadasa": "links resemble the Ariyadasa dataset (a different collector; collected 2020-21)",
}


def scenario(entry: Mapping[str, object], monthly_links: int, phishing_rate: float) -> dict:
    """Counts per month in each band, from the band shares measured on one test set."""
    phishing, legitimate = monthly_links * phishing_rate, monthly_links * (1 - phishing_rate)
    p = {band: entry["bands"]["phishing"][band] / entry["phishing"] for band in ("Block", "Review", "Allow")}
    q = {band: entry["bands"]["legitimate"][band] / entry["legitimate"] for band in ("Block", "Review", "Allow")}
    blocked = phishing * p["Block"] + legitimate * q["Block"]
    reviewed = phishing * p["Review"] + legitimate * q["Review"]
    return {
        "phishing_links": round(phishing), "legitimate_links": round(legitimate),
        "phishing_blocked": round(phishing * p["Block"]), "legitimate_blocked": round(legitimate * q["Block"]),
        "phishing_sent_to_review": round(phishing * p["Review"]),
        "legitimate_sent_to_review": round(legitimate * q["Review"]),
        "phishing_allowed_missed": round(phishing * p["Allow"]),
        "share_of_blocked_links_that_are_phishing": phishing * p["Block"] / blocked if blocked else None,
        "share_of_reviewed_links_that_are_phishing": phishing * p["Review"] / reviewed if reviewed else None,
        "share_of_all_links_blocked_or_reviewed": (blocked + reviewed) / monthly_links,
        "share_of_phishing_blocked_or_reviewed": p["Block"] + p["Review"],
        "links_a_slower_check_would_see": round(reviewed),
        "links_a_slower_check_would_not_need_to_see": round(monthly_links - reviewed),
    }


def business_case(metrics: Mapping[str, object], monthly_links: int = MONTHLY_LINKS) -> dict:
    post = metrics["post_hoc"]
    results = post["review"]["results"]
    out = {
        "what_this_is": "Arithmetic on measured rates. Not a measurement of any real deployment.",
        "assumptions": {
            "monthly_links": monthly_links,
            "phishing_rates": list(PHISHING_RATES),
            "phishing_rates_note": "illustrative; the two base rates already used in reports/results.md. "
                                   "The real rate for any organization is not known here.",
            "bands": "Block = frozen 0.1% threshold; Review = the 1% threshold added after the test results were seen",
            "each_scenario_assumes": "the organization's links resemble the named test set",
            "no_adaptive_attacker": "the scenarios assume phishing links are not edited to pass; see under_evasion",
            "not_estimated": "breaches prevented or money saved: nothing in this project measures either",
        },
        "scenarios": {},
        "under_evasion": {},
    }
    for key, assumption in TEST_SETS.items():
        out["scenarios"][key] = {"assumes": assumption, "models": {}}
        for name, label in MODELS.items():
            out["scenarios"][key]["models"][name] = {
                "label": label,
                **{f"phishing_rate_{rate:g}": scenario(results[key][name]["all"], monthly_links, rate)
                   for rate in PHISHING_RATES},
            }
    targeted = post["targeted_robustness"]["models"]
    for name, label in MODELS.items():
        out["under_evasion"][name] = {
            "label": label,
            "assumes": "every phishing link carries the three ordinary-looking edits of the targeted test "
                       "(www. prefix, https, a benign-looking path); measured on the later-in-time test",
            "share_of_phishing_blocked": targeted[name]["block"]["after"]["all"]["value"],
            "share_of_phishing_blocked_or_reviewed": targeted[name]["review"]["after"]["all"]["value"],
            "share_of_phishing_allowed": 1 - targeted[name]["review"]["after"]["all"]["value"],
        }
    return out


def write_business_case(metrics_path: Path | None = None, out_path: Path | None = None) -> Path:
    metrics = M.load_json(metrics_path or E.REPORTS_DIR / "metrics.json")
    return M.save_json(business_case(metrics), Path(out_path or E.REPORTS_DIR / "business_case.json"))


if __name__ == "__main__":
    path = write_business_case()
    case = M.load_json(path)
    for key, entry in case["scenarios"].items():
        for name, model in entry["models"].items():
            for rate in PHISHING_RATES:
                s = model[f"phishing_rate_{rate:g}"]
                print(f"{key:17s} {name:5s} rate {rate:<6g} blocked {s['phishing_blocked']:>6,} phishing + "
                      f"{s['legitimate_blocked']:>7,} legitimate | review {s['phishing_sent_to_review']:>6,} + "
                      f"{s['legitimate_sent_to_review']:>7,} | missed {s['phishing_allowed_missed']:>6,} | "
                      f"reviewed-or-blocked share of links {s['share_of_all_links_blocked_or_reviewed']:.2%}")
    print("saved", path)
