"""Tests for src/posthoc.py and the targeted edits in src/mutate.py, on invented data."""

import numpy as np
import pandas as pd
import pytest

from src import evaluate as E
from src import features as F
from src import models as M
from src import mutate
from src import posthoc as P


def _scores(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    raw = rng.normal(loc=np.where(y == 1, 2.0, -2.0), scale=2.0)
    return y, raw


CALIBRATION = {"x": [-10.0, 10.0], "y": [0.0, 1.0]}


# ------------------------------------------------------------------ targeted edits

def test_www_edit_prefixes_the_host_once_and_leaves_ip_hosts_alone():
    assert mutate.targeted_edit("http://login.example.com/a", "www") == "http://www.login.example.com/a"
    assert mutate.targeted_edit("http://www.example.com/a", "www") == "http://www.example.com/a"
    assert mutate.targeted_edit("http://192.0.2.7/a", "www") == "http://192.0.2.7/a"
    assert mutate.targeted_edit("http://user@example.com:8080/a", "www") == "http://user@www.example.com:8080/a"


def test_https_edit_changes_only_the_scheme():
    assert mutate.targeted_edit("http://example.com/a/b", "https") == "https://example.com/a/b"
    assert mutate.targeted_edit("https://example.com/a/b", "https") == "https://example.com/a/b"


@pytest.mark.parametrize("url, expected", [
    ("http://example.com", "http://example.com/about-us"),
    ("http://example.com/", "http://example.com/about-us"),
    ("http://example.com/login/", "http://example.com/login/about-us"),
    ("http://example.com/login/index.php", "http://example.com/login/index.php/about-us"),
    ("http://example.com/login?next=1#top", "http://example.com/login/about-us"),
])
def test_path_edit_appends_without_a_double_slash(url, expected):
    out = mutate.targeted_edit(url, "path", "/about-us")
    assert out == expected
    assert F.extract_features(out, strip_query=True)["has_double_slash"] == 0


def test_all_edit_is_the_three_edits_together_and_the_result_is_normalized():
    out = mutate.targeted_edit("http://Secure-Login.example.net/verify/", "all", "/en/support/contact")
    assert out == "https://www.secure-login.example.net/verify/en/support/contact"
    assert F.normalize_url(out, strip_query=True) == out
    with pytest.raises(ValueError):
        mutate.targeted_edit("http://example.com/", "shorten")
    assert mutate.targeted_edit("", "all") == ""


def test_targeted_many_is_reproducible_and_path_and_all_share_one_draw():
    urls = [f"http://host{i}.example.org/p{i}" for i in range(400)]
    assert mutate.targeted_many(urls, "path") == mutate.targeted_many(urls, "path")
    draw = mutate.benign_path_draw(len(urls))
    assert set(draw.tolist()) == set(range(len(mutate.BENIGN_PATHS)))
    for url, i, path_only, everything in zip(urls, draw, mutate.targeted_many(urls, "path"),
                                             mutate.targeted_many(urls, "all")):
        assert path_only.endswith(mutate.BENIGN_PATHS[i]) and everything.endswith(mutate.BENIGN_PATHS[i])
        assert everything.startswith("https://www.")


def test_the_registered_edits_have_not_been_changed():
    # docs/design_decisions.md, decision 23, fixed these before the test was run.
    assert list(mutate.TARGETED_EDITS) == ["www", "https", "path", "all"]
    assert mutate.BENIGN_PATHS == ("/about-us", "/en/support/contact", "/blog/2025/03/company-news",
                                   "/products/category/item-details.html",
                                   "/news/articles/annual-report-2024.html")


# ------------------------------------------------------------------ review point

def test_review_point_is_the_block_rule_at_one_percent():
    y, raw = _scores()
    point = P.review_point(raw, y, CALIBRATION)
    flagged = raw >= point["raw_score"]
    fpr = flagged[y == 0].mean()
    assert fpr <= 0.01 and point["validation_fpr"] == pytest.approx(fpr)
    assert point["validation_tpr"] == pytest.approx(flagged[y == 1].mean())
    assert point["target_max_fpr"] == 0.01
    # the next lower score would break the target
    lower = np.sort(raw[raw < point["raw_score"]])[-1]
    assert (raw >= lower)[y == 0].mean() > 0.01 or (raw >= lower)[y == 0].mean() == fpr
    block = M.choose_thresholds(raw, y)["block"]["raw_score"]
    assert point["raw_score"] < block
    assert point["calibrated_score"] == pytest.approx((point["raw_score"] + 10) / 20)


def test_frozen_thresholds_reproduced_detects_a_mismatch():
    y, raw = _scores(seed=1)
    frozen = M.choose_thresholds(raw, y)
    assert P.frozen_thresholds_reproduced(raw, y, frozen)["reproduced"] is True
    shifted = {k: {"raw_score": frozen[k]["raw_score"] + 0.5} for k in ("block", "warn")}
    assert P.frozen_thresholds_reproduced(raw, y, shifted)["reproduced"] is False


def test_bands():
    assert P.bands([-1.0, 1.0, 2.0, 3.0, 5.0], block=3.0, review=1.0).tolist() == [
        "Allow", "Review", "Review", "Block", "Block"]


def test_evaluate_points_matches_direct_counts_and_the_main_block_numbers():
    y, raw = _scores(seed=2)
    weights = E.bootstrap_weights(len(y), 100)
    thresholds = {"block": 3.0, "review": 1.0}
    got = P.evaluate_points(y, raw, thresholds, weights)
    for name, t in thresholds.items():
        flagged = raw >= t
        assert got["metrics"][f"{name}_recall"]["value"] == pytest.approx(flagged[y == 1].mean())
        assert got["metrics"][f"{name}_fpr"]["value"] == pytest.approx(flagged[y == 0].mean())
        m = got["metrics"][f"{name}_recall"]
        assert m["ci_low"] <= m["value"] <= m["ci_high"]
    assert not any(k.startswith(("roc", "pr_", "warn")) for k in got["metrics"])
    # same code and resamples as the main evaluation: block metrics are identical
    main = E.evaluate_scores(y, raw, {"thresholds": {"block": {"raw_score": 3.0}, "warn": {"raw_score": -1.0}}},
                               weights)["metrics"]
    for key in ("block_recall", "block_fpr", "block_precision", "block_f1"):
        assert got["metrics"][key] == main[key]
    # every row lands in exactly one band
    assert sum(got["bands"]["legitimate"].values()) == (y == 0).sum()
    assert got["bands"]["phishing"]["Review"] == ((raw >= 1.0) & (raw < 3.0) & (y == 1)).sum()
    assert got["bands"]["phishing"]["Block"] + got["bands"]["phishing"]["Review"] == ((raw >= 1.0) & (y == 1)).sum()
    empty = P.evaluate_points(y, raw, thresholds, weights, np.zeros(len(y), bool))
    assert empty["rows"] == 0 and empty["metrics"] == {}


def test_evaluate_review_covers_both_models_splits_and_domains():
    y, raw = _scores(n=1500, seed=3)
    df = pd.DataFrame({"label": y})
    groups = pd.DataFrame({
        "bare_homepage": np.arange(len(y)) % 2 == 0, "domain_seen": np.arange(len(y)) % 3 == 0,
        "had_query": np.arange(len(y)) % 5 == 0, "registered_domain": [f"d{i % 40}.example" for i in range(len(y))]})
    thresholds = {m: {"block": 3.0, "review": 1.0} for m in P.MODELS}
    out = P.evaluate_review(df, {"lgbm": raw, "cnn": raw + 0.1}, groups, thresholds, domain_fpr=True)
    assert set(out) == set(P.MODELS)
    for entry in out.values():
        assert set(entry["splits"]) == set(E.SPLITS)
        for halves in entry["splits"].values():
            assert sum(h["rows"] for h in halves.values()) == entry["all"]["rows"]
        by_domain = entry["false_positive_rate_by_domain"]
        assert set(by_domain) >= {"block", "review"} and "warn" not in by_domain
        assert by_domain["review"]["per_url"]["value"] == pytest.approx(entry["all"]["metrics"]["review_fpr"]["value"])


# ------------------------------------------------------------------ targeted robustness

def _fake_scorers():
    # A scorer that likes http and dislikes www.: the edits must lower its score.
    def raw(df):
        return np.array([3.0 - 2.5 * u.startswith("https://") - 2.5 * ("//www." in u) - 0.01 * len(u)
                         for u in df["url"]])
    return {name: {"raw": raw, "label": name} for name in P.MODELS}


def test_targeted_robustness_reports_before_after_and_paired_change():
    urls = [f"http://login{i}.example.net/verify" for i in range(300)] + \
           [f"https://www.shop{i}.example.com/" for i in range(100)]
    frame = pd.DataFrame({"url": [F.normalize_url(u, strip_query=True) for u in urls]})
    thresholds = {m: {"block": 2.0, "review": 0.0} for m in P.MODELS}
    out = P.targeted_robustness(frame, _fake_scorers(), thresholds, n_boot=200)
    assert out["phishing_urls"] == 400 and out["benign_paths"] == list(mutate.BENIGN_PATHS)
    assert out["urls_changed"] == {"www": 300, "https": 300, "path": 400, "all": 400}
    block = out["models"]["lgbm"]["block"]
    assert block["before"]["value"] == pytest.approx(0.75)
    assert block["after"]["https"]["value"] == pytest.approx(0.0)
    assert block["after"]["https"]["change"]["value"] == pytest.approx(-0.75)
    change = block["after"]["https"]["change"]
    assert change["ci_low"] <= change["value"] <= change["ci_high"] < 0
    review = out["models"]["cnn"]["review"]
    assert review["before"]["value"] == pytest.approx(0.75) and review["after"]["all"]["value"] == 0.0
    per_path = out["by_benign_path"]["lgbm"]["block"]
    assert [p["path"] for p in per_path] == list(mutate.BENIGN_PATHS) and sum(p["urls"] for p in per_path) == 400


def test_plot_targeted_writes_a_png(tmp_path):
    frame = pd.DataFrame({"url": [f"http://login{i}.example.net/verify" for i in range(60)]})
    out = P.targeted_robustness(frame, _fake_scorers(), {m: {"block": 2.0, "review": 0.0} for m in P.MODELS}, n_boot=50)
    P.plot_targeted(out, tmp_path / "chart.png")
    assert (tmp_path / "chart.png").stat().st_size > 10_000


def test_warn_on_validation_reads_the_saved_metrics():
    metric = {"value": 0.66, "ci_low": 0.65, "ci_high": 0.67}
    saved = {
        "results": {"phreshphish_val": {m: {"all": {"legitimate": 1000, "metrics": {"warn_recall": metric, "warn_fpr": metric}}}
                                        for m in P.MODELS}},
        "scorers": {m: {"thresholds": {"warn": {"target_min_tpr": 0.95, "validation_false_positives": 660}}}
                    for m in P.MODELS},
    }
    out = P.warn_on_validation(saved)
    assert out["cnn"]["warn_fpr"] == metric and out["lgbm"]["legitimate_validation_urls_flagged"] == 660


# ------------------------------------------------------------------ decision 24: the CNN tuning check

def _fake_check(shipped_score: float) -> dict:
    runs = [{"learning_rate": lr, "dropout": d, "shipped_settings": (lr, d) == (0.001, 0.3),
             "validation_pr_auc": score}
            for (lr, d), score in zip([(0.0003, 0.3), (0.001, 0.3), (0.003, 0.5)], [0.80, shipped_score, 0.84])]
    for rank, run in enumerate(sorted(runs, key=lambda r: r["validation_pr_auc"], reverse=True), start=1):
        run["rank"] = rank
    best = max(runs, key=lambda r: r["validation_pr_auc"])
    shipped = next(r for r in runs if r["shipped_settings"])
    gap = best["validation_pr_auc"] - shipped["validation_pr_auc"]
    return {"runs": runs, "best": {k: best[k] for k in ("learning_rate", "dropout", "validation_pr_auc")},
            "shipped": {"learning_rate": 0.001, "dropout": 0.3, "rank": shipped["rank"],
                        "validation_pr_auc_retrained": shipped["validation_pr_auc"]},
            "shipped_among_best": gap <= 0.005}


def test_cnn_tuning_sentence_follows_the_result_either_way():
    from src import report
    won = report.cnn_tuning_sentence(_fake_check(0.85))
    assert "ranked first of 3" in won and "next best, learning rate 0.003 with dropout 0.5, reached 0.8400" in won
    assert "so they were among the best" in won
    lost = report.cnn_tuning_sentence(_fake_check(0.81))
    assert "ranked 2 of 3" in lost and "against a best of 0.8400" in lost and "were not among the best" in lost
    for sentence in (won, lost):
        assert "Post hoc" in sentence and "no test set was scored" in sentence and "kept either way" in sentence


def test_saved_cnn_tuning_check_is_the_planned_grid_and_scores_no_test_set():
    from src import posthoc as P
    if not P.CNN_CHECK_FILE.exists():
        pytest.skip("run the last section of notebooks/03_cnn.ipynb first")
    check = M.load_json(P.CNN_CHECK_FILE)
    assert check["post_hoc"] is True and "No test set was scored" in check["what"]
    grid = {(r["learning_rate"], r["dropout"]) for r in check["runs"]}
    assert grid == {(lr, d) for lr in P.CNN_CHECK_LEARNING_RATES for d in P.CNN_CHECK_DROPOUTS}
    assert (M.CNN_TRAINING["learning_rate"], M.CNN_CONFIG["dropout"]) in grid          # the shipped settings
    assert sorted(r["rank"] for r in check["runs"]) == list(range(1, len(grid) + 1))
    assert set(check["runs"][0]) == {"learning_rate", "dropout", "shipped_settings", "epochs_run", "best_epoch",
                                      "validation_pr_auc", "rank"}                      # validation PR-AUC only
    # the retrained shipped settings reproduce the frozen model's validation PR-AUC
    saved = M.load_json(M.MODELS_DIR / "cnn_operating.json")["training"]["validation_pr_auc"]
    assert check["shipped"]["validation_pr_auc_saved"] == saved
    assert abs(check["shipped"]["validation_pr_auc_retrained"] - saved) < 1e-3
    assert check["shipped"]["rank"] == 1
