"""Tests for src/evaluate.py and src/mutate.py on invented data."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src import evaluate as E
from src import features as F
from src import mutate


def _scores(n=4000, seed=0, ties=False):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    raw = rng.normal(loc=np.where(y == 1, 1.5, -1.5), scale=1.5)
    if ties:
        raw = np.round(raw)
    return y, raw


OPERATING = {"thresholds": {"block": {"raw_score": 2.0}, "warn": {"raw_score": -0.5}}}


# ------------------------------------------------------------------ metrics agree with scikit-learn

@pytest.mark.parametrize("ties", [False, True])
def test_point_metrics_match_scikit_learn(ties):
    y, raw = _scores(ties=ties)
    m = E.evaluate_scores(y, raw, OPERATING)["metrics"]
    assert m["roc_auc"]["value"] == pytest.approx(roc_auc_score(y, raw), abs=1e-12)
    assert m["pr_auc"]["value"] == pytest.approx(average_precision_score(y, raw), abs=1e-12)
    fpr, tpr, _ = roc_curve(y, raw, drop_intermediate=False)
    assert m["tpr_at_fpr_0.001"]["value"] == pytest.approx(tpr[fpr <= 0.001].max())
    assert m["fpr_at_tpr_0.95"]["value"] == pytest.approx(fpr[tpr >= 0.95].min())
    for name, t in (("block", 2.0), ("warn", -0.5)):
        flagged = raw >= t
        tp, fp, fn = (flagged & (y == 1)).sum(), (flagged & (y == 0)).sum(), (~flagged & (y == 1)).sum()
        assert m[f"{name}_recall"]["value"] == pytest.approx(tp / (tp + fn))
        assert m[f"{name}_fpr"]["value"] == pytest.approx(fp / (y == 0).sum())
        assert m[f"{name}_precision"]["value"] == pytest.approx(tp / (tp + fp))
        assert m[f"{name}_f1"]["value"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
        recall, rate = tp / (tp + fn), fp / (y == 0).sum()
        assert m[f"{name}_precision_at_base_rate_0.01"]["value"] == pytest.approx(
            0.01 * recall / (0.01 * recall + 0.99 * rate))


def test_weighted_statistics_equal_the_statistics_of_the_resampled_rows():
    y, raw = _scores(n=800, seed=1, ties=True)
    weights = E.bootstrap_weights(len(y), n_boot=3, seed=5)
    curve = E.Curve(y, raw, {"block": 2.0, "warn": -0.5})
    for w in weights:
        repeated = np.repeat(np.arange(len(y)), w)
        got = curve.stats(w)
        assert got["roc_auc"] == pytest.approx(roc_auc_score(y[repeated], raw[repeated]), abs=1e-12)
        assert got["pr_auc"] == pytest.approx(average_precision_score(y[repeated], raw[repeated]), abs=1e-12)
        flagged = raw[repeated] >= -0.5
        assert got["warn_recall"] == pytest.approx(flagged[y[repeated] == 1].mean())
        assert got["warn_fpr"] == pytest.approx(flagged[y[repeated] == 0].mean())


def test_bootstrap_weights_are_resamples_and_reproducible():
    w = E.bootstrap_weights(500, n_boot=20, seed=3)
    assert w.shape == (20, 500) and (w.sum(axis=1) == 500).all()
    assert np.array_equal(w, E.bootstrap_weights(500, n_boot=20, seed=3))
    assert not np.array_equal(w[0], w[1])


def test_intervals_contain_the_point_estimate_and_narrow_with_more_data():
    y, raw = _scores(n=6000, seed=2)
    big = E.evaluate_scores(y, raw, OPERATING, E.bootstrap_weights(len(y), 200))["metrics"]["roc_auc"]
    small = E.evaluate_scores(y[:500], raw[:500], OPERATING, E.bootstrap_weights(500, 200))["metrics"]["roc_auc"]
    assert big["ci_low"] < big["value"] < big["ci_high"]
    assert (big["ci_high"] - big["ci_low"]) < (small["ci_high"] - small["ci_low"])


def test_masks_restrict_rows_and_one_class_groups_do_not_crash():
    y, raw = _scores(n=1000, seed=3)
    weights = E.bootstrap_weights(len(y), 50)
    mask = np.arange(len(y)) < 300
    sub = E.evaluate_scores(y, raw, OPERATING, weights, mask)
    assert sub["rows"] == 300 and sub["phishing"] == int(y[:300].sum())
    assert sub["metrics"]["roc_auc"]["value"] == pytest.approx(roc_auc_score(y[:300], raw[:300]))
    only_phishing = E.evaluate_scores(y, raw, OPERATING, weights, y == 1)
    assert only_phishing["metrics"]["roc_auc"]["value"] is None
    assert only_phishing["metrics"]["warn_recall"]["value"] == pytest.approx((raw[y == 1] >= -0.5).mean())
    assert only_phishing["metrics"]["warn_fpr"]["value"] is None
    assert E.evaluate_scores(y, raw, OPERATING, weights, np.zeros(len(y), bool))["rows"] == 0


def test_confusion_counts_every_row_once():
    y, raw = _scores(n=1000, seed=4)
    confusion = E.evaluate_scores(y, raw, OPERATING)["confusion"]
    assert sum(confusion["legitimate"].values()) == (y == 0).sum()
    assert confusion["phishing"]["Block"] == ((raw >= 2.0) & (y == 1)).sum()
    assert confusion["legitimate"]["Warn"] == ((raw >= -0.5) & (raw < 2.0) & (y == 0)).sum()


# ------------------------------------------------------------------ false positives by domain

def test_fpr_by_domain_separates_per_url_from_per_domain():
    # One big domain where everything is flagged, nine small ones where nothing is.
    y = np.zeros(190, dtype=int)
    raw = np.r_[np.full(100, 5.0), np.full(90, -5.0)]
    domains = ["big.example"] * 100 + [f"small{i}.example" for i in range(9) for _ in range(10)]
    out = E.fpr_by_domain(y, raw, domains, OPERATING, n_boot=300)
    assert out["registered_domains"] == 10 and out["legitimate_rows"] == 190
    assert out["block"]["per_url"]["value"] == pytest.approx(100 / 190)
    assert out["block"]["per_domain"]["value"] == pytest.approx(0.1)
    assert out["block"]["domains_with_any_false_positive"] == 1
    assert out["block"]["share_of_false_positives_in_top_10_domains"] == 1.0
    # Resampling domains gives a wide interval: the result hinges on one domain.
    assert out["block"]["per_url"]["ci_low"] == 0.0 and out["block"]["per_url"]["ci_high"] > 0.7


def test_fpr_by_domain_ignores_phishing_rows():
    y = np.r_[np.zeros(50, int), np.ones(50, int)]
    raw = np.r_[np.full(50, -5.0), np.full(50, 5.0)]
    out = E.fpr_by_domain(y, raw, [f"d{i % 5}.example" for i in range(100)], OPERATING, n_boot=50)
    assert out["legitimate_rows"] == 50 and out["warn"]["per_url"]["value"] == 0.0


# ------------------------------------------------------------------ template rule, groups, formatting

def test_template_rule_scores_only_the_homepage_template_as_legitimate():
    urls = [F.normalize_url(u, strip_query=True) for u in (
        "https://www.example.com", "https://www.example.com/", "http://www.example.com/",
        "https://example.com/", "https://www.example.com/a")]
    assert E.template_rule_raw(urls).tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_add_groups():
    frame = pd.DataFrame({
        "url": ["https://www.seen.example/", "https://a.unseen.example/robots.txt", "http://x.seen.example/p"],
        "had_query": [False, True, False],
    })
    groups = E.add_groups(frame, {"seen.example"})
    assert groups["bare_homepage"].tolist() == [True, False, False]
    assert groups["domain_seen"].tolist() == [True, False, True]
    assert groups["had_query"].tolist() == [False, True, False]
    assert groups["resource_kind"].tolist() == ["page", "robots.txt", "page"]
    assert E.registered_domains(frame) == {"seen.example", "unseen.example"}


def test_fmt():
    assert E.fmt({"value": 0.9312, "ci_low": 0.927, "ci_high": 0.9351}) == "93.1% (92.7-93.5)"
    assert E.fmt({"value": 0.98123, "ci_low": 0.98, "ci_high": 0.982}, percent=False, digits=3) == "0.981 (0.980-0.982)"
    assert E.fmt({"value": None, "ci_low": None, "ci_high": None}) == "n/a" and E.fmt(None) == "n/a"
    assert E.fmt({"value": 0.5, "ci_low": None, "ci_high": None}) == "50.0%"


# ------------------------------------------------------------------ mutation

def test_mutation_keeps_the_shape_of_the_url():
    rng = np.random.default_rng(0)
    url = "https://secure-login.account-check.example.com/verify/session/index.html"
    for strength in (1, 3, 8):
        for _ in range(200):
            out = mutate.mutate_url(url, strength, rng)
            before, after = F.split_url(url), F.split_url(out)
            assert after.scheme == "https"
            assert after.host.endswith(".com") and after.host.count(".") == before.host.count(".")
            assert after.path.count("/") == before.path.count("/")
            assert F.normalize_url(out, strip_query=True) == out


def test_mutation_strength_zero_is_the_identity_and_more_edits_change_more():
    urls = [f"http://login-update{i}.example.net/account/{i}/verify" for i in range(300)]
    assert mutate.mutate_many(urls, 0) == [F.normalize_url(u, strip_query=True) for u in urls]
    changed = [np.mean([a != b for a, b in zip(urls, mutate.mutate_many(urls, k))]) for k in (1, 3, 8)]
    assert changed[0] > 0.5 and changed[0] <= changed[1] <= changed[2] and changed[2] > 0.99
    assert mutate.mutate_many(urls, 3, seed=1) == mutate.mutate_many(urls, 3, seed=1)
    assert mutate.mutate_many(urls, 3, seed=1) != mutate.mutate_many(urls, 3, seed=2)


@pytest.mark.parametrize("url", ["http://192.0.2.10/", "https://example.com/", "https://x.co/", "", "https://[::1]/a"])
def test_mutation_handles_urls_with_little_or_nothing_to_edit(url):
    rng = np.random.default_rng(1)
    for strength in (1, 5):
        out = mutate.mutate_url(url, strength, rng)
        assert isinstance(out, str)
    assert F.split_url(mutate.mutate_url("http://192.0.2.10/", 5, rng)).host == "192.0.2.10"
