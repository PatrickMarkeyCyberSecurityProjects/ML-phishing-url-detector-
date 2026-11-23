"""The written results must be exactly what src/report.py produces from the saved metrics."""

import pytest

from src import evaluate as E
from src import models as M
from src import report

METRICS = E.REPORTS_DIR / "metrics.json"
RESULTS = E.REPORTS_DIR / "results.md"
needs_metrics = pytest.mark.skipif(not METRICS.exists(), reason="run notebooks/04_evaluation.ipynb first")


@needs_metrics
def test_results_md_is_in_step_with_metrics_json():
    assert RESULTS.read_text(encoding="utf-8") == report.render_results_md(M.load_json(METRICS))


@needs_metrics
def test_results_md_opens_with_what_the_models_cannot_be_trusted_to_do():
    text = RESULTS.read_text(encoding="utf-8")
    first_heading = next(line for line in text.splitlines() if line.startswith("## "))
    assert first_heading == "## What these models cannot be trusted to do"
    assert text.index("## What these models cannot be trusted to do") < text.index("## Headline table")
    for figure in ("04_model_comparison.png", "04_roc_pr_curves.png", "04_confusion_matrices.png",
                   "04_shap_summary.png", "04_robustness.png"):
        assert f"figures/{figure}" in text and (E.FIGURES_DIR / figure).exists()


@needs_metrics
def test_metrics_cover_every_scorer_test_set_split_and_required_metric():
    metrics = M.load_json(METRICS)
    required = {"roc_auc", "pr_auc", "tpr_at_fpr_0.001", "fpr_at_tpr_0.95", "block_recall", "block_precision",
                "block_f1", "block_fpr", "warn_recall", "warn_precision", "warn_f1", "warn_fpr",
                "block_precision_at_base_rate_0.01", "block_precision_at_base_rate_0.001",
                "warn_precision_at_base_rate_0.01", "warn_precision_at_base_rate_0.001"}
    for dataset in ("phreshphish_eval", "wangchuk", "ariyadasa"):
        for scorer in E.SCORERS:
            entry = metrics["results"][dataset][scorer]
            assert required <= set(entry["all"]["metrics"]), (dataset, scorer)
            for name in ("roc_auc", "block_fpr", "warn_recall"):
                m = entry["all"]["metrics"][name]
                assert m["ci_low"] <= m["value"] <= m["ci_high"], (dataset, scorer, name)
            assert set(entry["splits"]) == {"url_shape", "registered_domain", "original_query"}
            for groups in entry["splits"].values():
                assert sum(g["rows"] for g in groups.values()) == entry["all"]["rows"]
        assert metrics["datasets"][dataset]["rows"] == metrics["results"][dataset]["lgbm"]["all"]["rows"]
    for dataset in ("wangchuk", "ariyadasa"):
        assert "false_positive_rate_by_domain" in metrics["results"][dataset]["cnn"]
    assert "sensitivity_pages_only" in metrics["results"]["wangchuk"]["lgbm"]
    assert metrics["robustness"]["strengths"][0] == 0 and len(metrics["robustness"]["models"]) == 2
    assert metrics["datasets"]["phreshphish_eval"]["rows"] == 50_000


def test_post_hoc_section_renders_from_invented_numbers_and_is_labelled_post_hoc():
    """The post hoc writer, on invented scores: no test data is touched here."""
    import numpy as np
    import pandas as pd

    from src import posthoc as P

    rng = np.random.default_rng(0)
    n = 1200
    y = (rng.random(n) < 0.4).astype(int)
    raw = rng.normal(loc=np.where(y == 1, 2.0, -2.0), scale=2.0)
    df = pd.DataFrame({"label": y, "url": [f"http://host{i}.example.net/p/{i}" for i in range(n)]})
    groups = pd.DataFrame({"bare_homepage": np.arange(n) % 2 == 0, "domain_seen": np.arange(n) % 3 == 0,
                           "had_query": np.arange(n) % 5 == 0,
                           "registered_domain": [f"d{i % 30}.example" for i in range(n)]})
    thresholds = {m: {"block": 3.0, "review": 1.0} for m in P.MODELS}
    scores = {"lgbm": raw, "cnn": raw + 0.2}
    full = P.evaluate_review(df, scores, groups, thresholds, domain_fpr=True)
    scorers = {m: {"raw": (lambda frame: np.array([3.5 - 2.0 * u.startswith("https://") for u in frame["url"]])),
                   "label": m} for m in P.MODELS}
    metric = {"value": 0.66, "ci_low": 0.65, "ci_high": 0.67}
    metrics = {
        "results": {"phreshphish_eval": {m: {"all": {"metrics": {"warn_fpr": metric}}} for m in P.MODELS}},
        "post_hoc": {
            "decided": P.DECIDED,
            "warn_on_validation": {m: {"target_min_detection": 0.95, "warn_detection": metric, "warn_fpr": metric,
                                       "legitimate_validation_urls": 1000,
                                       "legitimate_validation_urls_flagged": 660} for m in P.MODELS},
            "review": {"target_max_fpr": 0.01,
                       "results": {key: full for key in ("phreshphish_val", "phreshphish_eval", "wangchuk", "ariyadasa")}},
            "targeted_robustness": P.targeted_robustness(df[y == 1], scorers, thresholds, n_boot=50),
        },
    }
    text = "\n".join(report.post_hoc_section(metrics))
    assert text.startswith(f"## {report.POST_HOC_HEADING}") and "**Post hoc.**" in text
    assert "already unusable before any test set was scored" in text
    assert "Edits chosen to look legitimate lower detection" in text
    assert "figures/05_targeted_robustness.png" in text and "nan" not in text.lower()


@needs_metrics
def test_post_hoc_additions_leave_the_main_results_alone():
    metrics = M.load_json(METRICS)
    if "post_hoc" not in metrics:
        pytest.skip("run notebooks/05_post_hoc.ipynb first")
    text = RESULTS.read_text(encoding="utf-8")
    assert text.index("## Headline table") < text.index(f"## {report.POST_HOC_HEADING}")
    # the main part of results.md is what it would be without the post hoc key, apart from the labelled pointers
    main = report.render_results_md({k: v for k, v in metrics.items() if k != "post_hoc"})

    def table(md: str, heading: str) -> list[str]:
        section = md.split(f"## {heading}")[1].split("\n## ")[0]
        return [line for line in section.splitlines() if line.startswith("|")]

    for heading in ("Headline table", "The benchmark trap", "Credibility splits", "Robustness to random edits"):
        assert table(text, heading) == table(main, heading), heading
    post = metrics["post_hoc"]
    assert "after the test results had been seen" in post["decided"]
    assert post["review"]["target_max_fpr"] == 0.01
    for model in ("lgbm", "cnn"):
        frozen = metrics["scorers"][model]["thresholds"]
        point = post["review"]["thresholds"][model]
        assert frozen["warn"]["raw_score"] < point["raw_score"] < frozen["block"]["raw_score"]
        assert point["validation_fpr"] <= 0.01
        for key in ("phreshphish_eval", "wangchuk", "ariyadasa"):
            new = post["review"]["results"][key][model]["all"]
            old = metrics["results"][key][model]["all"]
            assert new["metrics"]["block_recall"]["value"] == pytest.approx(old["metrics"]["block_recall"]["value"], abs=1e-12)
            assert new["metrics"]["block_fpr"]["value"] == pytest.approx(old["metrics"]["block_fpr"]["value"], abs=1e-12)
            assert sum(new["bands"]["phishing"].values()) == old["phishing"]
            assert new["bands"]["phishing"]["Block"] == old["confusion"]["phishing"]["Block"]
        before = post["targeted_robustness"]["models"][model]["block"]["before"]["value"]
        assert before == pytest.approx(metrics["results"]["phreshphish_eval"][model]["all"]["metrics"]["block_recall"]["value"], abs=1e-12)
    assert (E.FIGURES_DIR / "05_targeted_robustness.png").exists()
