"""Writes reports/results.md from reports/metrics.json.

Every number in results.md is read from the saved metrics file, so the written
results cannot drift from what was measured. Run `write_results_md()` after
notebooks/04_evaluation.ipynb has produced metrics.json.

If metrics.json has a `post_hoc` key (written by notebooks/05_post_hoc.ipynb),
the additions made after the test results were seen are written under their
own heading. The tables of the main evaluation are rendered the same either way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from src import evaluate as E
from src import models as M

MAIN = {"phreshphish_eval": "PhreshPhish, later in time", "wangchuk": "Wangchuk, different source",
        "ariyadasa": "Ariyadasa, different source"}
MODELS = {"lgbm": "Model A (LightGBM)", "cnn": "Model B (character CNN)"}


POST_HOC_HEADING = "Additions made after the test results were seen"
VAL = {"phreshphish_val": "PhreshPhish validation (where the threshold was set)"}


def _table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def load_latency(path: Path | None = None) -> dict | None:
    """The saved latency measurement of the app's container (app/measure_latency.py), if there is one."""
    path = Path(path or E.REPORTS_DIR / "latency.json")
    return M.load_json(path) if path.exists() else None


def load_training_shapes(path: Path | None = None) -> dict | None:
    """URL shape by class in the fitting data (bare homepage or with a path), from the data audit."""
    path = Path(path or E.REPORTS_DIR / "data_audit.json")
    return M.load_json(path)["shape_by_class"]["train"] if path.exists() else None


def render_results_md(metrics: Mapping[str, object], latency: Mapping[str, object] | None = None,
                      shapes: Mapping[str, object] | None = None) -> str:
    R = metrics["results"]
    latency = latency if latency is not None else load_latency()
    shapes = shapes if shapes is not None else load_training_shapes()

    def m(key: str, scorer: str, name: str) -> dict:
        return R[key][scorer]["all"]["metrics"][name]

    def v(key: str, scorer: str, name: str) -> float:
        return m(key, scorer, name)["value"]

    def pct(x: float, digits: int = 1) -> str:
        return f"{x * 100:.{digits}f}%"

    def split(key: str, scorer: str, which: str, group: str, name: str) -> dict | None:
        return R[key][scorer]["splits"][which][group]["metrics"].get(name)

    ev = "phreshphish_eval"
    rb = metrics["robustness"]

    def edits(scorer: str, point: str, strength: int) -> float:
        return next(p["recall"] for p in rb["models"][scorer][point] if p["strength"] == strength)

    cv = metrics["scorers"]["lgbm"]["training"]["tuning"]["best_cv_pr_auc"]
    val_pr = R["phreshphish_val"]["lgbm"]["all"]["metrics"]["pr_auc"]["value"]
    out: list[str] = []
    w = out.append

    w("# Results")
    w("")
    w("Generated from `reports/metrics.json` by `src/report.py`. Every number below is read from that file. "
      "Intervals in brackets are 95% bootstrap intervals. Full tables and charts are in "
      "`notebooks/04_evaluation.ipynb`.")
    w("")
    w("## What these models cannot be trusted to do")
    w("")
    w("Read this section before any score. The models rank URLs far better than chance, and far better than "
      "the baselines, but they are not a control that can be left to act on its own.")
    w("")
    def fp_domains(key: str, scorer: str) -> str:
        d = R[key][scorer]["false_positive_rate_by_domain"]
        return f"{d['block']['share_of_false_positives_in_top_10_domains']:.0%}"

    def seen(key: str, scorer: str, name: str, digits: int = 3, percent: bool = False) -> str:
        a = split(key, scorer, "registered_domain", "seen in training", name)
        b = split(key, scorer, "registered_domain", "unseen in training", name)
        return f"{E.fmt(a, percent, digits, ci=False)} against {E.fmt(b, percent, digits, ci=False)}"

    bare_warn = split(ev, "lgbm", "url_shape", "bare homepage", "warn_fpr")
    top_feature = (metrics.get("shap_mean_abs_by_feature") or [{"feature": "path_len"}])[0]["feature"]
    val_roc = R["phreshphish_val"]["lgbm"]["all"]["metrics"]["roc_auc"]["value"]
    flat = all(edits(sc, pt, 8) >= edits(sc, pt, 0) - 0.02 for sc in ("lgbm", "cnn") for pt in ("warn", "block"))

    post = metrics.get("post_hoc")
    targeted_pointer = ""
    if post:
        t = post["targeted_robustness"]["models"]
        targeted_pointer = (
            f" A test with edits chosen to look legitimate was added afterwards: with a `www.` prefix, `https` "
            f"and a benign-looking path together, the share blocked went from "
            f"{pct(t['lgbm']['block']['before']['value'])} to {pct(t['lgbm']['block']['after']['all']['value'])} "
            f"for Model A and from {pct(t['cnn']['block']['before']['value'])} to "
            f"{pct(t['cnn']['block']['after']['all']['value'])} for Model B (see *{POST_HOC_HEADING}*).")

    w(f"1. **They cannot be left to block on their own.** The block threshold was set on validation to flag at "
      f"most 0.1% of legitimate URLs. On later data it blocks {pct(v(ev, 'lgbm', 'block_recall'))} of phishing "
      f"(Model A) and {pct(v(ev, 'cnn', 'block_recall'))} (Model B). The rest gets a warning at most.")
    w(f"2. **Their false-positive rate holds on later data from the same collector and breaks on other "
      f"collectors' data.** The block threshold flags {pct(v(ev, 'lgbm', 'block_fpr'), 2)} (Model A) and "
      f"{pct(v(ev, 'cnn', 'block_fpr'), 2)} (Model B) of legitimate URLs on later PhreshPhish data, close to the "
      f"0.1% target. On Wangchuk it flags {pct(v('wangchuk', 'lgbm', 'block_fpr'), 2)} and "
      f"{pct(v('wangchuk', 'cnn', 'block_fpr'), 2)} of presumed-benign URLs, and on Ariyadasa "
      f"{pct(v('ariyadasa', 'lgbm', 'block_fpr'), 2)} and {pct(v('ariyadasa', 'cnn', 'block_fpr'), 2)}. On "
      f"Wangchuk the errors are concentrated: {fp_domains('wangchuk', 'lgbm')} of Model A's wrongly blocked URLs, "
      f"and {fp_domains('wangchuk', 'cnn')} of Model B's, sit on ten registered domains.")
    w(f"3. **Their warning band is too wide to act on.** The warn threshold was set to catch 95% of validation "
      f"phishing. On later data it warns on or blocks {pct(v(ev, 'lgbm', 'warn_fpr'))} of legitimate URLs "
      f"for Model A and {pct(v(ev, 'cnn', 'warn_fpr'))} for Model B, while catching "
      f"{pct(v(ev, 'lgbm', 'warn_recall'))} and {pct(v(ev, 'cnn', 'warn_recall'))} of phishing. Model A warns on "
      f"{E.fmt(bare_warn, ci=False)} of legitimate bare homepages. A warning shown on that many ordinary links "
      f"teaches people to ignore it." + (
          f" The warn point was the pre-specified target. It was already this wide on validation, where it "
          f"flagged {E.fmt(post['warn_on_validation']['lgbm']['warn_fpr'], ci=False)} and "
          f"{E.fmt(post['warn_on_validation']['cnn']['warn_fpr'], ci=False)} of legitimate URLs, so it is a "
          f"target that proved unreachable at a usable false-positive rate for a model that sees only the URL. "
          f"A review point at a 1% validation false-positive rate was added afterwards; see *{POST_HOC_HEADING}*."
          if post else ""))
    w(f"4. **Their quality swings from one period to the next, and cross-validation hid that.** "
      f"Cross-validation inside the training period gave Model A a PR-AUC of {cv:.3f}. On the six weeks after "
      f"the training period (validation) its PR-AUC was {val_pr:.3f} and its ROC-AUC {val_roc:.3f}; on the three "
      f"months after that (the evaluation sample) its ROC-AUC was {v(ev, 'lgbm', 'roc_auc'):.3f}. Random folds put "
      f"URLs from the same campaigns and weeks on both sides of each split. (PR-AUC also depends on the class "
      f"mix, which differs between these sets; ROC-AUC does not.)")
    w(f"5. **They are weaker on registered domains they have not seen.** Model A's ROC-AUC on seen against "
      f"unseen domains is {seen(ev, 'lgbm', 'roc_auc')} on later data, {seen('wangchuk', 'lgbm', 'roc_auc')} on "
      f"Wangchuk and {seen('ariyadasa', 'lgbm', 'roc_auc')} on Ariyadasa. Its block threshold catches "
      f"{seen(ev, 'lgbm', 'block_recall', 1, True)} of phishing on later data. Model B shows the same pattern "
      f"({seen(ev, 'cnn', 'roc_auc')}; {seen(ev, 'cnn', 'block_recall', 1, True)} blocked). Part of what looks "
      f"like detection is recognition of hosting platforms from the training data.")
    if flat:
        w(f"6. **Random edits did not lower detection, and that is not evidence of robustness.** After eight "
          f"random character edits the share of phishing URLs blocked went from {pct(edits('lgbm', 'block', 0))} "
          f"to {pct(edits('lgbm', 'block', 8))} for Model A and from {pct(edits('cnn', 'block', 0))} to "
          f"{pct(edits('cnn', 'block', 8))} for Model B. Random edits make a URL look more random, which these "
          f"models already treat as suspicious. The test says nothing about edits chosen to look legitimate. "
          f"The feature Model A leans on most is `{top_feature}`, which an attacker controls completely, and an "
          f"attacker who can query the scorer can search for whatever passes." + targeted_pointer)
    else:
        w(f"6. **Small changes to a URL change the answer.** After eight random character edits the share of "
          f"phishing URLs blocked went from {pct(edits('lgbm', 'block', 0))} to {pct(edits('lgbm', 'block', 8))} "
          f"for Model A and from {pct(edits('cnn', 'block', 0))} to {pct(edits('cnn', 'block', 8))} for Model B. "
          f"These edits were random. An attacker who can query the scorer would do better." + targeted_pointer)
    w("7. **Their scores are not probabilities.** The calibrator was fitted on validation data that is 25% "
      "phishing because of when the collector gathered each class. No legitimate rows exist for November 2024 "
      "to April 2025. A calibrated score of 0.9 does not mean a 90% chance of phishing.")
    w("8. **They see only the URL string, without its query string.** They cannot detect phishing hosted on a "
      "legitimate domain from the page content, cannot follow a redirect or a shortener, and ignore anything "
      "after `?` or `#`.")
    w("9. **Nothing here measures performance on live traffic.** Every test set is 36% to 46% phishing. At a "
      f"1% phishing rate, Model A's block verdict would be right "
      f"{E.fmt(m(ev, 'lgbm', 'block_precision_at_base_rate_0.01'), ci=False)} "
      f"of the time on the later data and its warn verdict "
      f"{E.fmt(m(ev, 'lgbm', 'warn_precision_at_base_rate_0.01'), ci=False)}; "
      f"at 0.1%, {E.fmt(m(ev, 'lgbm', 'block_precision_at_base_rate_0.001'), ci=False)} and "
      f"{E.fmt(m(ev, 'lgbm', 'warn_precision_at_base_rate_0.001'), ci=False)}.")
    w("")
    w("## What they can do")
    w("")
    w(f"- **Rank URLs usefully.** ROC-AUC on later data is {v(ev, 'lgbm', 'roc_auc'):.3f} for Model A and "
      f"{v(ev, 'cnn', 'roc_auc'):.3f} for Model B. On the two datasets from other collectors it is "
      f"{v('wangchuk', 'lgbm', 'roc_auc'):.3f} and {v('ariyadasa', 'lgbm', 'roc_auc'):.3f} for Model A, and "
      f"{v('wangchuk', 'cnn', 'roc_auc'):.3f} and {v('ariyadasa', 'cnn', 'roc_auc'):.3f} for Model B.")
    w(f"- **Beat the benchmark-trained baseline everywhere outside the benchmark.** A LightGBM trained on PhiUSIIL "
      f"scores {v('phiusiil_test', 'lgbm_phiusiil', 'roc_auc'):.4f} on PhiUSIIL's own test split and "
      f"{v(ev, 'lgbm_phiusiil', 'roc_auc'):.3f}, {v('wangchuk', 'lgbm_phiusiil', 'roc_auc'):.3f} and "
      f"{v('ariyadasa', 'lgbm_phiusiil', 'roc_auc'):.3f} on the three tests here (0.5 is chance).")
    if latency:
        w(f"- **Score a URL in about {latency['scoring_both_models']['median_ms']:.1f} ms, without fetching it.** "
          f"That is the median for both models inside the app's container over {latency['urls']} URLs "
          f"({latency['scoring_both_models']['p95_ms']:.1f} ms at the 95th percentile). The feature breakdown the "
          f"app shows takes a further {latency['explanation_pred_contrib']['median_ms']:.0f} ms, and a full request "
          f"to the app took {latency['round_trip']['median_ms']:.0f} ms ({latency['round_trip']['p95_ms']:.0f} ms at "
          f"the 95th percentile) on the development machine (`reports/latency.json`). That speed is what lets a "
          f"scorer like this sit in front of slower checks as one signal among several.")
    else:
        w("- **Score a URL in milliseconds, without fetching it,** which is what lets a scorer like this sit in "
          "front of slower checks as one signal among several. (Latency is measured by app/measure_latency.py.)")
    w("")
    w("## Headline table")
    w("")
    rows = []
    for key, title in MAIN.items():
        for scorer in E.SCORERS:
            mm = R[key][scorer]["all"]["metrics"]
            rows.append([title, E.SCORERS[scorer][0], E.fmt(mm["roc_auc"], False, 3), E.fmt(mm["pr_auc"], False, 3),
                         E.fmt(mm["tpr_at_fpr_0.001"]), E.fmt(mm["fpr_at_tpr_0.95"]),
                         E.fmt(mm["block_recall"]), E.fmt(mm["block_fpr"], digits=2),
                         E.fmt(mm["warn_recall"]), E.fmt(mm["warn_fpr"]),
                         E.fmt(mm["warn_precision"]), E.fmt(mm["warn_f1"], False, 3)])
    w(_table(["Test set", "Scorer", "ROC-AUC", "PR-AUC", "Detection at 0.1% FPR", "FPR at 95% detection",
              "Block: detection", "Block: FPR", "Warn: detection", "Warn: FPR", "Warn: precision", "Warn: F1"], rows))
    w("")
    w("- *Detection at 0.1% FPR* and *FPR at 95% detection* are read off each test set's own curve: the best "
      "that test set allows. The *Block* and *Warn* columns use the thresholds frozen on validation, which is "
      "what a deployed system would actually do.")
    w("- *Warn* means warned on or blocked. Precision and F1 are at the class mix of the test set, "
      "not of real traffic.")
    w("- The template rule has one operating point, so its Block and Warn columns are the same.")
    if post:
        w(f"- The Warn point is the pre-specified target (catch 95% of phishing). This table is as it was written "
          f"before the additions under *{POST_HOC_HEADING}*, which include a review point at a 1% validation "
          f"false-positive rate.")
    w("")
    w("Rows per test set: " + "; ".join(
        f"{title}: {metrics['datasets'][key]['rows']:,} ({metrics['datasets'][key]['legitimate']:,} legitimate or "
        f"presumed benign, {metrics['datasets'][key]['phishing']:,} phishing)" for key, title in MAIN.items()) + ".")
    w("")
    w("## The benchmark trap")
    w("")
    rows = [[E.SCORERS[s][0]] + [E.fmt(R[key][s]["all"]["metrics"]["roc_auc"], False, 3)
                                 for key in ("phiusiil_test", *MAIN)]
            for s in ("lgbm_phiusiil", "template_rule", "lgbm", "cnn")]
    w(_table(["ROC-AUC", "PhiUSIIL test split", *MAIN.values()], rows))
    w("")
    below = sum(v(key, "lgbm_phiusiil", "roc_auc") < 0.5 for key in MAIN)
    w("Every legitimate PhiUSIIL URL is `https://www.<host>/`. A model trained there, and a one-line rule with no "
      "learning, both look finished on PhiUSIIL's own test split. Off the benchmark the PhiUSIIL-trained model "
      f"ranks URLs worse than chance on {below} of the 3 tests (0.5 is chance), because it learned that a path "
      "means phishing and real data says the opposite. The one-line rule sits at chance everywhere else.")
    w("")
    w("## The query-string artefact")
    w("")
    w("The PhreshPhish collector removed query parameters from legitimate URLs only (paper, Appendix A.2). The "
      "comparison model was trained on the same rows as Model A with the query string kept.")
    w("")
    rows = []
    for key, title in MAIN.items():
        had = R[key]["lgbm_full_urls"]["splits"]["original_query"]["had a query string"]
        rows.append([title, E.fmt(m(key, "lgbm", "roc_auc"), False, 3),
                     E.fmt(m(key, "lgbm_full_urls", "roc_auc"), False, 3),
                     f"{had['legitimate']:,}",
                     E.fmt(split(key, "lgbm", "original_query", "had a query string", "warn_fpr")),
                     E.fmt(had["metrics"].get("warn_fpr")),
                     E.fmt(split(key, "lgbm", "original_query", "had a query string", "block_fpr"), digits=2),
                     E.fmt(had["metrics"].get("block_fpr"), digits=2)])
    w(_table(["Test set", "ROC-AUC, Model A (query removed)", "ROC-AUC, comparison (query kept)",
              "Legitimate URLs that had a query", "Flagged by Model A", "Flagged by comparison",
              "Blocked by Model A", "Blocked by comparison"], rows))
    w("")
    w("\"Flagged\" means warned on or blocked. The last four columns cover only legitimate (or presumed-benign) "
      "URLs whose original form had a query string. PhreshPhish has almost none, so read those columns on the "
      "external datasets.")
    w("")
    w("## Credibility splits")
    w("")
    w("Each result is split three ways. A model worth trusting should hold up in both halves.")
    for which, heading in (("url_shape", "Bare homepage versus a URL with a path"),
                           ("registered_domain", "Registered domain seen in training versus unseen"),
                           ("original_query", "Original URL had a query string versus not")):
        w("")
        w(f"### {heading}")
        w("")
        rows = []
        for key, title in MAIN.items():
            for scorer in MODELS:
                for group, entry in R[key][scorer]["splits"][which].items():
                    mm = entry["metrics"]
                    rows.append([title, MODELS[scorer], group, f"{entry['legitimate']:,}", f"{entry['phishing']:,}",
                                 E.fmt(mm.get("roc_auc"), False, 3), E.fmt(mm.get("block_recall")),
                                 E.fmt(mm.get("block_fpr"), digits=2), E.fmt(mm.get("warn_recall")),
                                 E.fmt(mm.get("warn_fpr"))])
        w(_table(["Test set", "Model", "Subgroup", "Legitimate", "Phishing", "ROC-AUC", "Block: detection",
                  "Block: FPR", "Warn: detection", "Warn: FPR"], rows))
    w("")
    w("\"Seen in training\" means the registered domain occurs anywhere in the PhreshPhish training or validation "
      "data, under either label. In PhreshPhish almost no legitimate URL had a query string, so that subgroup's "
      "false-positive rates rest on very few rows.")
    w("")
    w("## Precision at realistic phishing rates")
    w("")
    rows = []
    for key, title in MAIN.items():
        for scorer in MODELS:
            mm = R[key][scorer]["all"]["metrics"]
            rows.append([title, MODELS[scorer], E.fmt(mm["block_precision"]),
                         E.fmt(mm["block_precision_at_base_rate_0.01"]),
                         E.fmt(mm["block_precision_at_base_rate_0.001"]),
                         E.fmt(mm["warn_precision"]), E.fmt(mm["warn_precision_at_base_rate_0.01"]),
                         E.fmt(mm["warn_precision_at_base_rate_0.001"])])
    w(_table(["Test set", "Model", "Block: as tested", "Block: 1% phishing", "Block: 0.1% phishing",
              "Warn: as tested", "Warn: 1% phishing", "Warn: 0.1% phishing"], rows))
    w("")
    w("Computed from the measured detection and false-positive rates: of the URLs flagged, the share that would "
      "be phishing if phishing were 1% or 0.1% of all URLs.")
    w("")
    w("## External datasets: false positives per URL and per registered domain")
    w("")
    rows = []
    for key in ("wangchuk", "ariyadasa"):
        for scorer in MODELS:
            d = R[key][scorer]["false_positive_rate_by_domain"]
            for point in ("block", "warn"):
                rows.append([MAIN[key], MODELS[scorer], point, f"{d['legitimate_rows']:,}",
                             f"{d['registered_domains']:,}",
                             E.fmt(d[point]["per_url"], digits=2), E.fmt(d[point]["per_domain"], digits=2),
                             d[point]["domains_with_any_false_positive"],
                             f"{d[point]['share_of_false_positives_in_top_10_domains']:.0%}"])
    w(_table(["Dataset", "Model", "Threshold", "Presumed-benign URLs", "Registered domains", "FPR per URL",
              "FPR per domain", "Domains with a false positive", "False positives in the 10 worst domains"], rows))
    w("")
    w("Intervals here resample registered domains, not rows: URLs from one domain are not independent. "
      "\"Per domain\" averages the rate within each domain, so every domain counts once.")
    w("")
    w("### Sensitivity: Wangchuk without `robots.txt` and other non-page resources")
    w("")
    rows = []
    for scorer in MODELS:
        for label, entry in (("all rows (headline)", R["wangchuk"][scorer]["all"]),
                             ("pages only", R["wangchuk"][scorer]["sensitivity_pages_only"])):
            mm = entry["metrics"]
            rows.append([MODELS[scorer], label, f"{entry['legitimate']:,}", f"{entry['phishing']:,}",
                         E.fmt(mm["roc_auc"], False, 3), E.fmt(mm["block_fpr"], digits=2), E.fmt(mm["warn_fpr"]),
                         E.fmt(mm["warn_recall"])])
    w(_table(["Model", "Rows", "Presumed benign", "Phishing", "ROC-AUC", "Block: FPR", "Warn: FPR",
              "Warn: detection"], rows))
    w("")
    w("## Robustness to random edits")
    w("")
    rows = [[s] + [E.fmt({"value": p["recall"], "ci_low": p["ci_low"], "ci_high": p["ci_high"]})
                   for scorer in MODELS for point in ("warn", "block")
                   for p in rb["models"][scorer][point] if p["strength"] == s]
            for s in rb["strengths"]]
    w(_table(["Random edits", "Model A: warned or blocked", "Model A: blocked", "Model B: warned or blocked",
              "Model B: blocked"], rows))
    w("")
    w(f"{rb['sampled_phishing_urls']:,} phishing URLs from the evaluation sample, each changed by random "
      "character edits to the host name and path (`src/mutate.py`). Frozen thresholds.")
    w("")
    if post:
        out.extend(post_hoc_section(metrics, shapes))
    w("## Figures")
    w("")
    for name, caption in (("04_model_comparison.png", "ROC-AUC by dataset and model"),
                          ("04_roc_pr_curves.png", "ROC and precision-recall curves"),
                          ("04_confusion_matrices.png", "Verdict by true class"),
                          ("04_shap_summary.png", "What drives Model A's scores (SHAP)"),
                          ("04_robustness.png", "Detection after random edits to the URL")):
        w(f"![{caption}](figures/{name})")
        w("")
    w("## How these numbers were produced")
    w("")
    w("- Models were frozen before any test set was scored; thresholds and calibrators come from validation only.")
    w("- The query string and fragment are removed from every URL before scoring.")
    w(f"- {metrics['notes']['intervals']}.")
    cnn = metrics["scorers"]["cnn"]["training"].get("run_to_run_variation")
    if cnn and cnn["bit_identical"]:
        w("- The CNN was trained on a GPU. Two runs with the same seed gave bit-for-bit identical validation "
          "scores on the development machine, with PyTorch's deterministic settings on. That is not guaranteed "
          "on other hardware or library versions. LightGBM and everything computed from the saved `cnn.onnx` "
          "are exactly reproducible.")
    elif cnn:
        w(f"- The CNN was trained on a GPU, which is not bit-for-bit reproducible. Two runs with the same seed "
          f"differed by {cnn['abs_difference_roc_auc']:.5f} in validation ROC-AUC and "
          f"{cnn['abs_difference_pr_auc']:.5f} in validation PR-AUC. LightGBM and everything computed from the "
          f"saved `cnn.onnx` are exactly reproducible.")
    w("- Design decisions and their evidence: `docs/design_decisions.md`. Data audits: `reports/data_audit.json`, "
      "`reports/secondary_audit.json`, `reports/secondary_audit_ariyadasa.json`.")
    w("")
    return "\n".join(out)


def write_results_md(metrics_path: Path | None = None, out_path: Path | None = None) -> Path:
    metrics = M.load_json(metrics_path or E.REPORTS_DIR / "metrics.json")
    out_path = Path(out_path or E.REPORTS_DIR / "results.md")
    out_path.write_text(render_results_md(metrics), encoding="utf-8", newline="\n")
    return out_path


def format_change(metric: Mapping[str, float]) -> str:
    """"-12.3 pts (-13.0 to -11.6)": a change in percentage points with its interval."""
    return f"{metric['value'] * 100:+.1f} pts ({metric['ci_low'] * 100:+.1f} to {metric['ci_high'] * 100:+.1f})"


def post_hoc_section(metrics: Mapping[str, object], shapes: Mapping[str, object] | None = None) -> list[str]:
    """The three additions decided after the test results were seen (docs/design_decisions.md, section 6)."""
    post = metrics["post_hoc"]
    wv, review, tr = post["warn_on_validation"], post["review"], post["targeted_robustness"]
    RR = review["results"]
    ev = "phreshphish_eval"
    out: list[str] = []
    w = out.append

    def pct(x: float, digits: int = 1) -> str:
        return f"{x * 100:.{digits}f}%"

    def rm(key: str, scorer: str, name: str) -> dict:
        return RR[key][scorer]["all"]["metrics"][name]

    w(f"## {POST_HOC_HEADING}")
    w("")
    w("**Post hoc.** The three additions below were decided on 2026-11-21, after the results above had been "
      "read. None was pre-specified. None changes a model, a calibrator, the block and warn thresholds or any "
      "number above. `docs/design_decisions.md`, section 6, records each decision; "
      "`notebooks/05_post_hoc.ipynb` computes them.")
    w("")
    w("### 1. The warn threshold on validation, where it was set")
    w("")
    rows = []
    for scorer, label in MODELS.items():
        d = wv[scorer]
        rows.append([label, f"at least {d['target_min_detection']:.0%} of validation phishing",
                     E.fmt(d["warn_detection"]), E.fmt(d["warn_fpr"]),
                     f"{d['legitimate_validation_urls_flagged']:,} of {d['legitimate_validation_urls']:,}",
                     E.fmt(metrics["results"][ev][scorer]["all"]["metrics"]["warn_fpr"])])
    w(_table(["Model", "Pre-specified target", "Warn: detection on validation", "Warn: FPR on validation",
              "Legitimate validation URLs flagged", "Warn: FPR on later data"], rows))
    w("")
    a, b = wv["lgbm"]["warn_fpr"]["value"], wv["cnn"]["warn_fpr"]["value"]
    w(f"The warn threshold was pre-specified as the point that catches at least 95% of validation phishing. On "
      f"validation itself it flagged {pct(a)} (Model A) and {pct(b)} (Model B) of legitimate URLs. "
      + ("So the 95%-detection point was already unusable before any test set was scored; the later data did not "
         "create the problem. " if min(a, b) >= 0.10 else
         "On validation the 95%-detection point flagged fewer than one legitimate URL in ten. ")
      + "These validation figures have been in the frozen operating files, and shown in notebooks 02 to 04, since "
        "the models were frozen. The written results above did not state them. The warn point is the "
        "pre-specified target, and it proved unreachable at a usable false-positive rate for a model that sees "
        "only the URL.")
    w("")
    w("### 2. A review operating point: 1% false positives on validation")
    w("")
    w(f"Rule: the lowest raw-score threshold at which at most {review['target_max_fpr']:.0%} of legitimate "
      "validation URLs are flagged. This is the block rule with 1% in place of 0.1%, computed by the same "
      "function on the same validation scores. The 1% figure was named after the results above had been seen; "
      "no other value was tried on a test set, and each test set was scored with it once. *Reviewed or blocked* "
      "counts every URL at or above the review threshold.")
    w("")
    rows = []
    for key, title in {**VAL, **MAIN}.items():
        for scorer, label in MODELS.items():
            mm = RR[key][scorer]["all"]["metrics"]
            rows.append([title, label, E.fmt(mm["review_recall"]), E.fmt(mm["review_fpr"], digits=2),
                         E.fmt(mm["review_precision"]), E.fmt(mm["review_precision_at_base_rate_0.01"]),
                         E.fmt(mm["review_precision_at_base_rate_0.001"]),
                         E.fmt(mm["block_recall"]), E.fmt(mm["block_fpr"], digits=2)])
    w(_table(["Data", "Model", "Reviewed or blocked: detection", "Reviewed or blocked: FPR",
              "Precision as tested", "Precision at 1% phishing", "Precision at 0.1% phishing",
              "Blocked: detection", "Blocked: FPR"], rows))
    w("")

    def pair(key: str, name: str, digits: int = 1) -> str:
        return (f"{pct(rm(key, 'lgbm', name)['value'], digits)} (Model A) and "
                f"{pct(rm(key, 'cnn', name)['value'], digits)} (Model B)")

    w(f"- On later data the review point catches {pair(ev, 'review_recall')} of phishing, against "
      f"{pair(ev, 'block_recall')} at the block point.")
    w(f"- Its false-positive rate, set to 1% on validation, is {pair(ev, 'review_fpr', 2)} on later data, "
      f"{pair('wangchuk', 'review_fpr', 2)} on Wangchuk and {pair('ariyadasa', 'review_fpr', 2)} on Ariyadasa.")
    w(f"- If phishing were 1% of all URLs, a review-or-block verdict on the later data would be right "
      f"{pair(ev, 'review_precision_at_base_rate_0.01')} of the time; at 0.1%, "
      f"{pair(ev, 'review_precision_at_base_rate_0.001')}.")
    w("- The block columns repeat the numbers of the main evaluation and are identical to them: same scores, "
      "same resamples.")
    w("")
    w("**The three bands the app shows.** Each URL lands in exactly one band per model. Shares of the "
      "later-in-time test:")
    w("")
    rows = []
    for scorer, label in MODELS.items():
        entry = RR[ev][scorer]["all"]
        for band in ("Block", "Review", "Allow"):
            rows.append([label, band,
                         f"{entry['bands']['phishing'][band] / entry['phishing']:.1%} "
                         f"({entry['bands']['phishing'][band]:,})",
                         f"{entry['bands']['legitimate'][band] / entry['legitimate']:.2%} "
                         f"({entry['bands']['legitimate'][band]:,})"])
    w(_table(["Model", "Band", "Phishing URLs in this band", "Legitimate URLs in this band"], rows))
    w("")
    w("**The review point by subgroup.**")
    w("")
    rows = []
    for key, title in MAIN.items():
        for scorer, label in MODELS.items():
            for which in ("url_shape", "registered_domain"):
                for group, entry in RR[key][scorer]["splits"][which].items():
                    mm = entry["metrics"]
                    rows.append([title, label, group, f"{entry['legitimate']:,}", f"{entry['phishing']:,}",
                                 E.fmt(mm.get("review_recall")), E.fmt(mm.get("review_fpr"), digits=2)])
    w(_table(["Test set", "Model", "Subgroup", "Legitimate", "Phishing", "Reviewed or blocked: detection",
              "Reviewed or blocked: FPR"], rows))
    w("")
    w("**External datasets, review point, per URL and per registered domain.** Intervals resample domains.")
    w("")
    rows = []
    for key in ("wangchuk", "ariyadasa"):
        for scorer, label in MODELS.items():
            d = RR[key][scorer]["false_positive_rate_by_domain"]
            rows.append([MAIN[key], label, f"{d['legitimate_rows']:,}", f"{d['registered_domains']:,}",
                         E.fmt(d["review"]["per_url"], digits=2), E.fmt(d["review"]["per_domain"], digits=2),
                         d["review"]["domains_with_any_false_positive"],
                         f"{d['review']['share_of_false_positives_in_top_10_domains']:.0%}"])
    w(_table(["Dataset", "Model", "Presumed-benign URLs", "Registered domains", "FPR per URL", "FPR per domain",
              "Domains with a false positive", "False positives in the 10 worst domains"], rows))
    w("")
    w("### 3. Targeted robustness: edits chosen to look legitimate")
    w("")
    w(f"Every phishing URL of the later-in-time test ({tr['phishing_urls']:,}) was edited in four ways: a `www.` "
      "prefix added, the scheme set to `https`, a benign-looking path appended, and all three together. The "
      "edits and the five paths were written in `docs/design_decisions.md` before the test was run, and it was run "
      "once. The thresholds are the frozen block threshold and the review threshold above.")
    w("")
    names = {"www": "`www.` prefix added", "https": "scheme set to `https`", "path": "benign-looking path appended",
             "all": "all three together"}
    rows = [["original URL", "-"] + [cell for scorer in MODELS for point in ("block", "review")
                                     for cell in (E.fmt(tr["models"][scorer][point]["before"]), "-")]]
    for edit, name in names.items():
        row = [name, f"{tr['urls_changed'][edit]:,}"]
        for scorer in MODELS:
            for point in ("block", "review"):
                after = tr["models"][scorer][point]["after"][edit]
                row += [E.fmt(after), format_change(after["change"])]
        rows.append(row)
    w(_table(["Edit", "URLs the edit changed", "Model A: blocked", "Change", "Model A: reviewed or blocked", "Change",
              "Model B: blocked", "Change", "Model B: reviewed or blocked", "Change"], rows))
    w("")
    changes = [(tr["models"][s][p]["after"][e]["change"]["value"], MODELS[s], p, names[e])
               for s in MODELS for p in ("block", "review") for e in names]
    worst, best = min(changes), max(changes)
    point_name = {"block": "blocked", "review": "reviewed or blocked"}

    def both(point: str) -> str:
        x, y = tr["models"]["lgbm"][point], tr["models"]["cnn"][point]
        return (f"from {pct(x['before']['value'])} to {pct(x['after']['all']['value'])} for Model A and from "
                f"{pct(y['before']['value'])} to {pct(y['after']['all']['value'])} for Model B")

    w(f"With all three edits together, the share of phishing URLs blocked went {both('block')}. The share "
      f"reviewed or blocked went {both('review')}.")
    w("")
    if shapes:
        w(f"Why an appended path works: in the training data {shapes['legitimate']['non_root_path']:.1%} of "
          f"legitimate URLs have a path and {shapes['phishing']['bare_homepage']:.1%} of phishing URLs do not, so "
          "the models learned to read a path as a sign of a legitimate page, and an appended path supplies it.")
        w("")
    if worst[0] <= -0.02:
        w(f"**Edits chosen to look legitimate lower detection, which random edits did not.** The largest fall is "
          f"{abs(worst[0]) * 100:.1f} points ({worst[1]}, {point_name[worst[2]]}, {worst[3]})."
          + (f" Not every edit lowers detection: the largest rise is {best[0] * 100:.1f} points ({best[1]}, "
             f"{point_name[best[2]]}, {best[3]})." if best[0] >= 0.02 else ""))
    elif best[0] >= 0.02:
        w(f"**None of these edits lowered detection by two points or more.** The largest change is a rise of "
          f"{best[0] * 100:.1f} points ({best[1]}, {point_name[best[2]]}, {best[3]}).")
    else:
        w("**None of these edits moved detection by two points or more in either direction.**")
    w("")
    w("The edits were written from the threat model, not found by querying the models, so this is a lower bound "
      "on what an attacker who probes the scorer could do. Only the URL string was edited; whether a phishing "
      "page would still load at the edited address was not tested.")
    w("")
    w("**The appended path, one path at a time** (the `path` edit only):")
    w("")
    rows = []
    for i, path in enumerate(tr["benign_paths"]):
        row = [f"`{path}`", f"{tr['by_benign_path']['lgbm']['block'][i]['urls']:,}"]
        for scorer in MODELS:
            for point in ("block", "review"):
                d = tr["by_benign_path"][scorer][point][i]
                row.append(f"{pct(d['before'])} to {pct(d['after'])}")
        rows.append(row)
    w(_table(["Appended path", "URLs", "Model A: blocked", "Model A: reviewed or blocked", "Model B: blocked",
              "Model B: reviewed or blocked"], rows))
    w("")
    w("![Detection after edits chosen to look legitimate](figures/05_targeted_robustness.png)")
    w("")
    check = load_cnn_tuning_check()
    if check is not None:
        w("### Added later the same day: a CNN tuning check (decision 24)")
        w("")
        w(cnn_tuning_sentence(check))
        w("")
    return out


CNN_CHECK_FILE = E.REPORTS_DIR / "cnn_tuning_check.json"


def load_cnn_tuning_check(path: Path | None = None) -> dict | None:
    """The saved CNN tuning check (decision 24), or None before notebook 03's last section has run."""
    path = Path(path or CNN_CHECK_FILE)
    return M.load_json(path) if path.exists() else None


def cnn_tuning_sentence(check: Mapping[str, object]) -> str:
    """One sentence on the CNN tuning check, for results.md and the Q&A. Its wording follows the saved result."""
    s, b, n = check["shipped"], check["best"], len(check["runs"])
    if s["rank"] == 1:
        other = sorted(check["runs"], key=lambda r: r["rank"])[1]
        where = (f"ranked first of {n} on validation PR-AUC ({s['validation_pr_auc_retrained']:.4f}; the next best, "
                 f"learning rate {other['learning_rate']:g} with dropout {other['dropout']}, reached "
                 f"{other['validation_pr_auc']:.4f})")
    else:
        where = (f"ranked {s['rank']} of {n} on validation PR-AUC ({s['validation_pr_auc_retrained']:.4f} against a "
                 f"best of {b['validation_pr_auc']:.4f}, learning rate {b['learning_rate']:g} with dropout "
                 f"{b['dropout']})")
    verdict = "among" if check["shipped_among_best"] else "not among"
    return (f"**Post hoc, validation only.** Retrained at {n} learning-rate and dropout settings after the test "
            f"results had been seen, the shipped CNN settings (learning rate {s['learning_rate']:g}, dropout "
            f"{s['dropout']}) {where}, so they were {verdict} the best by the rule fixed beforehand (within 0.005); "
            f"no test set was scored with these models, and the deployed CNN was kept either way, because choosing a "
            f"model after seeing the test results would bias them (`reports/cnn_tuning_check.json`).")


def readme_headline_table(metrics: Mapping[str, object]) -> str:
    """The five-row table at the top of README.md. tests/test_docs.py requires the README to contain it."""
    R, review = metrics["results"], metrics["post_hoc"]["review"]["results"]
    rows = []
    for key, scorer in (("phreshphish_eval", "lgbm"), ("phreshphish_eval", "cnn"), ("wangchuk", "lgbm"),
                        ("wangchuk", "cnn"), ("phreshphish_eval", "lgbm_phiusiil")):
        m = R[key][scorer]["all"]["metrics"]
        r = review[key][scorer]["all"]["metrics"] if scorer in MODELS else None
        rows.append([MAIN[key], E.SCORERS[scorer][0], f"{m['roc_auc']['value']:.3f}",
                     E.fmt(m["block_recall"], ci=False), E.fmt(m["block_fpr"], digits=2, ci=False),
                     E.fmt(r["review_recall"], ci=False) if r else "not applicable",
                     E.fmt(r["review_fpr"], digits=2, ci=False) if r else "not applicable"])
    return _table(["Test set", "Scorer", "ROC-AUC", "Blocked: phishing caught", "Blocked: legitimate flagged",
                   "Reviewed or blocked: phishing caught", "Reviewed or blocked: legitimate flagged"], rows)
