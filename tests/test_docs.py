"""The submission documents must agree with the saved numbers and with the app."""

import re
from pathlib import Path

import pytest

from src import business
from src import models as M
from src import report

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
METRICS = ROOT / "reports" / "metrics.json"
needs_results = pytest.mark.skipif(
    not METRICS.exists() or "post_hoc" not in M.load_json(METRICS), reason="run notebooks 04 and 05 first")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _flat(path: Path) -> str:
    return " ".join(_text(path).split())


# ------------------------------------------------------------------ README

@needs_results
def test_readme_headline_table_is_the_one_generated_from_metrics_json():
    assert report.readme_headline_table(M.load_json(METRICS)) in _text(ROOT / "README.md")


@needs_results
def test_readme_numbers_come_from_the_saved_files():
    readme = _flat(ROOT / "README.md")
    metrics = M.load_json(METRICS)
    R, post = metrics["results"], metrics["post_hoc"]
    shapes = M.load_json(ROOT / "reports" / "data_audit.json")["shape_by_class"]["train"]
    for name, label in (("lgbm", "Model A"), ("cnn", "Model B")):
        block = post["targeted_robustness"]["models"][name]["block"]
        assert f"from {block['before']['value']:.1%} to {block['after']['all']['value']:.1%} for {label}" in readme
        assert f"{R['phreshphish_eval'][name]['all']['metrics']['block_recall']['value']:.1%} ({label})" in readme
    assert f"{shapes['legitimate']['non_root_path']:.1%} of legitimate URLs have a path" in readme
    assert f"{shapes['phishing']['bare_homepage']:.1%} of phishing URLs do not" in readme


def test_readme_opens_with_what_a_reader_needs_first():
    text = _text(ROOT / "README.md")
    first_heading = text.index("\n## ")
    top = text[:first_heading]
    for item in ("Live demo", "| Test set |", "04_model_comparison.png", "python app/app.py", "(#limitations)",
                 "reports/results.md"):
        assert item in top, item
    assert len(top.splitlines()) <= 45, "the first screen should stay short"
    for heading in ("## Limitations", "## Reproduce", "## Data sources and licences", "## Use of generative AI"):
        assert heading in text
    ai = text[text.index("## Use of generative AI"):]
    assert "docs/ai_usage.md" in ai and len(re.findall(r"[.!?](?:\s|$)", ai.split("\n", 1)[1].strip())) == 2
    for figure in re.findall(r"\]\(((?:reports|docs)/[^)]+\.png)\)", text):
        assert (ROOT / figure).exists(), figure


def test_every_document_the_readme_links_to_exists():
    for link in re.findall(r"\]\(((?:docs|reports)/[^)#]+)\)", _text(ROOT / "README.md")):
        assert (ROOT / link).exists(), link


@needs_results
def test_business_case_file_is_in_step_with_metrics():
    saved = M.load_json(ROOT / "reports" / "business_case.json")
    assert saved == business.business_case(M.load_json(METRICS))
    assert "Not a measurement" in saved["what_this_is"] and "not_estimated" in saved["assumptions"]
    later = saved["scenarios"]["phreshphish_eval"]["models"]["cnn"]["phishing_rate_0.001"]
    assert later["phishing_blocked"] + later["phishing_sent_to_review"] + later["phishing_allowed_missed"] == 1000


@needs_results
def test_business_scenario_arithmetic():
    entry = {"phishing": 1000, "legitimate": 10000,
             "bands": {"phishing": {"Block": 500, "Review": 300, "Allow": 200},
                       "legitimate": {"Block": 10, "Review": 100, "Allow": 9890}}}
    s = business.scenario(entry, 1_000_000, 0.001)
    assert (s["phishing_links"], s["phishing_blocked"], s["phishing_sent_to_review"], s["phishing_allowed_missed"]) == (
        1000, 500, 300, 200)
    assert s["legitimate_blocked"] == 999 and s["legitimate_sent_to_review"] == 9990
    assert s["share_of_blocked_links_that_are_phishing"] == pytest.approx(500 / 1499)
    assert s["share_of_phishing_blocked_or_reviewed"] == pytest.approx(0.8)
    assert s["links_a_slower_check_would_see"] == 10290


@needs_results
def test_threat_model_test_column_quotes_the_saved_results():
    flat = _flat(DOCS / "problem_and_threat_model.md")
    metrics = M.load_json(METRICS)
    post = metrics["post_hoc"]
    for name, label in (("lgbm", "Model A"), ("cnn", "Model B")):
        block = post["targeted_robustness"]["models"][name]["block"]
        assert f"from {block['before']['value']:.1%} to {block['after']['all']['value']:.1%} ({label})" in flat
        warn = post["warn_on_validation"][name]["warn_fpr"]["value"]
        assert f"{warn:.1%} ({label})" in flat
    seen = metrics["results"]["phreshphish_eval"]["lgbm"]["splits"]["registered_domain"]
    assert (f"{seen['seen in training']['metrics']['block_recall']['value']:.1%} of phishing on seen domains and "
            f"{seen['unseen in training']['metrics']['block_recall']['value']:.1%} on unseen ones") in flat
    assert "How this project tests it" in flat and "post hoc" in flat


def test_template_rule_figures_come_from_the_saved_file():
    rule = M.load_json(ROOT / "reports" / "phiusiil_template_rule.json")
    assert rule["correct"] == rule["counts"]["legitimate_called_legitimate"] + rule["counts"]["phishing_called_phishing"]
    assert rule["rows"] == rule["legitimate"] + rule["phishing"] == sum(rule["counts"].values())
    accuracy, rows = f"{rule['accuracy']:.2%}", f"{rule['rows']:,}"
    assert (accuracy, rows) == ("99.73%", "235,795")         # the figures the documents were written with
    quoting = [doc for doc in DOCUMENTS if "archive" not in doc.parts and "99.7" in _text(doc)]
    assert quoting
    external = "99.79%"                                      # the PhiUSIIL authors' figure, a different number
    for doc in quoting:
        for found in re.findall(r"99\.7\d%", _text(doc)):
            assert found in (accuracy, external), f"{doc.relative_to(ROOT)} says {found}"
    for doc in ("problem_and_threat_model.md", "design_decisions.md"):
        assert "reports/phiusiil_template_rule.json" in _text(DOCS / doc), doc


# ------------------------------------------------------------------ links and paths across the repository

DOCUMENTS = [ROOT / "README.md", ROOT / "app" / "README.md", ROOT / "reports" / "results.md",
             *sorted((ROOT / "docs").rglob("*.md"))]


def test_every_relative_link_in_every_document_resolves():
    for doc in DOCUMENTS:
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", _text(doc)):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            assert (doc.parent / target).resolve().exists(), f"{doc.relative_to(ROOT)} links to {target}"


def test_backticked_repository_paths_exist():
    pattern = r"`((?:docs|reports|notebooks|src|app|frozen|tests|models)/[A-Za-z0-9_./-]+\.[a-z]+)`"
    for doc in DOCUMENTS:
        if "archive" in doc.parts:
            continue
        for path in set(re.findall(pattern, _text(doc))):
            assert (ROOT / path).exists(), f"{doc.relative_to(ROOT)} names {path}"


def test_no_file_points_to_a_moved_or_renamed_document():
    stale = ("docs/design_change.md", "docs/research.md", "docs/assignment.md", "docs/research_corrections.md")
    files = [*DOCUMENTS, *(ROOT / "src").glob("*.py"), *(ROOT / "app").glob("*.py"),
             *(ROOT / "notebooks").glob("*.ipynb")]
    for path in files:
        if "archive" in path.parts:
            continue
        text = _text(path)
        for name in stale:
            assert name not in text, f"{path.relative_to(ROOT)} still points to {name}"
