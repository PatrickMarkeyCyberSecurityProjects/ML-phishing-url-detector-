"""Tests for src/audit.py. No network: small invented frames only."""

import pandas as pd
import pytest

from src import audit as A
from src import data as D
from src import features as F


def _frame(rows, start="2025-01-01", source="x"):
    """rows: (raw URL, label, day offset). Returns a harmonized-style frame (main track)."""
    return pd.DataFrame({
        "url": [F.normalize_url(u, strip_query=True) for u, _, _ in rows],
        "url_full": [F.normalize_url(u) for u, _, _ in rows],
        "had_query": [bool(F.split_url(u).query) for u, _, _ in rows],
        "url_raw": [u for u, _, _ in rows],
        "label": [label for _, label, _ in rows],
        "source": source,
        "date": [pd.Timestamp(start) + pd.Timedelta(days=d) for _, _, d in rows],
    })


def _mixed(prefix, n=40, start="2025-01-01"):
    """Legitimate and phishing rows whose shape does not give the class away."""
    rows = []
    for i in range(n):
        rows.append((f"https://www.{prefix}-good{i}.example/page/{i}", 0, i % 10))
        rows.append((f"https://{prefix}-good{i}.example", 0, i % 10))
        rows.append((f"http://{prefix}-bad{i}.example/login/{i}", 1, i % 10))
        rows.append((f"https://www.{prefix}-bad{i}.example", 1, i % 10))
    return _frame(rows, start)


# ------------------------------------------------------------------ measurements

def test_feature_auc_is_direction_free():
    assert A.feature_auc([0, 0, 1, 1], [0, 0, 1, 1]) == {"auc": 1.0, "direction_free": 1.0}
    assert A.feature_auc([0, 0, 1, 1], [1, 1, 0, 0]) == {"auc": 0.0, "direction_free": 1.0}
    assert A.feature_auc([0, 1, 0, 1], [1, 1, 0, 0])["direction_free"] == 0.5


def test_shape_shares_and_top_domains():
    profiled = D.profile(_frame([
        ("https://www.a.example", 0, 0), ("a.example/x?q=1", 0, 0),
        ("http://b.example/", 1, 0), ("http://c.b.example/y", 1, 0),
    ]))
    shares = A.shape_shares(profiled)
    assert shares["legitimate"]["bare_homepage"] == 0.5
    assert shares["legitimate"]["non_root_path"] == 0.5 and shares["legitimate"]["has_query"] == 0.5
    assert shares["legitimate"]["leading_www"] == 0.5
    assert shares["legitimate"]["robots_txt"] == 0.0
    assert shares["legitimate"]["scheme_as_shipped"] == {"https": 0.5, "none": 0.5}
    assert shares["legitimate"]["ends_at_host_as_shipped"] == 0.5
    assert shares["phishing"]["scheme_as_shipped"] == {"http": 1.0}
    top = A.top_domains(profiled, n=1)
    assert top["phishing"] == [{"registered_domain": "b.example", "rows": 2, "share": 1.0}]
    assert top["legitimate_distinct_domains"] == 1


def test_url_overlap_counts_exact_and_loose_matches():
    a = _frame([("https://www.one.example/", 0, 0), ("http://two.example/x", 1, 0), ("https://three.example/", 0, 0)])
    b = _frame([("http://one.example", 1, 0), ("http://two.example/x", 1, 0)])
    out = A.url_overlap(a, b)
    assert (out["exact"], out["loose"]) == (1, 2)
    assert out["loose_by_class"] == {"legitimate": 1, "phishing": 1}
    assert out["loose_with_different_label"] == 1


def test_domain_overlap_by_class_uses_class_specific_references():
    a = D.profile(_frame([("https://x.shared.example/a", 0, 0), ("https://new.example/b", 0, 0),
                          ("http://y.shared.example/c", 1, 0)]))
    b = D.profile(_frame([("https://shared.example/z", 1, 0)]))
    out = A.domain_overlap_by_class(a, b)
    assert out["legitimate"]["share_with_domain_in_any"] == 0.5
    assert out["legitimate"]["share_with_domain_in_legitimate"] == 0.0
    assert out["phishing"]["share_with_domain_in_phishing"] == 1.0
    assert out["all"]["share_with_domain_in_any"] == pytest.approx(2 / 3, abs=1e-6)


def test_feature_pattern_scan_flags_one_sided_features_only():
    rows = [(f"https://www.good{i}.example/page/{i}", 0, 0) for i in range(300)]
    rows += [(f"https://bad{i}.example/page/{i}?id={i}", 1, 0) for i in range(60)]
    rows += [(f"https://bad{i}.example/page/{i}", 1, 0) for i in range(60, 300)]
    frame = _frame(rows)
    full = A.feature_pattern_scan(frame, strip_query=False)
    assert set(full["flagged"]) >= set(A.QUERY_FEATURES)
    assert full["features"]["query_len"]["nonzero_share_legitimate"] == 0.0
    assert full["features"]["query_len"]["nonzero_share_phishing"] == 0.2
    main = A.feature_pattern_scan(frame, strip_query=True)
    assert "query_len" not in main["flagged"] and "n_query_params" not in main["flagged"]
    assert main["features"]["query_len"]["nonzero_share_phishing"] == 0.0
    # `subdomain_depth` is non-zero for every legitimate row (www.) and no phishing row,
    # so it is one-sided in both tracks and is reported apart from the query features.
    assert "subdomain_depth" in main["flagged_besides_query_features"]
    assert main["features"]["path_len"]["flagged_nonzero"] is False


def test_resource_shares_count_robots_and_assets():
    profiled = D.profile(_frame([
        ("https://a.example/robots.txt", 0, 0), ("https://a.example/app.js", 0, 0),
        ("https://a.example/about", 0, 0), ("https://a.example/", 0, 0), ("http://b.example/x.png", 1, 0),
    ]))
    out = A.resource_shares(profiled)
    assert out["legitimate"]["robots_txt"] == 1 and out["legitimate"]["robots_txt_share"] == 0.25
    assert out["legitimate"]["other_non_page_resource"] == 1 and out["legitimate"]["pages"] == 2
    assert out["phishing"]["other_non_page_resource_share"] == 1.0


def test_class_by_month():
    frame = _frame([("https://a.example/", 0, 0), ("https://b.example/", 1, 0), ("https://c.example/", 1, 40)])
    assert A.class_by_month(frame) == {
        "2025-01": {"legitimate": 1, "phishing": 1}, "2025-02": {"legitimate": 0, "phishing": 1},
    }


# ------------------------------------------------------------------ gate 1

def _gate1(train, val, eval_sample, test_full):
    return A.build_data_audit(
        train=D.profile(train), val=D.profile(val), eval_sample=D.profile(eval_sample),
        test_full=test_full, reports={},
    )["gate1"]


def test_gate1_passes_when_test_is_later_and_no_feature_separates_the_classes():
    train, val = _mixed("tr", start="2025-01-01"), _mixed("va", start="2025-02-01")
    test = _mixed("te", start="2025-03-01")
    gate = _gate1(train, val, test, test)
    assert gate["passed"] is True and gate["reasons"] == [] and gate["date_overlap"] is False


def test_gate1_stops_on_a_shared_day():
    train, val = _mixed("tr", start="2025-01-01"), _mixed("va", start="2025-02-01")
    test = _mixed("te", start="2025-02-10")          # validation runs to 2025-02-10
    audit = A.build_data_audit(train=D.profile(train), val=D.profile(val), eval_sample=D.profile(test),
                               test_full=test, reports={})
    assert audit["gate1"]["passed"] is False and audit["gate1"]["date_overlap"] is True
    assert audit["temporal"]["shared_days"] == ["2025-02-10"]
    assert audit["temporal"]["train_rows_on_or_after_test_min"]["rows"] == 16


def test_gate1_stops_when_one_feature_separates_the_classes_in_either_direction():
    rows = [(f"https://www.good{i}.example", 0, i % 5) for i in range(40)]
    rows += [(f"https://www.bad{i}.example/login/{i}", 1, i % 5) for i in range(40)]
    leaky = _frame(rows, "2025-03-01")
    train, val = _mixed("tr", start="2025-01-01"), _mixed("va", start="2025-02-01")
    gate = _gate1(train, val, leaky, leaky)
    assert gate["passed"] is False
    assert gate["highest_single_feature_auc"]["direction_free"] == 1.0
    assert gate["highest_single_feature_auc"]["split"] == "eval_sample"
    # The reversed artefact (paths go with legitimate URLs) is caught as well.
    reversed_rows = [(u, 1 - label, d) for u, label, d in rows]
    gate = _gate1(train, val, _frame(reversed_rows, "2025-03-01"), _frame(reversed_rows, "2025-03-01"))
    assert gate["passed"] is False


# ------------------------------------------------------------------ gate 2

def _gate2(secondary_rows, documented=None, empty_urls=0):
    raw = pd.DataFrame({"URL": [u for u, _, _ in secondary_rows],
                        "Label": [str(label) for _, label, _ in secondary_rows]})
    harmonized, h = D.harmonize(raw, url_col="URL", label_col="Label", label_map=D.WANGCHUK_LABEL_MAP,
                                source="S", strip_query=True)
    dedup, d = D.deduplicate(harmonized)
    train, val, ev = _mixed("tr"), _mixed("va", start="2025-02-01"), _mixed("te", start="2025-03-01")
    final, x = D.remove_cross_dataset_overlap(dedup, pd.concat([train, val]))
    counts = raw["Label"].value_counts()
    spec = {"name": "S", "doi": "10.0/x", "filename": "s.csv", "sha256": "abc",
            "documented_counts": documented or {0: int(counts.get("0", 0)), 1: int(counts.get("1", 0))}}
    source = {"file_sha256": "abc", "file_bytes": 1, "columns": ["URL", "Label"], "url_column": "URL",
              "label_column": "Label", "label_direction": "1 = phishing", "label_map": {"1": 1, "0": 0},
              "raw_label_counts": {str(k): int(v) for k, v in counts.items()}, "empty_urls": empty_urls}
    return A.build_secondary_audit(
        spec=spec, source=source, harmonize_report=h, dedup_report=d, removal_report=x,
        as_shipped=D.profile(harmonized), deduplicated=D.profile(dedup), final=D.profile(final),
        train=D.profile(train), val=D.profile(val), eval_sample=D.profile(ev),
    )


def _secondary_rows(n=30):
    rows = []
    for i in range(n):
        rows.append((f"https://www.s-good{i}.example/page/{i}", 0, 0))
        rows.append((f"https://s-good{i}.example/about", 0, 0))
        rows.append((f"http://s-bad{i}.example/login/{i}", 1, 0))
        rows.append((f"https://www.s-bad{i}.example", 1, 0))
    return rows


def test_gate2_accepts_a_clean_secondary_and_removes_training_urls():
    rows = _secondary_rows() + [("http://tr-bad0.example/login/0", 1, 0), ("https://s-good0.example/about", 0, 0)]
    audit = _gate2(rows)
    assert audit["gate2"]["passed"] is True
    assert audit["duplicates"]["duplicate_rows_after_normalization"] == 1
    assert audit["overlap_with_phreshphish"]["vs_train_and_val"]["loose"] == 1
    assert audit["overlap_with_phreshphish"]["removed"]["removed"] == 1
    assert audit["overlap_with_phreshphish"]["remaining_after_removal"]["loose"] == 0
    assert audit["class_counts"]["match"] is True
    assert audit["dataset"]["legitimate_class_name"] == "presumed benign"


def test_gate2_rejects_a_homepage_only_legitimate_class():
    rows = [(f"https://www.s-good{i}.example", 0, 0) for i in range(40)]
    rows += [(f"https://www.s-bad{i}.example", 1, 0) for i in range(20)]
    rows += [(f"https://s-bad{i}.example/x", 1, 0) for i in range(20)]
    audit = _gate2(rows)
    assert audit["gate2"]["passed"] is False
    assert any("bare homepages" in reason for reason in audit["gate2"]["reasons"])


def test_gate2_rejects_a_shape_feature_that_separates_the_classes():
    rows = [(f"https://www.s-good{i}.example/a/{i}", 0, 0) for i in range(40)]
    rows += [(f"http://s-bad{i}.example/a/{i}", 1, 0) for i in range(40)]
    audit = _gate2(rows)
    assert audit["gate2"]["passed"] is False
    assert audit["gate2"]["criteria"]["highest_single_feature_auc"]["direction_free"] == 1.0


def test_gate2_rejects_counts_that_do_not_match_the_documentation():
    audit = _gate2(_secondary_rows(), documented={0: 1, 1: 1})
    assert audit["gate2"]["passed"] is False
    assert audit["gate2"]["criteria"]["raw_urls_and_labels_recoverable"] is False


def test_phishing_provenance_detects_the_same_list():
    secondary = _frame([("http://bad1.example/a", 1, 0), ("http://bad2.example/b", 1, 0), ("https://ok.example/", 0, 0)])
    same = _frame([("http://bad1.example/a", 1, 0), ("http://bad2.example/b", 1, 0)])[["url", "url_raw"]]
    other = _frame([("http://bad1.example/a", 1, 0), ("http://else.example/b", 1, 0)])[["url", "url_raw"]]
    spec = {"name": "ref", "doi": "10.0/ref", "published": "2024-04-02"}
    assert A.phishing_provenance(secondary, same, spec)["same_list"] is True
    assert A.phishing_provenance(secondary, other, spec)["same_list"] is False


def test_write_and_read_json(tmp_path):
    path = A.write_json({"a": 1, "when": pd.Timestamp("2025-01-01")}, tmp_path / "sub" / "x.json")
    assert A.read_json(path)["a"] == 1


def test_template_rule_result_counts_every_cell_of_the_table():
    profiled = pd.DataFrame({"label": [0, 0, 1, 1, 1], "homepage_template": [True, False, False, False, True]})
    rule = A.template_rule_result(profiled)
    assert rule["counts"] == {"legitimate_called_legitimate": 1, "legitimate_called_phishing": 1,
                              "phishing_called_phishing": 2, "phishing_called_legitimate": 1}
    assert (rule["rows"], rule["legitimate"], rule["phishing"], rule["correct"]) == (5, 2, 3, 3)
    assert rule["accuracy"] == pytest.approx(0.6)
