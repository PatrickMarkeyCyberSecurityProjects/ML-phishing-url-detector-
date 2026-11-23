"""Tests for src/data.py. No network: the steps run on small invented frames.

The last group checks the saved files in data/processed/ and is skipped until
the pipeline has been run.
"""

import json

import pandas as pd
import pytest

from src import data as D
from src import features as F


def _phiusiil_like(n_legit=60, n_phish=40):
    """A frame with PhiUSIIL's shape and polarity (1 = legitimate)."""
    rows = [(f"https://www.site{i}.example", 1) for i in range(n_legit)]
    rows += [(f"http://bad{i}.example/login/{i}", 0) for i in range(n_phish)]
    return pd.DataFrame(rows, columns=["URL", "label"])


def _harmonized(n_legit=600, n_phish=400):
    frame, _ = D.harmonize(
        _phiusiil_like(n_legit, n_phish), url_col="URL", label_col="label",
        label_map=D.PHIUSIIL_LABEL_MAP, source="PhiUSIIL",
    )
    return frame


# ------------------------------------------------------------------ label mapping

def test_label_maps_use_the_project_convention():
    assert D.PHIUSIIL_LABEL_MAP == {1: 0, 0: 1}            # ships inverted
    assert D.PHRESHPHISH_LABEL_MAP == {"phish": 1, "benign": 0}


def test_phiusiil_polarity_is_flipped():
    frame, report = D.harmonize(
        _phiusiil_like(60, 40), url_col="URL", label_col="label",
        label_map=D.PHIUSIIL_LABEL_MAP, source="PhiUSIIL", expected_counts={0: 60, 1: 40},
    )
    assert report["class_counts_after_mapping"] == {0: 60, 1: 40}
    assert frame.loc[frame["url"].str.contains("bad"), "label"].eq(1).all()
    assert frame.loc[frame["url"].str.contains("site"), "label"].eq(0).all()
    assert list(frame.columns) == ["url", "url_full", "url_raw", "label", "source", "had_query"]
    assert frame["label"].dtype == "int8"


def test_phreshphish_string_labels_map_to_the_same_convention():
    raw = pd.DataFrame(
        {"url": ["https://a.example", "http://b.example/x"], "label": ["benign", "phish"],
         "date": pd.to_datetime(["2025-01-01", "2025-02-01"])}
    )
    frame, report = D.harmonize(
        raw, url_col="url", label_col="label", label_map=D.PHRESHPHISH_LABEL_MAP,
        source="PhreshPhish", expected_counts={0: 1, 1: 1}, keep=("date",),
    )
    assert frame["label"].tolist() == [0, 1]
    assert "date" in frame.columns


def test_wrong_class_counts_raise():
    with pytest.raises(D.ClassCountMismatch):
        D.harmonize(
            _phiusiil_like(60, 40), url_col="URL", label_col="label",
            label_map=D.PHIUSIIL_LABEL_MAP, source="PhiUSIIL", expected_counts={0: 40, 1: 60},
        )


@pytest.mark.parametrize("bad_label", [2, "phish", None])
def test_unknown_label_values_raise(bad_label):
    raw = pd.DataFrame({"URL": ["https://a.example", "https://b.example"], "label": [1, bad_label]})
    with pytest.raises(ValueError):
        D.harmonize(raw, url_col="URL", label_col="label",
                    label_map=D.PHIUSIIL_LABEL_MAP, source="x")


def test_label_map_must_target_zero_and_one():
    raw = pd.DataFrame({"URL": ["https://a.example"], "label": [1]})
    with pytest.raises(ValueError):
        D.harmonize(raw, url_col="URL", label_col="label", label_map={1: 2}, source="x")


# ------------------------------------------------------------------ harmonize: URLs

def test_harmonize_normalizes_and_reports_changes():
    raw = pd.DataFrame(
        {"URL": [" HTTPS://WWW.Example.com/A ", "example.org/b", "https://xn--mnchen-3ya.example",
                 "https://ok.example", "   ", None],
         "label": [1, 0, 0, 1, 1, 0]}
    )
    frame, report = D.harmonize(raw, url_col="URL", label_col="label",
                                label_map=D.PHIUSIIL_LABEL_MAP, source="x")
    assert frame["url"].tolist() == [
        "https://www.example.com/A", "https://example.org/b",
        "https://münchen.example/", "https://ok.example/",
    ]
    assert frame["url_raw"].iloc[0] == " HTTPS://WWW.Example.com/A "
    assert report["empty_urls_dropped"] == 2
    assert report["rows_in"] == 6 and report["rows_out"] == 4
    assert report["urls_changed"] == 4
    assert report["n_root_slash_added"] == 2
    assert report["root_slash_added_by_class"] == {"legitimate": 1, "phishing": 1}
    assert report["n_scheme_added"] == 1
    assert report["n_host_lowercased"] == 1
    assert report["n_punycode_decoded"] == 1
    assert (frame["url"].map(F.normalize_url) == frame["url"]).all()


def test_harmonize_main_track_removes_the_query_and_remembers_it():
    raw = pd.DataFrame({
        "URL": ["https://a.example/x?id=1", "https://a.example/x?id=2", "https://b.example/#top",
                "https://c.example/y"],
        "label": [0, 0, 1, 1],
    })
    frame, report = D.harmonize(raw, url_col="URL", label_col="label", label_map=D.PHIUSIIL_LABEL_MAP,
                                source="x", strip_query=True)
    assert frame["url"].tolist() == ["https://a.example/x", "https://a.example/x",
                                     "https://b.example/", "https://c.example/y"]
    assert frame["url_full"].tolist() == raw["URL"].tolist()
    assert frame["had_query"].tolist() == [True, True, False, False]       # a fragment is not a query
    assert report["strip_query"] is True
    assert report["had_query_by_class"] == {"legitimate": 0, "phishing": 2}
    assert report["query_removed_by_class"] == {"legitimate": 1, "phishing": 2}
    full, full_report = D.harmonize(raw, url_col="URL", label_col="label", label_map=D.PHIUSIIL_LABEL_MAP,
                                    source="x")
    assert (full["url"] == full["url_full"]).all() and full_report["n_query_removed"] == 0
    # Deduplication then works on the query-stripped key.
    dedup, d = D.deduplicate(frame)
    assert d["duplicate_rows_dropped"] == 1 and dedup["url"].is_unique


def test_collapse_report_counts_without_dropping():
    frame = pd.DataFrame({"url": ["u1", "u1", "u2", "u2", "u3"], "label": [1, 1, 0, 1, 0]})
    out = D.collapse_report(frame)
    assert out["rows"] == 5 and out["distinct_urls"] == 3 and out["surplus_rows"] == 2
    assert out["rows_sharing_a_url"] == 4 and out["urls_shared_by_several_rows"] == 2
    assert out["urls_with_conflicting_labels"] == 1 and out["rows_with_conflicting_labels"] == 2


def test_drop_on_or_after_ends_the_frame_before_the_cut():
    frame = pd.DataFrame({"url": list("abcd"), "label": [0, 1, 0, 1],
                          "date": pd.to_datetime(["2025-09-06", "2025-09-07", "2025-09-08", "2025-09-09"])})
    out, report = D.drop_on_or_after(frame, "2025-09-08")
    assert out["url"].tolist() == ["a", "b"]
    assert report["rows_dropped"] == 2 and report["dropped_by_class"] == {"legitimate": 1, "phishing": 1}
    assert report["date_max_after"] == "2025-09-07"


def test_index_sql_rows_are_parsed_as_text_and_never_lost():
    text = "\n".join([
        "-- phpMyAdmin SQL Dump",
        "INSERT INTO `index` (`rec_id`, `url`, `website`, `result`, `created_date`) VALUES",
        "(1, 'http://bad.example/a.php', '1.html', 1, '2021-02-17 20:29:32'),",
        "(2, 'https://ok.example/it''s?q=a\\'b', '2.html', 0, '2021-10-31 16:35:38');",
        "INSERT INTO `index` (`rec_id`, `url`, `website`, `result`, `created_date`) VALUES",
        "(3, 'https://ok.example/c,d)', '3.html', 0, '2021-10-31 16:35:39');",
        "COMMIT;",
    ])
    frame = D.parse_index_sql(text)
    assert frame["rec_id"].tolist() == ["1", "2", "3"]
    assert frame["url"].tolist() == ["http://bad.example/a.php", "https://ok.example/it's?q=a'b",
                                     "https://ok.example/c,d)"]
    assert frame["result"].tolist() == ["1", "0", "0"]
    with pytest.raises(ValueError, match="could not be parsed"):
        D.parse_index_sql("(4, 'broken row")


@pytest.mark.parametrize(
    "url, kind",
    [
        ("https://www.example.com/robots.txt", "robots.txt"),
        ("https://www.example.com/ROBOTS.TXT?x=1", "robots.txt"),
        ("https://cdn.example.com/app.min.js", "other non-page resource"),
        ("https://example.com/img/logo.PNG", "other non-page resource"),
        ("https://example.com/sitemap.xml", "other non-page resource"),
        ("https://example.com/", "page"),
        ("https://example.com/about", "page"),
        ("https://example.com/index.html", "page"),
        ("https://example.com/report.pdf", "page"),
        ("https://example.com/v1.2/docs", "page"),
        ("", "page"),
    ],
)
def test_resource_kind(url, kind):
    assert D.resource_kind(url) == kind


# ------------------------------------------------------------------ deduplication

def test_deduplicate_drops_copies_and_conflicts():
    frame = pd.DataFrame(
        {"url": ["https://a.example", "https://a.example", "https://b.example",
                 "https://c.example", "https://c.example", "https://d.example"],
         "label": [0, 0, 1, 0, 1, 1]}
    )
    out, report = D.deduplicate(frame)
    assert out["url"].tolist() == ["https://a.example", "https://b.example", "https://d.example"]
    assert report["duplicate_rows_dropped"] == 1
    assert report["duplicate_rows_dropped_by_class"] == {"legitimate": 1, "phishing": 0}
    assert report["conflicting_urls"] == 1 and report["conflicting_rows_dropped"] == 2
    assert report["rows_out"] == 3
    assert out["url"].is_unique


def test_deduplicate_tie_break_keeps_earliest_date():
    frame = pd.DataFrame(
        {"url": ["https://a.example"] * 3, "label": [1, 1, 1],
         "date": pd.to_datetime(["2025-06-01", "2025-01-01", "2025-03-01"])}
    )
    out, _ = D.deduplicate(frame, tie_break=("date",))
    assert out["date"].iloc[0] == pd.Timestamp("2025-01-01")


# ------------------------------------------------------------------ cross-dataset removal

def test_cross_dataset_overlap_is_removed_on_the_loose_key():
    primary = pd.DataFrame(
        {"url": ["https://www.shared.example", "http://also.example/x", "https://keep.example",
                 "https://keep.example/other"],
         "label": [0, 1, 0, 1]}
    )
    secondary = pd.DataFrame(
        {"url": ["http://shared.example/", "http://also.example/x", "https://unrelated.example"],
         "label": [1, 1, 0]}
    )
    out, report = D.remove_cross_dataset_overlap(primary, secondary)
    assert out["url"].tolist() == ["https://keep.example", "https://keep.example/other"]
    assert report["exact_matches"] == 1
    assert report["loose_matches"] == 2 and report["removed"] == 2
    assert report["removed_with_different_label"] == 1
    assert report["removed_by_class"] == {"legitimate": 1, "phishing": 1}
    remaining = set(out["url"].map(F.canonical_key))
    assert remaining.isdisjoint(set(secondary["url"].map(F.canonical_key)))


# ------------------------------------------------------------------ split and sample

def test_split_is_70_15_15_stratified_and_disjoint():
    frame = _harmonized(600, 400)
    parts = D.split_70_15_15(frame, seed=42)
    assert [len(parts[k]) for k in ("train", "val", "test")] == [700, 150, 150]
    for part in parts.values():
        assert part["label"].mean() == pytest.approx(0.4, abs=0.005)
    urls = [set(parts[k]["url"]) for k in ("train", "val", "test")]
    assert urls[0].isdisjoint(urls[1]) and urls[0].isdisjoint(urls[2]) and urls[1].isdisjoint(urls[2])
    assert set.union(*urls) == set(frame["url"])


def test_split_is_reproducible_and_independent_of_row_order():
    frame = _harmonized(600, 400)
    a = D.split_70_15_15(frame, seed=42)
    b = D.split_70_15_15(frame.sample(frac=1.0, random_state=7), seed=42)
    for name in ("train", "val", "test"):
        assert a[name]["url"].tolist() == b[name]["url"].tolist()
    c = D.split_70_15_15(frame, seed=43)
    assert a["test"]["url"].tolist() != c["test"]["url"].tolist()


# ------------------------------------------------------------------ PhreshPhish guard

def test_only_test_split_files_are_accepted():
    D.assert_test_split_only(
        ["datasets/phreshphish/phreshphish@abc/data/test-00000-of-00021.parquet"]
    )
    for bad in (
        ["datasets/phreshphish/phreshphish@abc/data/train-00000-of-00056.parquet"],
        ["data/test-00000-of-00021.parquet", "data/train-00000-of-00056.parquet"],
        ["data/test-00000.parquet.train"],
        [],
    ):
        with pytest.raises(ValueError):
            D.assert_test_split_only(bad)


def test_loader_has_no_way_to_ask_for_another_split():
    import inspect

    assert D.PHRESHPHISH_TEST_GLOB == "data/test-*.parquet"
    assert set(inspect.signature(D.load_phreshphish_test_raw).parameters) == {"raw_dir", "refresh"}
    assert "html" not in D.PHRESHPHISH_COLUMNS


def test_train_loader_accepts_only_train_split_files():
    import inspect

    D.assert_train_split_only(["datasets/phreshphish/phreshphish@abc/data/train-000.parquet"])
    for bad in (
        ["datasets/phreshphish/phreshphish@abc/data/test-000.parquet"],
        ["data/train-000.parquet", "data/test-000.parquet"],
        ["data/train-000.parquet.test"],
        [],
    ):
        with pytest.raises(ValueError):
            D.assert_train_split_only(bad)
    assert D.PHRESHPHISH_TRAIN_GLOB == "data/train-*.parquet"
    assert set(inspect.signature(D.load_phreshphish_train_raw).parameters) == {"raw_dir", "refresh"}


def test_each_phreshphish_loader_is_wired_to_its_own_split(monkeypatch):
    calls = []
    monkeypatch.setattr(D, "_load_phreshphish_split", lambda *args: calls.append(args) or (None, {}))
    D.load_phreshphish_test_raw()
    D.load_phreshphish_train_raw()
    assert calls[0][:3] == ("test", D.PHRESHPHISH_TEST_GLOB, D.assert_test_split_only)
    assert calls[1][:3] == ("train", D.PHRESHPHISH_TRAIN_GLOB, D.assert_train_split_only)
    assert calls[0][3] != calls[1][3] and calls[0][4] != calls[1][4]     # separate cache files


# ------------------------------------------------------------------ temporal split, pinned sample

def _dated(n_days=20, per_day=10):
    rows = [
        (f"https://site{d}-{i}.example/p", int(i % 2), pd.Timestamp("2025-01-01") + pd.Timedelta(days=d))
        for d in range(n_days) for i in range(per_day)
    ]
    return pd.DataFrame(rows, columns=["url", "label", "date"])


def test_temporal_split_holds_out_the_most_recent_whole_days():
    frame = _dated(20, 10)
    parts, report = D.temporal_split(frame.sample(frac=1.0, random_state=1), val_share=0.15)
    assert len(parts["val"]) == 30 and len(parts["train"]) == 170       # 3 whole days
    assert report["cutoff_date"] == "2025-01-18" and report["val_share"] == 0.15
    assert parts["train"]["date"].max() < parts["val"]["date"].min()
    assert set(parts["train"]["url"]).isdisjoint(parts["val"]["url"])
    assert report["train"]["date_max"] == "2025-01-17" and report["val"]["date_min"] == "2025-01-18"
    assert report["val"]["legitimate"] + report["val"]["phishing"] == 30


def test_temporal_split_never_divides_a_day():
    frame = _dated(4, 10)                       # each day is 25% of the rows
    parts, report = D.temporal_split(frame, val_share=0.15)
    assert len(parts["val"]) == 10 and report["val_share"] == 0.25
    assert parts["val"]["date"].nunique() == 1


def test_temporal_split_rejects_missing_dates_and_bad_shares():
    frame = _dated(4, 2)
    with pytest.raises(ValueError):
        D.temporal_split(frame.assign(date=pd.NaT))
    with pytest.raises(ValueError):
        D.temporal_split(frame, val_share=1.0)


def test_select_by_ids_returns_exactly_the_pinned_rows():
    frame = pd.DataFrame({"url": ["https://c.example/", "https://a.example/", "https://b.example/"],
                          "label": [0, 1, 0], "sha256": ["x", "y", "z"]})
    out = D.select_by_ids(frame, ["z", "y"])
    assert out["url"].tolist() == ["https://a.example/", "https://b.example/"]
    for bad in (["y", "missing"], ["y", "y"]):
        with pytest.raises(ValueError):
            D.select_by_ids(frame, bad)


def test_pinned_evaluation_ids_are_50000_distinct_hashes():
    ids = D.load_eval_ids()
    assert len(ids) == len(set(ids)) == 50_000
    assert all(len(i) == 64 and set(i) <= set("0123456789abcdef") for i in ids)
    assert ids == sorted(ids)


def test_download_refuses_a_file_with_the_wrong_hash(tmp_path):
    spec = {"key": "k", "filename": "f.csv", "dataset": "d", "file_id": "i", "sha256": "0" * 64}
    (tmp_path / "k").mkdir()
    (tmp_path / "k" / "f.csv").write_text("URL,Label\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        D.download_mendeley_file(spec, raw_dir=tmp_path)       # file exists, so no network is used
    good = dict(spec, sha256=D.sha256_file(tmp_path / "k" / "f.csv"))
    assert D.download_mendeley_file(good, raw_dir=tmp_path).name == "f.csv"


def test_secondary_label_map_and_documented_counts():
    assert D.WANGCHUK_LABEL_MAP == {"1": 1, "0": 0}
    assert D.MENDELEY_WANGCHUK["documented_counts"] == {0: 94_919, 1: 54_807}
    assert D.raw_scheme("HTTPS://a.example") == "https"
    assert D.raw_scheme(" http://a.example") == "http"
    assert D.raw_scheme("a.example/x") == "none" and D.raw_scheme(None) == "none"


# ------------------------------------------------------------------ saving and helpers

def test_save_processed_writes_parquet_and_a_stable_manifest(tmp_path):
    parts = D.split_70_15_15(_harmonized(60, 40))
    first = D.save_processed(parts, tmp_path / "a", extra={"seed": 42})
    second = D.save_processed(parts, tmp_path / "b", extra={"seed": 42})
    assert first == second
    assert first["files"]["train"]["rows"] == 70
    assert first["files"]["train"]["legitimate"] + first["files"]["train"]["phishing"] == 70
    assert D.load_manifest(tmp_path / "a") == json.loads(json.dumps(first))
    back = pd.read_parquet(tmp_path / "a" / "train.parquet")
    assert back["url"].tolist() == parts["train"]["url"].tolist()
    table = D.split_counts_table(first)
    assert table.loc["train.parquet", "rows"] == 70


def test_defang_makes_urls_unclickable():
    assert D.defang("https://www.example.com/a.html") == "hxxps://www[.]example[.]com/a[.]html"
    assert D.defang("HTTP://example.com") == "hxxp://example[.]com"
    assert D.defang("example.com") == "example[.]com"
    assert D.defang(None) == ""


def test_profile_and_the_homepage_template_flag():
    frame = _harmonized(60, 40)
    profiled = D.profile(frame)
    assert set(F.FEATURE_NAMES) <= set(profiled.columns)
    flagged = D.add_homepage_template(profiled)
    assert flagged.loc[flagged["label"] == 0, "homepage_template"].all()
    assert not flagged.loc[flagged["label"] == 1, "homepage_template"].any()


def test_profile_reads_the_shipped_spelling_from_url_raw():
    raw = ["https://www.a.example", "https://www.a.example/", "a.example", "http://a.example?q=1"]
    frame = pd.DataFrame({"url": [F.normalize_url(u) for u in raw], "url_raw": raw, "label": 0, "source": "x"})
    profiled = D.profile(frame)
    assert profiled["resource_kind"].tolist() == ["page"] * 4
    assert profiled["raw_ends_at_host"].tolist() == [True, False, True, False]
    assert profiled["raw_scheme"].tolist() == ["https", "https", "none", "http"]
    assert profiled["is_bare_homepage"].tolist() == [True, True, True, False]
    assert profiled["path_len"].tolist() == [1, 1, 1, 1]


def test_homepage_template_requires_https_www_and_nothing_after_the_host():
    urls = ["https://www.a.example", "https://www.a.example/", "http://www.a.example",
            "https://a.example", "https://www.a.example/x", "https://www.a.example?q=1"]
    frame = pd.DataFrame({"url": urls, "label": 0, "source": "x"})
    flagged = D.add_homepage_template(D.profile(frame))
    assert flagged["homepage_template"].tolist() == [True, False, False, False, False, False]
    assert flagged["homepage_template_any_slash"].tolist() == [True, True, False, False, False, False]
    assert flagged["nothing_after_host"].tolist() == [True, False, True, True, False, False]


def test_markdown_table():
    text = D.markdown_table([["a|b", 1]], ["x", "y"])
    assert text.splitlines() == ["| x | y |", "|---|---|", "| a\\|b | 1 |"]


def test_set_seeds_makes_numpy_repeatable():
    import numpy as np

    D.set_seeds(42)
    first = np.random.rand(3)
    D.set_seeds(42)
    assert (np.random.rand(3) == first).all()


# ------------------------------------------------------------------ saved files

PROCESSED = D.PROCESSED_DIR


def _processed_is_current() -> bool:
    path = PROCESSED / "manifest.json"
    return path.exists() and json.loads(path.read_text(encoding="utf-8")).get("design") == D.DESIGN


needs_processed = pytest.mark.skipif(
    not _processed_is_current(),
    reason="data/processed/ is missing or was built for an earlier design (run notebooks/01_data.ipynb)",
)


@pytest.fixture(scope="module")
def saved():
    manifest = D.load_manifest(PROCESSED)
    frames = {name: pd.read_parquet(PROCESSED / info["file"]) for name, info in manifest["files"].items()}
    return manifest, frames


EXPECTED_FILES = {
    "phreshphish_train", "phreshphish_val", "phreshphish_eval", "phreshphish_test_full",
    "wangchuk", "ariyadasa", "phiusiil_train", "phiusiil_val", "phiusiil_test",
}
#: The pinned evaluation sample and the full test split keep every row, so a URL
#: may repeat there once its query is removed. Every other file has unique URLs.
KEEPS_EVERY_ROW = {"phreshphish_eval", "phreshphish_test_full"}


@needs_processed
def test_saved_files_match_the_manifest(saved):
    manifest, frames = saved
    assert set(frames) == EXPECTED_FILES
    for name, info in manifest["files"].items():
        frame = frames[name]
        assert len(frame) == info["rows"]
        assert D.class_counts(frame) == {"legitimate": info["legitimate"], "phishing": info["phishing"]}
        assert D.content_hash(frame) == info["content_sha256"]
        assert set(frame["label"].unique()) == {0, 1}
        assert (frame["url"] != "").all()
        assert frame["url"].is_unique or name in KEEPS_EVERY_ROW, name


@needs_processed
def test_saved_urls_are_main_track_and_already_normalized(saved):
    _, frames = saved
    for name, frame in frames.items():
        sample = frame.head(3000)
        assert (sample["url"].map(lambda u: F.normalize_url(u, strip_query=True)) == sample["url"]).all(), name
        assert not sample["url"].str.contains(r"[?#]").any(), name
        assert (sample["url_full"].map(F.normalize_url) == sample["url_full"]).all(), name
        assert (sample["url_full"].map(lambda u: F.normalize_url(u, strip_query=True)) == sample["url"]).all(), name


@needs_processed
def test_training_ends_before_validation_and_validation_before_the_test_split(saved):
    _, frames = saved
    train, val, test = frames["phreshphish_train"], frames["phreshphish_val"], frames["phreshphish_test_full"]
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()
    assert frames["phreshphish_eval"]["date"].min() >= test["date"].min()
    share = len(val) / (len(train) + len(val))
    assert 0.15 <= share + 0.001 and share < 0.17


@needs_processed
def test_no_training_or_validation_url_appears_in_any_evaluation_set(saved):
    _, frames = saved
    fit_keys = set(frames["phreshphish_train"]["url"].map(F.canonical_key))
    val_keys = set(frames["phreshphish_val"]["url"].map(F.canonical_key))
    assert fit_keys.isdisjoint(val_keys)
    for name in ("phreshphish_test_full", "phreshphish_eval", "wangchuk", "ariyadasa"):
        keys = set(frames[name]["url"].map(F.canonical_key))
        assert keys.isdisjoint(fit_keys), name
        assert keys.isdisjoint(val_keys), name


@needs_processed
def test_eval_sample_is_exactly_the_pinned_rows(saved):
    manifest, frames = saved
    sample = frames["phreshphish_eval"]
    assert len(sample) == 50_000
    assert sorted(sample["sha256"]) == D.load_eval_ids()
    assert D.class_counts(sample) == {"legitimate": 27_149, "phishing": 22_851}
    assert set(sample["sha256"]) <= set(frames["phreshphish_test_full"]["sha256"])
    assert manifest["phreshphish"]["revision"] == D.PHRESHPHISH_REVISION


@needs_processed
def test_phiusiil_baseline_split_is_70_15_15_and_clear_of_the_test_split(saved):
    _, frames = saved
    train, val, test = (frames[f"phiusiil_{k}"] for k in ("train", "val", "test"))
    total = len(train) + len(val) + len(test)
    assert len(train) / total == pytest.approx(0.70, abs=0.001)
    assert len(val) / total == pytest.approx(0.15, abs=0.001)
    a, b, c = set(train["url"]), set(val["url"]), set(test["url"])
    assert a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c)
    test_keys = set(frames["phreshphish_test_full"]["url"].map(F.canonical_key))
    assert set(train["url"].map(F.canonical_key)).isdisjoint(test_keys)


@needs_processed
def test_saved_audits_agree_with_the_saved_files(saved):
    from src import audit as A

    _, frames = saved
    data_audit = A.read_json(A.REPORTS_DIR / "data_audit.json")
    assert data_audit["gate1"]["passed"] is True
    assert data_audit["splits"]["train"]["rows"] == len(frames["phreshphish_train"])
    assert data_audit["splits"]["val"]["rows"] == len(frames["phreshphish_val"])
    assert data_audit["feature_pattern_scan"]["main_track"]["flagged"] == []
    assert data_audit["feature_pattern_scan"]["full_url"]["flagged_besides_query_features"] == []
    for key, name in (("wangchuk", "secondary_audit.json"), ("ariyadasa", "secondary_audit_ariyadasa.json")):
        audit = A.read_json(A.REPORTS_DIR / name)
        assert audit["gate2"]["passed"] is True and audit["decision"]["accepted"] is True
        assert audit["class_counts"]["final_after_overlap_removal"] == D.class_counts(frames[key])
