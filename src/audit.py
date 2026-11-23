"""Dataset audits and the two acceptance gates that run before any model is trained.

    build_data_audit       PhreshPhish train, validation and evaluation sample -> Gate 1
    build_secondary_audit  the unseen secondary dataset                        -> Gate 2
    template_rule_result   the one-line rule on PhiUSIIL  -> reports/phiusiil_template_rule.json

Both return plain dictionaries that are written to reports/ as JSON. Nothing
here fits a model: a "single-feature AUC" uses the raw value of one column as the
score. All gates are direction-free: an AUC of 0.03 separates the classes as
well as one of 0.97, so the gate looks at max(AUC, 1 - AUC).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src import data as D
from src import features as F

REPORTS_DIR = D.ROOT / "reports"

#: A single URL-shape feature that separates the classes this well, in either
#: direction, means the dataset can be solved without learning about phishing.
GATE_AUC = 0.95
#: Gate 2: the secondary's legitimate class must be mostly real pages, not homepages.
GATE_MAX_LEGIT_BARE_HOMEPAGE = 0.50

#: Shape features checked by both gates. `has_path` is "path presence": the URL
#: has a path longer than "/".
GATE1_FEATURES = ("has_path", "path_len", "is_https", "has_www", "url_len")
GATE2_FEATURES = ("has_path", "path_len", "is_https", "has_www")

#: One-sided feature scan: a feature that is non-zero for under 0.5% of one class
#: and over 5% of the other is present in practice for one class only. That is
#: the signature the query string left in PhreshPhish, where the collector
#: removed query parameters from legitimate URLs.
SCAN_RARE = 0.005
SCAN_COMMON = 0.05
QUERY_FEATURES = ("query_len", "n_query_params")


# --------------------------------------------------------------------------- #
# Small measurements
# --------------------------------------------------------------------------- #

def feature_auc(labels: Sequence[int], values: Sequence[float]) -> dict:
    """ROC-AUC of one column used alone as a phishing score, and its direction-free form."""
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(np.asarray(labels), np.asarray(values, dtype=float)))
    return {"auc": round(auc, 6), "direction_free": round(max(auc, 1.0 - auc), 6)}


def shape_auc(profiled: pd.DataFrame, columns: Sequence[str]) -> dict:
    return {col: feature_auc(profiled["label"], profiled[col]) for col in columns}


def split_summary(df: pd.DataFrame) -> dict:
    counts = D.class_counts(df)
    out = {"rows": int(len(df)), **counts,
           "phishing_share": round(counts["phishing"] / len(df), 6) if len(df) else 0.0}
    if "date" in df.columns:
        out.update({k: v for k, v in D.date_range(df).items() if k != "rows"})
    return out


def class_by_month(df: pd.DataFrame) -> dict:
    """Rows per calendar month and class. Shows when each class was collected."""
    table = (df.groupby([df["date"].dt.to_period("M").astype(str), "label"]).size()
             .unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0))
    return {month: {"legitimate": int(row[0]), "phishing": int(row[1])} for month, row in table.iterrows()}


def shape_shares(profiled: pd.DataFrame) -> dict:
    """Per class: how the URLs are shaped. Shares are of the rows in that class."""
    out = {}
    for label, name in D.CLASS_NAMES.items():
        g = profiled[profiled["label"] == label]
        if g.empty:
            continue
        schemes = g["raw_scheme"].value_counts(normalize=True)
        out[name] = {
            "rows": int(len(g)),
            "bare_homepage": round(float(g["is_bare_homepage"].mean()), 6),
            "non_root_path": round(float(g["has_path"].mean()), 6),
            "has_query": round(float(g["has_query"].mean()), 6),
            "robots_txt": round(float(g["resource_kind"].eq("robots.txt").mean()), 6),
            "other_non_page_resource": round(float(g["resource_kind"].eq("other non-page resource").mean()), 6),
            "leading_www": round(float(g["has_www"].mean()), 6),
            "scheme_as_shipped": {str(k): round(float(v), 6) for k, v in schemes.items()},
            "ends_at_host_as_shipped": round(float(g["raw_ends_at_host"].mean()), 6),
            "median_url_len": float(g["url_len"].median()),
            "median_path_len": float(g["path_len"].median()),
        }
    return out


def feature_pattern_scan(frame: pd.DataFrame, strip_query: bool) -> dict:
    """Which of the 31 features are present for one class only?

    For every feature: the share of legitimate and of phishing rows where it is
    non-zero. A feature is flagged when that share is under SCAN_RARE for one
    class and over SCAN_COMMON for the other. The mirrored test (share of rows
    where the feature is zero) is reported too, because for a flag such as
    `is_https` the rare event is the zero.

    `strip_query=True` scans the main track (`url`), False the full URL (`url_full`).
    """
    column = "url" if strip_query else "url_full"
    matrix = F.feature_matrix(frame[column], strip_query=strip_query)
    labels = frame["label"].to_numpy()
    legit, phish = matrix[labels == 0], matrix[labels == 1]
    rows, flagged, flagged_zero = {}, [], []
    for j, name in enumerate(F.FEATURE_NAMES):
        nz_l, nz_p = float((legit[:, j] != 0).mean()), float((phish[:, j] != 0).mean())
        one_sided = min(nz_l, nz_p) < SCAN_RARE and max(nz_l, nz_p) > SCAN_COMMON
        zero_sided = min(1 - nz_l, 1 - nz_p) < SCAN_RARE and max(1 - nz_l, 1 - nz_p) > SCAN_COMMON
        rows[name] = {
            "nonzero_share_legitimate": round(nz_l, 6),
            "nonzero_share_phishing": round(nz_p, 6),
            "flagged_nonzero": bool(one_sided),
            "flagged_zero": bool(zero_sided),
        }
        if one_sided:
            flagged.append(name)
        if zero_sided:
            flagged_zero.append(name)
    return {
        "track": "main (query removed)" if strip_query else "full URL",
        "rows": {"legitimate": int(len(legit)), "phishing": int(len(phish))},
        "rule": f"non-zero for under {SCAN_RARE:.1%} of one class and over {SCAN_COMMON:.0%} of the other",
        "flagged": flagged,
        "flagged_besides_query_features": [f for f in flagged if f not in QUERY_FEATURES],
        "flagged_on_zero_share": flagged_zero,
        "features": rows,
    }


def top_domains(profiled: pd.DataFrame, n: int = 10) -> dict:
    """The `n` most frequent registered domains per class, with their share of the class."""
    out = {}
    for label, name in D.CLASS_NAMES.items():
        g = profiled[profiled["label"] == label]
        counts = g["registered_domain"].value_counts().head(n)
        out[name] = [
            {"registered_domain": str(domain), "rows": int(count), "share": round(count / len(g), 6)}
            for domain, count in counts.items()
        ]
        out[f"{name}_distinct_domains"] = int(g["registered_domain"].nunique())
        out[f"{name}_top{n}_share"] = round(float(counts.sum() / len(g)), 6) if len(g) else 0.0
    return out


def url_overlap(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Rows of `a` whose URL also appears in `b`: exact string, and loose key.

    The loose key (`src.features.canonical_key`) ignores the scheme, a leading
    `www.` and trailing slashes.
    """
    exact = a["url"].isin(set(b["url"]))
    b_keys = dict(zip(b["url"].map(F.canonical_key), b["label"]))
    a_keys = a["url"].map(F.canonical_key)
    loose = a_keys.isin(b_keys.keys())
    return {
        "rows": int(len(a)),
        "exact": int(exact.sum()),
        "loose": int(loose.sum()),
        "loose_by_class": D.class_counts(a[loose]),
        "loose_with_different_label": int((a_keys[loose].map(b_keys) != a.loc[loose, "label"]).sum()),
    }


def domain_overlap_by_class(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Share of `a` rows, per class, whose registered domain occurs in `b`.

    `a` and `b` must be profiled frames. Three reference sets are used: any row
    of `b`, the legitimate rows of `b`, and the phishing rows of `b`.
    """
    reference = {
        "any": set(b["registered_domain"]),
        "legitimate": set(b.loc[b["label"] == 0, "registered_domain"]),
        "phishing": set(b.loc[b["label"] == 1, "registered_domain"]),
    }
    out = {}
    for label, name in D.CLASS_NAMES.items():
        g = a[a["label"] == label]
        out[name] = {"rows": int(len(g)), "distinct_domains": int(g["registered_domain"].nunique())}
        for ref_name, domains in reference.items():
            seen = g["registered_domain"].isin(domains)
            out[name][f"share_with_domain_in_{ref_name}"] = round(float(seen.mean()), 6) if len(g) else 0.0
        seen_any = g["registered_domain"].isin(reference["any"])
        out[name]["distinct_domains_seen"] = int(g.loc[seen_any, "registered_domain"].nunique())
    seen_all = a["registered_domain"].isin(reference["any"])
    out["all"] = {"rows": int(len(a)), "share_with_domain_in_any": round(float(seen_all.mean()), 6)}
    return out


# --------------------------------------------------------------------------- #
# Gate 1: PhreshPhish train, validation and the evaluation sample
# --------------------------------------------------------------------------- #

def temporal_overlap(train_all: pd.DataFrame, test_full: pd.DataFrame) -> dict:
    """Do the dates of the training data reach the dates of the test split?"""
    train_days, test_days = train_all["date"].dt.normalize(), test_full["date"].dt.normalize()
    train_max, test_min = train_days.max(), test_days.min()
    late_train = train_all[train_days >= test_min]
    early_test = test_full[test_days <= train_max]
    shared = sorted(set(train_days.unique()) & set(test_days.unique()))
    return {
        "train_date_min": str(train_days.min().date()),
        "train_date_max": str(train_max.date()),
        "test_date_min": str(test_min.date()),
        "test_date_max": str(test_days.max().date()),
        "overlap": bool(train_max >= test_min),
        "shared_days": [str(pd.Timestamp(d).date()) for d in shared],
        "train_rows_on_or_after_test_min": {"rows": int(len(late_train)), **D.class_counts(late_train)},
        "test_rows_on_or_before_train_max": {"rows": int(len(early_test)), **D.class_counts(early_test)},
    }


def build_data_audit(
    *,
    train: pd.DataFrame,
    val: pd.DataFrame,
    eval_sample: pd.DataFrame,
    test_full: pd.DataFrame,
    reports: Mapping[str, object],
    extra: Mapping[str, object] | None = None,
) -> dict:
    """Audit of the PhreshPhish data, ending in the Gate 1 decision.

    `train` is the fitting part of the train split, `val` its most recent 15%,
    `eval_sample` the pinned 50,000-row sample of the test split and
    `test_full` the whole deduplicated test split. The first three must be
    profiled frames (`src.data.profile`). `reports` carries the step reports
    from harmonizing, deduplicating, overlap removal and the temporal split.
    `extra` is added at the top level (the feature scan, the collapse report).
    """
    train_all = pd.concat([train, val], ignore_index=True)
    parts = {"train": train, "val": val, "eval_sample": eval_sample}

    temporal = temporal_overlap(train_all, test_full)
    auc = {name: shape_auc(part, GATE1_FEATURES) for name, part in parts.items()}
    worst = max(
        ((name, col, auc[name][col]["direction_free"]) for name in auc for col in GATE1_FEATURES),
        key=lambda item: item[2],
    )
    reasons = []
    if temporal["overlap"]:
        late = temporal["train_rows_on_or_after_test_min"]["rows"]
        reasons.append(
            f"train dates reach the test split: latest train date {temporal['train_date_max']}, "
            f"earliest test date {temporal['test_date_min']} ({late:,} train rows on or after it)"
        )
    if worst[2] >= GATE_AUC:
        reasons.append(f"{worst[1]} alone reaches AUC {worst[2]:.3f} (direction-free) on {worst[0]}")

    return {
        "label_convention": "1 = phishing, 0 = legitimate",
        "roles": {
            "train": "PhreshPhish train split, rows before the validation cutoff: model fitting",
            "val": "PhreshPhish train split, most recent days: early stopping, calibration, thresholds",
            "eval_sample": "pinned 50,000-row sample of the PhreshPhish test split: scored once, at the end",
        },
        "splits": {
            **{name: split_summary(part) for name, part in parts.items()},
            "test_full": split_summary(test_full),
        },
        "steps": dict(reports),
        "temporal": temporal,
        "class_by_month": {"train_and_val": class_by_month(train_all), "eval_sample": class_by_month(eval_sample)},
        "url_overlap_after_removal": {
            "train_vs_test_full": url_overlap(train, test_full),
            "val_vs_test_full": url_overlap(val, test_full),
            "val_vs_train": url_overlap(val, train),
        },
        "registered_domain_overlap": {
            "note": "share of evaluation-sample rows whose registered domain occurs in train + val; "
                    "the registered domain is approximated without a public-suffix list",
            "eval_sample_vs_train_and_val": domain_overlap_by_class(eval_sample, train_all),
            "val_vs_train": domain_overlap_by_class(val, train),
        },
        "shape_by_class": {name: shape_shares(part) for name, part in parts.items()},
        "bare_homepage_share": {
            name: {cls: shares["bare_homepage"] for cls, shares in shape_shares(part).items()}
            for name, part in parts.items()
        },
        "single_feature_auc": auc,
        "all_feature_auc_train_only": dict(sorted(
            shape_auc(train, F.FEATURE_NAMES).items(), key=lambda kv: -kv[1]["direction_free"]
        )),
        "gate1": {
            "auc_threshold_direction_free": GATE_AUC,
            "features_checked": list(GATE1_FEATURES),
            "checked_on": list(parts),
            "date_overlap": temporal["overlap"],
            "highest_single_feature_auc": {"split": worst[0], "feature": worst[1], "direction_free": worst[2]},
            "passed": not reasons,
            "reasons": reasons,
        },
        **dict(extra or {}),
    }


# --------------------------------------------------------------------------- #
# Gate 2: the secondary dataset
# --------------------------------------------------------------------------- #

def build_secondary_audit(
    *,
    spec: Mapping[str, object],
    source: Mapping[str, object],
    harmonize_report: Mapping[str, object],
    dedup_report: Mapping[str, object],
    removal_report: Mapping[str, object],
    as_shipped: pd.DataFrame,
    deduplicated: pd.DataFrame,
    final: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    eval_sample: pd.DataFrame,
    phiusiil_train: pd.DataFrame | None = None,
    phishing_provenance: Mapping[str, object] | None = None,
    description: Mapping[str, object] | None = None,
) -> dict:
    """Audit of a candidate secondary dataset, ending in the Gate 2 decision.

    `as_shipped` is every harmonized row, `deduplicated` one row per URL, and
    `final` the deduplicated rows left after removing every URL that matches
    PhreshPhish train or validation. All three, and `train`, `val` and
    `eval_sample`, must be profiled frames. The gate is applied to `final`.
    """
    train_all = pd.concat([train, val], ignore_index=True)
    documented = {int(k): int(v) for k, v in dict(spec.get("documented_counts", {})).items()}
    loaded = {int(k): int(v) for k, v in harmonize_report["class_counts_after_mapping"].items()}

    features = list(GATE2_FEATURES)
    auc_final = shape_auc(final, features)
    auc_shipped = shape_auc(as_shipped, features)
    # `is_https` after normalization treats a URL shipped without a scheme as
    # https. This version uses only the scheme as written.
    for frame, target in ((final, auc_final), (as_shipped, auc_shipped)):
        target["is_https_as_shipped"] = feature_auc(frame["label"], frame["raw_scheme"].eq("https"))
    worst = max(auc_final.items(), key=lambda kv: kv[1]["direction_free"])

    shares_final = shape_shares(final)
    legit_bare = shares_final["legitimate"]["bare_homepage"]
    recoverable = (
        int(source["empty_urls"]) == 0
        and set(source["raw_label_counts"]) == set(source["label_map"])
        and (not documented or loaded == documented)
    )
    reasons = []
    if not recoverable:
        reasons.append("raw URLs or labels are not recoverable as documented")
    if legit_bare >= GATE_MAX_LEGIT_BARE_HOMEPAGE:
        reasons.append(f"{legit_bare:.1%} of legitimate URLs are bare homepages")
    if worst[1]["direction_free"] >= GATE_AUC:
        reasons.append(f"{worst[0]} alone reaches AUC {worst[1]['direction_free']:.3f} (direction-free)")

    audit = {
        "description": dict(description) if description else None,
        "dataset": {
            "name": spec["name"], "doi": spec["doi"], "published": spec.get("published"),
            "licence": spec.get("licence"), "file": spec["filename"],
            "file_sha256": source["file_sha256"], "file_bytes": source["file_bytes"],
            "sha256_matches_published": source["file_sha256"] == spec["sha256"],
            "columns": source["columns"], "url_column": source["url_column"],
            "label_column": source["label_column"],
            "label_direction": source["label_direction"],
            "label_map_applied": source["label_map"],
            "legitimate_class_name": "presumed benign",
        },
        "class_counts": {
            "documented": {"legitimate": documented.get(0), "phishing": documented.get(1)},
            "loaded": {"legitimate": loaded.get(0), "phishing": loaded.get(1)},
            "match": bool(documented) and loaded == documented,
            "after_deduplication": D.class_counts(deduplicated),
            "final_after_overlap_removal": D.class_counts(final),
        },
        "duplicates": {
            "rows_as_shipped": int(len(as_shipped)),
            "exact_duplicate_rows_as_shipped": int(as_shipped["url_raw"].duplicated().sum()),
            "duplicate_rows_after_normalization": int(dedup_report["duplicate_rows_dropped"]),
            "duplicate_rows_dropped_by_class": dedup_report.get("duplicate_rows_dropped_by_class"),
            "urls_with_conflicting_labels": int(dedup_report["conflicting_urls"]),
            "conflicting_rows_dropped": int(dedup_report["conflicting_rows_dropped"]),
            "rows_after_deduplication": int(len(deduplicated)),
        },
        "normalization": {k: harmonize_report[k] for k in (
            "urls_changed", "n_whitespace_stripped", "n_scheme_added", "n_host_lowercased",
            "n_punycode_decoded", "n_root_slash_added", "scheme_added_by_class",
            "root_slash_added_by_class", "strip_query", "had_query_by_class",
            "had_query_share_by_class", "query_removed_by_class")},
        "dates": D.date_range(final) if "date" in final.columns else None,
        "non_page_resources": {
            "note": "from the URL path: robots.txt, or a file extension in src.data.NON_PAGE_EXTENSIONS",
            "as_shipped": resource_shares(as_shipped),
            "final": resource_shares(final),
        },
        "shape_by_class": {"as_shipped": shape_shares(as_shipped), "final": shares_final},
        "top_registered_domains": {"as_shipped": top_domains(as_shipped), "final": top_domains(final)},
        "single_feature_auc": {"as_shipped": auc_shipped, "final": auc_final},
        "overlap_with_phreshphish": {
            "note": "computed on the deduplicated secondary, before removal; every loose match "
                    "with train or validation is then removed",
            "vs_train": url_overlap(deduplicated, train),
            "vs_val": url_overlap(deduplicated, val),
            "vs_train_and_val": url_overlap(deduplicated, train_all),
            "removed": dict(removal_report),
            "remaining_after_removal": url_overlap(final, train_all),
            "registered_domain_vs_train": domain_overlap_by_class(final, train),
            "registered_domain_vs_val": domain_overlap_by_class(final, val),
            "registered_domain_vs_train_and_val": domain_overlap_by_class(final, train_all),
            "vs_eval_sample_count_only": url_overlap(final, eval_sample),
        },
        "phishing_provenance": dict(phishing_provenance) if phishing_provenance else None,
        "gate2": {
            "applied_to": "final (deduplicated, PhreshPhish train and validation URLs removed)",
            "criteria": {
                "raw_urls_and_labels_recoverable": recoverable,
                "legitimate_bare_homepage_share": legit_bare,
                "legitimate_bare_homepage_limit": GATE_MAX_LEGIT_BARE_HOMEPAGE,
                "highest_single_feature_auc": {"feature": worst[0], **worst[1]},
                "auc_threshold_direction_free": GATE_AUC,
            },
            "passed": not reasons,
            "reasons": reasons,
        },
    }
    if phiusiil_train is not None:
        audit["overlap_with_phiusiil_train_count_only"] = url_overlap(final, phiusiil_train)
    return audit


def resource_shares(profiled: pd.DataFrame) -> dict:
    """Per class: rows and share that are robots.txt files or other non-page resources."""
    out = {}
    for label, name in D.CLASS_NAMES.items():
        g = profiled[profiled["label"] == label]
        robots = int(g["resource_kind"].eq("robots.txt").sum())
        other = int(g["resource_kind"].eq("other non-page resource").sum())
        out[name] = {
            "rows": int(len(g)),
            "robots_txt": robots,
            "robots_txt_share": round(robots / len(g), 6) if len(g) else 0.0,
            "other_non_page_resource": other,
            "other_non_page_resource_share": round(other / len(g), 6) if len(g) else 0.0,
            "pages": int(len(g) - robots - other),
        }
    return out


def phishing_provenance(secondary: pd.DataFrame, reference: pd.DataFrame, spec: Mapping[str, object]) -> dict:
    """Is the secondary's phishing class the same list as an earlier published one?

    Compares the phishing URLs of `secondary` with `reference` (an older
    phishing-only file), as shipped and after normalization. If they are the
    same list, the secondary's phishing URLs cannot be newer than the
    reference's publication date.
    """
    phish = secondary[secondary["label"] == 1]
    ours_raw, theirs_raw = set(phish["url_raw"]), set(reference["url_raw"])
    # Compare full normalized URLs: the reference keeps its query strings.
    column = "url_full" if "url_full" in phish.columns else "url"
    ours, theirs = set(phish[column]), set(reference["url"])
    share = len(ours & theirs) / len(ours) if ours else 0.0
    same = share >= 0.999 and len(ours) == len(theirs)
    return {
        "reference": spec["name"], "reference_doi": spec["doi"], "reference_published": spec.get("published"),
        "reference_rows": int(len(reference)), "reference_distinct_urls": len(theirs_raw),
        "secondary_phishing_rows": int(len(phish)), "secondary_phishing_distinct_urls": len(ours_raw),
        "secondary_phishing_distinct_urls_normalized": len(ours),
        "distinct_urls_in_both_as_shipped": len(ours_raw & theirs_raw),
        "distinct_urls_in_both_normalized": len(ours & theirs),
        "share_of_secondary_phishing_urls_found_in_reference": round(share, 6),
        "same_list": bool(same),
        "conclusion": (
            f"The secondary's phishing class is the list published in {spec['doi']} on "
            f"{spec.get('published')}, so its phishing URLs predate that publication (April 2024)."
            if same else "The phishing class is not the same list as the reference."
        ),
    }


# --------------------------------------------------------------------------- #
# Running the audits on built data (called by notebooks/01_data.ipynb)
# --------------------------------------------------------------------------- #

#: How each accepted external dataset is described everywhere in the project.
SECONDARY_DESCRIPTIONS = {
    "wangchuk": {
        "role": "unseen secondary dataset (headline external test)",
        "description": "A different-source test. Its phishing URLs predate the training data, and its "
                       "presumed-benign class covers 547 registered domains.",
        "audit_file": "secondary_audit.json",
    },
    "ariyadasa": {
        "role": "second external check",
        "description": "A different-source test collected from December 2020 to November 2021, about three "
                       "years before the training data begins. Its presumed-benign class comes from Google "
                       "search results and an older phishing-research dataset.",
        "audit_file": "secondary_audit_ariyadasa.json",
    },
}
FROZEN_ON = "2026-11-21"


def run_data_audit(phreshphish, profiles: Mapping[str, pd.DataFrame]) -> dict:
    """Gate 1 audit from a `src.pipeline.build_phreshphish` result and its profiled frames."""
    frames = phreshphish.frames
    extra = {
        "main_track": "query string and fragment removed from every URL before deduplication, "
                      "overlap removal, training and scoring",
        "eval_sample_collapse": phreshphish.reports["eval_collapse"],
        "query_strings": {
            "note": "share of rows whose shipped URL had a query string",
            "train_split_as_shipped": phreshphish.reports["train_harmonize"]["had_query_share_by_class"],
            "test_split_as_shipped": phreshphish.reports["test_harmonize"]["had_query_share_by_class"],
        },
        "feature_pattern_scan": {
            "scanned_on": "train (fitting part)",
            "full_url": feature_pattern_scan(frames["train"], strip_query=False),
            "main_track": feature_pattern_scan(frames["train"], strip_query=True),
        },
    }
    return build_data_audit(
        train=profiles["train"], val=profiles["val"], eval_sample=profiles["eval"],
        test_full=frames["test_full"], reports=phreshphish.reports, extra=extra,
    )


def run_secondary_audit(key: str, secondary, profiles: Mapping[str, pd.DataFrame],
                        phiusiil_train: pd.DataFrame | None = None,
                        provenance: Mapping[str, object] | None = None) -> dict:
    """Gate 2 audit from a `src.pipeline.build_secondary` result. Adds the frozen decision."""
    audit = build_secondary_audit(
        spec=secondary.source["spec"], source=secondary.source["file"],
        harmonize_report=secondary.reports["harmonize"], dedup_report=secondary.reports["deduplicate"],
        removal_report=secondary.reports["removal_vs_train_and_val"],
        as_shipped=D.profile(secondary.frames["as_shipped"]),
        deduplicated=D.profile(secondary.frames["deduplicated"]),
        final=D.profile(secondary.frames["final"]),
        train=profiles["train"], val=profiles["val"], eval_sample=profiles["eval"],
        phiusiil_train=phiusiil_train, phishing_provenance=provenance,
        description=SECONDARY_DESCRIPTIONS[key],
    )
    audit["decision"] = {
        "accepted": bool(audit["gate2"]["passed"]),
        "frozen_on": FROZEN_ON,
        "frozen_before_any_model_was_trained": True,
        "use": "evaluation only: scored once with frozen models; nothing is fitted or tuned on it",
    }
    return audit


def gate_summary(audit: Mapping[str, object]) -> pd.DataFrame:
    """The Gate 2 criteria of a secondary audit as a small table."""
    c = audit["gate2"]["criteria"]
    top = c["highest_single_feature_auc"]
    rows = [
        ["Raw URLs and labels recoverable, class counts as documented",
         "yes" if c["raw_urls_and_labels_recoverable"] else "NO",
         "must be yes"],
        ["Presumed-benign URLs that are bare homepages", f"{c['legitimate_bare_homepage_share']:.1%}",
         f"under {c['legitimate_bare_homepage_limit']:.0%}"],
        [f"Highest single shape-feature AUC, direction-free ({top['feature']})", f"{top['direction_free']:.3f}",
         f"under {c['auc_threshold_direction_free']}"],
        ["Gate 2", "PASSED" if audit["gate2"]["passed"] else "FAILED: " + "; ".join(audit["gate2"]["reasons"]), ""],
    ]
    return pd.DataFrame(rows, columns=["criterion", "value", "requirement"]).set_index("criterion")


TEMPLATE_RULE_FILE = REPORTS_DIR / "phiusiil_template_rule.json"
TEMPLATE_RULE = "Legitimate if the URL is exactly https://www.<host> as shipped, otherwise phishing"


def template_rule_result(profiled: pd.DataFrame) -> dict:
    """The one-line template rule on every PhiUSIIL row, with the counts behind its accuracy.

    `profiled` is the harmonized PhiUSIIL frame after `D.add_homepage_template(D.profile(...))`:
    label 1 = phishing, 0 = legitimate, and `homepage_template` true when the URL as
    shipped is exactly `https://www.<host>`. No learning is involved.
    """
    says_legitimate = profiled["homepage_template"].astype(bool).to_numpy()
    legitimate = (profiled["label"] == 0).to_numpy()
    counts = {
        "legitimate_called_legitimate": int((legitimate & says_legitimate).sum()),
        "legitimate_called_phishing": int((legitimate & ~says_legitimate).sum()),
        "phishing_called_phishing": int((~legitimate & ~says_legitimate).sum()),
        "phishing_called_legitimate": int((~legitimate & says_legitimate).sum()),
    }
    correct = counts["legitimate_called_legitimate"] + counts["phishing_called_phishing"]
    return {
        "dataset": "PhiUSIIL (UCI id 967), every row as shipped, labels flipped to 1 = phishing",
        "rule": TEMPLATE_RULE,
        "learning": "none",
        "rows": int(len(profiled)),
        "legitimate": int(legitimate.sum()),
        "phishing": int((~legitimate).sum()),
        "counts": counts,
        "correct": int(correct),
        "accuracy": correct / len(profiled),
        "written_by": "notebooks/01_data.ipynb, section 1",
    }


def write_json(obj: Mapping[str, object], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
