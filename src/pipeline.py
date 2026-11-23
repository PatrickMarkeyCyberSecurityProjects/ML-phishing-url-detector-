"""The data build, start to finish. notebooks/01_data.ipynb runs these steps and shows the results.

Every step is a function in `src.data`; this module only fixes their order, so
the notebook, the tests and a rerun all build the same files.

    build_phreshphish()   train, validation, the pinned evaluation sample, the full test split
    build_secondary()     an external dataset: as shipped, deduplicated, and final
    build_phiusiil()      the cautionary baseline's own 70/15/15 split

All frames are in the main track: `url` has the query string and fragment
removed, and deduplication and overlap removal work on that form.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from src import data as D
from src import features as F

STRIP = F.MAIN_STRIP_QUERY


class Built(NamedTuple):
    frames: dict
    reports: dict
    source: dict


def build_phreshphish() -> Built:
    """PhreshPhish: training data from the train split, evaluation data from the test split.

    Order of steps for the train split:
      1. harmonize (main track);
      2. drop rows dated on or after the first test date, so training ends
         strictly before the test split begins;
      3. deduplicate, keeping the earliest sighting of each URL;
      4. remove every URL that loosely matches a URL anywhere in the test split;
      5. hold out the most recent days (at least 15% of rows) as validation;
      6. remove from validation any URL that loosely matches a training URL.

    The evaluation sample is the 50,000 test rows pinned by `sha256`. Its
    membership is kept as it is: rows that become the same string once the
    query is removed are counted in `eval_collapse`, not dropped.
    """
    te_raw, te_src = D.load_phreshphish_test_raw()
    te, te_h = D.harmonize(
        te_raw, url_col="url", label_col="label", label_map=D.PHRESHPHISH_LABEL_MAP, source="PhreshPhish",
        expected_counts=D.PHRESHPHISH_EXPECTED_COUNTS, keep=("date", "sha256"), strip_query=STRIP,
    )
    te = te.sort_values(["url", "date", "sha256"], kind="mergesort").reset_index(drop=True)
    ev = D.select_by_ids(te, D.load_eval_ids())

    tr_raw, tr_src = D.load_phreshphish_train_raw()
    tr, tr_h = D.harmonize(
        tr_raw, url_col="url", label_col="label", label_map=D.PHRESHPHISH_LABEL_MAP, source="PhreshPhish",
        expected_counts=D.PHRESHPHISH_TRAIN_EXPECTED_COUNTS, keep=("date", "sha256"), strip_query=STRIP,
    )
    tr_early, boundary = D.drop_on_or_after(tr, te["date"].min())
    tr_dedup, tr_d = D.deduplicate(tr_early, tie_break=("date", "sha256"))
    tr_clean, tr_x = D.remove_cross_dataset_overlap(tr_dedup, te)
    parts, split_report = D.temporal_split(tr_clean)
    val, val_x = D.remove_cross_dataset_overlap(parts["val"], parts["train"])

    frames = {"train": parts["train"], "val": val, "eval": ev, "test_full": te}
    reports = {
        "test_harmonize": te_h,
        "eval_collapse": D.collapse_report(ev),
        "test_full_collapse": D.collapse_report(te),
        "train_harmonize": tr_h,
        "train_boundary_drop": boundary,
        "train_deduplicate": tr_d,
        "train_removal_vs_full_test": tr_x,
        "validation_split": split_report,
        "validation_removal_vs_train": val_x,
    }
    return Built(frames, reports, {"test": te_src, "train": tr_src})


def _secondary_raw(key: str):
    if key == "wangchuk":
        raw, source = D.load_wangchuk_raw()
        return raw, source, D.MENDELEY_WANGCHUK, D.WANGCHUK_URL_COL, D.WANGCHUK_LABEL_COL, D.WANGCHUK_LABEL_MAP, ()
    if key == "ariyadasa":
        raw, source = D.load_ariyadasa_raw()
        return (raw, source, D.MENDELEY_ARIYADASA, D.ARIYADASA_URL_COL, D.ARIYADASA_LABEL_COL,
                D.ARIYADASA_LABEL_MAP, ("date",))
    raise ValueError(f"Unknown secondary dataset: {key!r}")


def build_secondary(key: str, train_all: pd.DataFrame) -> Built:
    """An external dataset, for evaluation only.

    Harmonize (main track), deduplicate, then remove every URL that loosely
    matches a PhreshPhish training or validation URL (`train_all`).
    Returns frames `as_shipped`, `deduplicated` and `final`.
    """
    raw, source, spec, url_col, label_col, label_map, keep = _secondary_raw(key)
    name = {"wangchuk": "Wangchuk", "ariyadasa": "Ariyadasa"}[key]
    shipped, h = D.harmonize(
        raw, url_col=url_col, label_col=label_col, label_map=label_map, source=name,
        expected_counts=spec["documented_counts"], keep=keep, strip_query=STRIP,
    )
    dedup, d = D.deduplicate(shipped, tie_break=keep)
    final, x = D.remove_cross_dataset_overlap(dedup, train_all)
    frames = {"as_shipped": shipped, "deduplicated": dedup, "final": final}
    reports = {"harmonize": h, "deduplicate": d, "removal_vs_train_and_val": x}
    return Built(frames, reports, {"file": source, "spec": spec})


def build_phiusiil(test_full: pd.DataFrame) -> Built:
    """PhiUSIIL, kept only as a cautionary baseline: its own 70/15/15 split.

    Harmonize (main track), deduplicate, remove every URL that loosely matches
    the PhreshPhish test split, then split at random, stratified, seed 42.
    """
    raw, source = D.load_phiusiil_raw()
    phi, h = D.harmonize(
        raw, url_col=D.PHIUSIIL_URL_COL, label_col=D.PHIUSIIL_LABEL_COL, label_map=D.PHIUSIIL_LABEL_MAP,
        source="PhiUSIIL", expected_counts=D.PHIUSIIL_EXPECTED_COUNTS, strip_query=STRIP,
    )
    dedup, d = D.deduplicate(phi)
    clean, x = D.remove_cross_dataset_overlap(dedup, test_full)
    frames = D.split_70_15_15(clean, seed=D.SEED)
    reports = {"harmonize": h, "deduplicate": d, "removal_vs_phreshphish_test": x}
    return Built(frames, reports, source)
