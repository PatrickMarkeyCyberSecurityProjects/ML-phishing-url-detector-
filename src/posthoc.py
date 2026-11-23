"""Additions made after the test results had been seen.

Everything here is post hoc: decided on 2026-11-21 after the results in
reports/results.md had been read (docs/design_decisions.md, section 6). Nothing
here changes a model, a calibrator, the frozen block and warn thresholds or
the main evaluation results. The output goes into reports/metrics.json under
the key `post_hoc`.

    warn_on_validation()     the frozen warn threshold's false-positive rate where it was set
    review_point()           the "review" threshold: 1% false positives on validation
    evaluate_review()        block and review metrics for one dataset, with 95% intervals
    targeted_robustness()    detection before and after edits chosen to look legitimate
    plot_targeted()          the chart of that test
    cnn_tuning_check()       decision 24: six CNN settings, validation PR-AUC only

The CNN tuning check (decision 24) was decided later the same day and is run
from notebooks/03_cnn.ipynb. It writes reports/cnn_tuning_check.json, scores no
test set and changes no deployed model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src import data as D
from src import evaluate as E
from src import models as M
from src import mutate
from src import style as S

DECIDED = "2026-11-21, after the test results had been seen (docs/design_decisions.md, section 6)"
REVIEW_FPR = 0.01
MODELS = ("lgbm", "cnn")
POINTS = ("block", "review")
BANDS = ("Allow", "Review", "Block")
REVIEW_FILE = M.MODELS_DIR / "review_operating.json"

#: Decision 24, written in docs/design_decisions.md before it was run.
CNN_CHECK_DECIDED = ("2026-11-21, after the test results had been seen and after the overnight review "
                     "(docs/design_decisions.md, decision 24)")
CNN_CHECK_LEARNING_RATES = (0.0003, 0.001, 0.003)
CNN_CHECK_DROPOUTS = (0.3, 0.5)
CNN_CHECK_AMONG_BEST = 0.005          # within this much validation PR-AUC of the best run
CNN_CHECK_FILE = E.REPORTS_DIR / "cnn_tuning_check.json"


# --------------------------------------------------------------------------- #
# a. The frozen warn threshold on validation
# --------------------------------------------------------------------------- #

def warn_on_validation(metrics: Mapping[str, object]) -> dict:
    """What the frozen warn threshold did on validation, read from the saved main evaluation."""
    out = {}
    for name in MODELS:
        saved = metrics["results"]["phreshphish_val"][name]["all"]
        point = metrics["scorers"][name]["thresholds"]["warn"]
        out[name] = {
            "target_min_detection": point["target_min_tpr"],
            "warn_detection": saved["metrics"]["warn_recall"],
            "warn_fpr": saved["metrics"]["warn_fpr"],
            "legitimate_validation_urls": saved["legitimate"],
            "legitimate_validation_urls_flagged": point["validation_false_positives"],
        }
    return out


# --------------------------------------------------------------------------- #
# b. The review operating point
# --------------------------------------------------------------------------- #

def review_point(raw_val: Sequence[float], y_val: Sequence[int], calibration: Mapping[str, Sequence[float]]) -> dict:
    """The review threshold: the lowest raw score flagging at most 1% of legitimate validation URLs.

    This is the block rule with 1% in place of 0.1%, computed by the same
    function, `src.models.choose_thresholds`. Validation data only.
    """
    point = dict(M.choose_thresholds(raw_val, y_val, block_fpr=REVIEW_FPR)["block"])
    point["calibrated_score"] = float(M.apply_calibration(calibration, [point["raw_score"]])[0])
    return point


def frozen_thresholds_reproduced(raw_val: Sequence[float], y_val: Sequence[int],
                                 frozen: Mapping[str, Mapping[str, float]], tolerance: float = 1e-4) -> dict:
    """Recompute the frozen block and warn thresholds from validation scores and compare.

    A check that the validation scores used for the review point are the ones
    the frozen thresholds were built from.
    """
    again = M.choose_thresholds(raw_val, y_val)
    out = {}
    for name in ("block", "warn"):
        difference = abs(again[name]["raw_score"] - frozen[name]["raw_score"])
        out[name] = {"frozen": float(frozen[name]["raw_score"]), "recomputed": float(again[name]["raw_score"]),
                     "abs_difference": float(difference)}
    out["reproduced"] = bool(all(out[name]["abs_difference"] <= tolerance for name in ("block", "warn")))
    return out


def bands(raw_scores, block: float, review: float) -> np.ndarray:
    """"Block", "Review" or "Allow" for each raw score."""
    s = np.asarray(raw_scores, dtype=np.float64)
    out = np.full(s.shape, "Allow", dtype=object)
    out[s >= review] = "Review"
    out[s >= block] = "Block"
    return out


def evaluate_points(y: np.ndarray, raw: np.ndarray, thresholds: Mapping[str, float],
                    weights: np.ndarray | None = None, mask: np.ndarray | None = None) -> dict:
    """Detection, false-positive rate, precision and F1 at the block and review thresholds.

    Uses the same curve code and the same bootstrap resamples as the main
    evaluation (`src.evaluate`), so the block numbers here equal its block numbers.
    "review" means at or above the review threshold: reviewed or blocked.
    `bands` counts every row once, by true class and band.
    """
    y = np.asarray(y).astype(int)
    raw = np.asarray(raw, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        y, raw = y[mask], raw[mask]
        weights = weights[:, mask] if weights is not None else None
    result = {"rows": int(len(y)), "legitimate": int((y == 0).sum()), "phishing": int((y == 1).sum())}
    if len(y) == 0:
        return {**result, "metrics": {}, "bands": {}}
    thresholds = {name: float(thresholds[name]) for name in POINTS}
    curve = E.Curve(y, raw, thresholds)
    point = curve.stats(np.ones(len(y)))
    boots = [curve.stats(w) for w in weights] if weights is not None else []
    metrics = {}
    for name, value in point.items():
        if not name.startswith(tuple(f"{p}_" for p in POINTS)):
            continue
        low, high = E.percentile_interval([b[name] for b in boots]) if boots else (None, None)
        metrics[name] = {"value": E.none_if_nan(value), "ci_low": low, "ci_high": high}
    band = bands(raw, thresholds["block"], thresholds["review"])
    counts = {cls: {b: int(((y == label) & (band == b)).sum()) for b in BANDS} for label, cls in D.CLASS_NAMES.items()}
    return {**result, "metrics": metrics, "bands": counts}


def evaluate_review(df: pd.DataFrame, scores: Mapping[str, np.ndarray], groups: pd.DataFrame | None,
                    thresholds: Mapping[str, Mapping[str, float]], *, splits: bool = True,
                    domain_fpr: bool = False) -> dict:
    """Block and review results for one dataset and both models: overall, the splits, and by domain."""
    y = df["label"].to_numpy()
    weights = E.bootstrap_weights(len(df))
    out = {}
    for name in MODELS:
        raw = np.asarray(scores[name], dtype=np.float64)
        entry = {"all": evaluate_points(y, raw, thresholds[name], weights)}
        if splits:
            entry["splits"] = {
                split: {label: evaluate_points(y, raw, thresholds[name], weights, groups[column].to_numpy() == value)
                        for label, value in values.items()}
                for split, (column, values) in E.SPLITS.items()
            }
        if domain_fpr:
            operating = {"thresholds": {p: {"raw_score": thresholds[name][p]} for p in POINTS}}
            entry["false_positive_rate_by_domain"] = E.fpr_by_domain(
                y, raw, groups["registered_domain"].to_numpy(), operating, points=POINTS)
        out[name] = entry
    return out


# --------------------------------------------------------------------------- #
# c. Targeted robustness
# --------------------------------------------------------------------------- #

def _rate(hit: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    """Share of rows hit, and the same share in every bootstrap resample."""
    return float(hit.mean()), (weights @ hit) / len(hit)


def _with_interval(value: float, boot: np.ndarray) -> dict:
    return {"value": float(value), "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5))}


def targeted_robustness(phishing: pd.DataFrame, scorers: Mapping[str, Mapping[str, object]],
                        thresholds: Mapping[str, Mapping[str, float]], seed: int = E.BOOTSTRAP_SEED,
                        n_boot: int = E.N_BOOTSTRAP) -> dict:
    """Detection on phishing URLs before and after edits chosen to look legitimate.

    `phishing` holds main-track phishing URLs. Each edit in
    `src.mutate.TARGETED_EDITS` is applied to every URL; both models score the
    result at the block and review thresholds. Intervals are bootstrap
    intervals over the URLs, with the same resamples before and after, so the
    interval of a change is a paired one.
    """
    urls = phishing["url"].tolist()
    n = len(urls)
    weights = E.bootstrap_weights(n, n_boot, seed).astype(np.float64)
    draw = mutate.benign_path_draw(n, seed)
    edited = {edit: mutate.targeted_many(urls, edit, seed) for edit in mutate.TARGETED_EDITS}
    original = np.asarray(urls, dtype=object)

    def raw_scores(name: str, strings: Sequence[str]) -> np.ndarray:
        return scorers[name]["raw"](pd.DataFrame({"url": list(strings), "url_full": list(strings)}))

    out = {
        "decided": DECIDED,
        "phishing_urls": int(n),
        "seed": int(seed),
        "edits": dict(mutate.TARGETED_EDITS),
        "benign_paths": list(mutate.BENIGN_PATHS),
        "urls_changed": {edit: int((np.asarray(strings, dtype=object) != original).sum())
                         for edit, strings in edited.items()},
        "models": {},
        "by_benign_path": {},
    }
    for name in MODELS:
        before_raw = raw_scores(name, urls)
        after_raw = {edit: raw_scores(name, strings) for edit, strings in edited.items()}
        out["models"][name] = {"label": scorers[name]["label"]}
        out["by_benign_path"][name] = {}
        for point in POINTS:
            t = float(thresholds[name][point])
            before_hit = (before_raw >= t).astype(np.float64)
            before, before_boot = _rate(before_hit, weights)
            after = {}
            for edit in mutate.TARGETED_EDITS:
                hit = (after_raw[edit] >= t).astype(np.float64)
                value, boot = _rate(hit, weights)
                after[edit] = {**_with_interval(value, boot),
                               "change": _with_interval(value - before, boot - before_boot)}
            out["models"][name][point] = {"before": _with_interval(before, before_boot), "after": after}
            path_hit = after_raw["path"] >= t
            out["by_benign_path"][name][point] = [
                {"path": path, "urls": int((draw == i).sum()),
                 "before": float(before_hit[draw == i].mean()) if (draw == i).any() else None,
                 "after": float(path_hit[draw == i].mean()) if (draw == i).any() else None}
                for i, path in enumerate(mutate.BENIGN_PATHS)
            ]
    return out


EDIT_LABELS = {"www": "www. prefix\nadded", "https": "Scheme set\nto https", "path": "Benign-looking\npath appended",
               "all": "All three\ntogether"}


def plot_targeted(result: Mapping[str, object], path: Path):
    """Grouped bars: share of phishing URLs still caught, original and after each targeted edit."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), sharey=True)
    fig.patch.set_facecolor(S.SURFACE)
    titles = {"block": "Blocked", "review": "Reviewed or blocked"}
    groups = ["original", *EDIT_LABELS]
    width, gap = 0.36, 0.04
    for ax, point in zip(axes, POINTS):
        S.style_axis(ax)
        for j, name in enumerate(MODELS):
            series = result["models"][name][point]
            for g, key in enumerate(groups):
                m = series["before"] if key == "original" else series["after"][key]
                x = g + (j - 0.5) * (width + gap)
                ax.bar(x, m["value"], width=width, color=E.SCORER_COLORS[name],
                       label=E.SHORT_LABELS[name] if g == 0 else None)
                ax.plot([x, x], [m["ci_low"], m["ci_high"]], color=S.INK, linewidth=1)
                ax.text(x, m["ci_high"] + 0.015, f"{m['value']:.0%}", ha="center", va="bottom", fontsize=8.5,
                        color=S.INK_2)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(["Original\nURL", *EDIT_LABELS.values()], fontsize=9, color=S.INK)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_title(titles[point], loc="left", fontsize=11, color=S.INK, pad=8)
    axes[0].set_ylabel("Share of phishing URLs still caught", fontsize=9, color=S.INK_2)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=S.INK_2, ncol=2)
    fig.suptitle(f"Detection after edits chosen to look legitimate ({result['phishing_urls']:,} phishing URLs, "
                 "later-in-time test; post hoc)", x=0.01, ha="left", fontsize=12.5, color=S.INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return S.save_figure(fig, path)


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #

def save_review_file(points: Mapping[str, Mapping[str, float]], validation_rows: Mapping[str, int],
                     path: Path = REVIEW_FILE) -> Path:
    """Write the review thresholds to their own file. The frozen operating files are not edited."""
    return M.save_json({
        "decided": DECIDED,
        "fitted_on": "validation set only",
        "target_max_fpr": REVIEW_FPR,
        "rule": "Block if raw >= the frozen block threshold; Review if raw >= review.raw_score; otherwise Allow",
        "note": "Calibrated scores are not real-world probabilities.",
        "models": {name: dict(point) for name, point in points.items()},
        "validation_rows": dict(validation_rows),
    }, path)


def model_thresholds(scorers: Mapping[str, Mapping[str, object]], review: Mapping[str, object]) -> dict:
    """{model: {"block": frozen raw threshold, "review": review raw threshold}}."""
    return {name: {"block": float(scorers[name]["operating"]["thresholds"]["block"]["raw_score"]),
                   "review": float(review["models"][name]["raw_score"])} for name in MODELS}


# --------------------------------------------------------------------------- #
# Decision 24. The CNN tuning check
# --------------------------------------------------------------------------- #

def cnn_tuning_check(ids_train: np.ndarray, y_train: np.ndarray, ids_stop: np.ndarray, y_stop: np.ndarray,
                     ids_val: np.ndarray, y_val: np.ndarray, *, device: str,
                     shipped_validation_pr_auc: float, log=print) -> dict:
    """Train the shipped architecture at six settings and measure validation PR-AUC only.

    Post hoc (decision 24). The same fitting data, early-stopping split, seed and
    early stopping as the shipped CNN; only the learning rate and dropout vary.
    No test set is scored, nothing is calibrated, no threshold is set, and no
    model is saved: the deployed CNN is kept whatever the result.
    """
    from sklearn.metrics import average_precision_score

    shipped = (M.CNN_TRAINING["learning_rate"], M.CNN_CONFIG["dropout"])
    runs = []
    for lr in CNN_CHECK_LEARNING_RATES:
        for dropout in CNN_CHECK_DROPOUTS:
            training = {**M.CNN_TRAINING, "learning_rate": lr}
            config = {**M.CNN_CONFIG, "dropout": dropout}
            model, history = M.train_cnn(ids_train, y_train, ids_stop, y_stop, device=device, seed=M.SEED,
                                         training=training, config=config, log=lambda line: None)
            best = max(history, key=lambda row: row["stop_pr_auc"])
            run = {"learning_rate": lr, "dropout": dropout, "shipped_settings": (lr, dropout) == shipped,
                   "epochs_run": len(history), "best_epoch": best["epoch"],
                   "validation_pr_auc": float(average_precision_score(y_val, M.cnn_raw(model, ids_val, device)))}
            runs.append(run)
            log(f"learning rate {lr:<6g} dropout {dropout:.1f}  epochs {run['epochs_run']:2d} "
                f"(best {run['best_epoch']:2d})  validation PR-AUC {run['validation_pr_auc']:.4f}")
    ranked = sorted(runs, key=lambda r: r["validation_pr_auc"], reverse=True)
    for rank, run in enumerate(ranked, start=1):
        run["rank"] = rank
    best_run = ranked[0]
    shipped_run = next(r for r in runs if r["shipped_settings"])
    gap = best_run["validation_pr_auc"] - shipped_run["validation_pr_auc"]
    return {
        "decided": CNN_CHECK_DECIDED,
        "post_hoc": True,
        "what": "Validation PR-AUC of the shipped CNN architecture at six learning-rate and dropout settings. "
                "No test set was scored with these models; the deployed CNN, its calibrator and its thresholds "
                "are unchanged, and none of these models was saved.",
        "grid": {"learning_rate": list(CNN_CHECK_LEARNING_RATES), "dropout": list(CNN_CHECK_DROPOUTS)},
        "fixed": {"architecture": {k: v for k, v in M.CNN_CONFIG.items() if k != "dropout"},
                  "training": {k: v for k, v in M.CNN_TRAINING.items() if k != "learning_rate"}, "seed": M.SEED,
                  "device": device},
        "rows": {"training": int(len(y_train)), "early_stopping": int(len(y_stop)), "validation": int(len(y_val))},
        "shipped": {"learning_rate": shipped[0], "dropout": shipped[1],
                    "validation_pr_auc_saved": float(shipped_validation_pr_auc),
                    "validation_pr_auc_retrained": shipped_run["validation_pr_auc"],
                    "rank": shipped_run["rank"]},
        "best": {k: best_run[k] for k in ("learning_rate", "dropout", "validation_pr_auc")},
        "gap_best_minus_shipped": gap,
        "among_best_rule": f"within {CNN_CHECK_AMONG_BEST} validation PR-AUC of the best run (fixed before running)",
        "shipped_among_best": bool(gap <= CNN_CHECK_AMONG_BEST),
        "runs": runs,
    }
