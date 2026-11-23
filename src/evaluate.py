"""Scoring frozen models and measuring them, with bootstrap intervals.

Everything here runs after the models are frozen. Nothing is fitted or tuned:
thresholds and calibrators are read from `models/*_operating.json`, which were
built from validation data only.

    load_scorers()        the two models, the comparison model and the two baselines
    score_frame()         raw scores of every scorer for one dataset
    evaluate_scores()     all metrics for one scorer on one dataset, with 95% intervals
    fpr_by_domain()       false-positive rate per URL and averaged per registered domain
    robustness()          recall against URL mutation strength

Metric definitions
    PR-AUC, ROC-AUC             threshold-free ranking quality
    detection at 0.1% FPR       best detection rate this dataset allows at a 0.1% false-positive rate
    FPR at 95% detection        false-positive rate needed to catch 95% of this dataset's phishing
    at the frozen thresholds    precision, recall (detection), F1 and false-positive rate at the
                                "block" and "warn" thresholds chosen on validation
    precision at a base rate    what precision would be if phishing were 1% or 0.1% of traffic,
                                from the measured detection and false-positive rates

Intervals are 95% percentile intervals from 1,000 bootstrap resamples of the
rows, the same resamples for every scorer on a dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src import data as D
from src import features as F
from src import models as M
from src import style as S

REPORTS_DIR = D.ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
BASE_RATES = (0.01, 0.001)
TARGET_FPR = 0.001
TARGET_TPR = 0.95

#: name -> (label shown in tables and charts, role)
SCORERS = {
    "lgbm": ("Model A: LightGBM", "model"),
    "cnn": ("Model B: character CNN", "model"),
    "lgbm_full_urls": ("Comparison: LightGBM on full URLs", "comparison"),
    "lgbm_phiusiil": ("Baseline: LightGBM trained on PhiUSIIL", "baseline"),
    "template_rule": ("Baseline: one-line template rule", "baseline"),
}


# --------------------------------------------------------------------------- #
# Scorers
# --------------------------------------------------------------------------- #

def template_rule_raw(urls: Sequence[str]) -> np.ndarray:
    """The one-line rule: legitimate (0) if the URL is `https://www.<host>/`, otherwise phishing (1)."""
    out = np.ones(len(urls), dtype=np.float64)
    for i, url in enumerate(urls):
        parts = F.split_url(url)
        if parts.scheme == "https" and parts.host.startswith("www.") and parts.path in ("", "/") \
                and not parts.query and not parts.fragment:
            out[i] = 0.0
    return out


#: The rule has one operating point, so both thresholds sit between its two outputs.
TEMPLATE_RULE_OPERATING = {
    "model": "template_rule", "track": "main",
    "thresholds": {"block": {"raw_score": 0.5}, "warn": {"raw_score": 0.5}},
    "calibration": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
}


def load_scorers(models_dir: Path = M.MODELS_DIR) -> dict:
    """Load every frozen scorer: a function from a frame to raw scores, and its operating file."""
    models_dir = Path(models_dir)
    boosters = {name: M.load_lgbm(models_dir / f"{name}.txt") for name in ("lgbm", "lgbm_full_urls", "lgbm_phiusiil")}
    session = M.onnx_session(models_dir / "cnn.onnx")
    functions: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
        "lgbm": lambda df: M.lgbm_raw(boosters["lgbm"], M.lexical_features(df, "main")),
        "cnn": lambda df: M.onnx_raw(session, M.char_ids(df, "main")),
        "lgbm_full_urls": lambda df: M.lgbm_raw(boosters["lgbm_full_urls"], M.lexical_features(df, "full")),
        "lgbm_phiusiil": lambda df: M.lgbm_raw(boosters["lgbm_phiusiil"], M.lexical_features(df, "main")),
        "template_rule": lambda df: template_rule_raw(df["url"].tolist()),
    }
    operating = {name: M.load_json(models_dir / f"{name}_operating.json") for name in boosters}
    operating["cnn"] = M.load_json(models_dir / "cnn_operating.json")
    operating["template_rule"] = TEMPLATE_RULE_OPERATING
    return {name: {"raw": functions[name], "operating": operating[name], "label": SCORERS[name][0],
                   "role": SCORERS[name][1]} for name in SCORERS}


def score_frame(df: pd.DataFrame, scorers: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    """Raw scores of every scorer for every row of `df` (one column per scorer)."""
    return pd.DataFrame({name: scorer["raw"](df) for name, scorer in scorers.items()}, index=df.index)


def add_groups(df: pd.DataFrame, train_domains: set) -> pd.DataFrame:
    """The columns the credibility splits and the sensitivity result need.

    bare_homepage   the main-track URL has no path beyond "/"
    domain_seen     its registered domain occurs in the PhreshPhish training or validation data
    had_query       the URL as shipped had a query string
    resource_kind   "page", "robots.txt" or "other non-page resource"
    """
    hosts = [F.split_url(u).host for u in df["url"]]
    out = pd.DataFrame(index=df.index)
    out["registered_domain"] = [F.registered_domain(h) for h in hosts]
    out["bare_homepage"] = [F.split_url(u).path in ("", "/") for u in df["url"]]
    out["domain_seen"] = out["registered_domain"].isin(train_domains)
    out["had_query"] = df["had_query"].to_numpy(dtype=bool)
    out["resource_kind"] = [D.resource_kind(u) for u in df["url"]]
    return out


def registered_domains(df: pd.DataFrame) -> set:
    return {F.registered_domain(F.split_url(u).host) for u in df["url"]}


# --------------------------------------------------------------------------- #
# Fast weighted curve statistics (one pass per bootstrap resample)
# --------------------------------------------------------------------------- #

class Curve:
    """Pre-sorted scores, so a resample needs only cumulative sums."""

    def __init__(self, y: np.ndarray, raw: np.ndarray, thresholds: Mapping[str, float]):
        order = np.argsort(-raw, kind="mergesort")
        self.order = order
        self.y = y[order].astype(np.float64)
        s = raw[order]
        self.ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]       # last row of each tied group
        distinct = s[self.ends]                                             # descending
        # number of tied groups whose score is >= threshold
        self.k = {name: int(np.searchsorted(-distinct, -t, side="right")) for name, t in thresholds.items()}

    def stats(self, weights: np.ndarray) -> dict:
        w = weights[self.order].astype(np.float64)
        tp = np.cumsum(w * self.y)[self.ends]
        fp = np.cumsum(w * (1.0 - self.y))[self.ends]
        P, N = tp[-1], fp[-1]
        out: dict[str, float] = {"n_pos": float(P), "n_neg": float(N)}
        nan = float("nan")
        if P > 0 and N > 0:
            tpr, fpr = tp / P, fp / N
            out["roc_auc"] = float(np.trapezoid(np.r_[0.0, tpr], np.r_[0.0, fpr]))
            precision = tp / np.maximum(tp + fp, 1e-300)
            out["pr_auc"] = float(np.sum(np.diff(np.r_[0.0, tpr]) * precision))
            ok = fpr <= TARGET_FPR
            out["tpr_at_fpr_0.001"] = float(tpr[ok].max()) if ok.any() else 0.0
            out["fpr_at_tpr_0.95"] = float(fpr[np.argmax(tpr >= TARGET_TPR)])
        else:
            out.update({"roc_auc": nan, "pr_auc": nan, "tpr_at_fpr_0.001": nan, "fpr_at_tpr_0.95": nan})
        for name, k in self.k.items():
            tp_k, fp_k = (tp[k - 1], fp[k - 1]) if k > 0 else (0.0, 0.0)
            recall = tp_k / P if P > 0 else nan
            fpr_k = fp_k / N if N > 0 else nan
            precision_k = tp_k / (tp_k + fp_k) if (tp_k + fp_k) > 0 else nan
            f1 = 2 * tp_k / (2 * tp_k + fp_k + (P - tp_k)) if P > 0 else nan
            out[f"{name}_recall"], out[f"{name}_fpr"] = float(recall), float(fpr_k)
            out[f"{name}_precision"], out[f"{name}_f1"] = float(precision_k), float(f1)
            for rate in BASE_RATES:
                denominator = rate * recall + (1 - rate) * fpr_k
                out[f"{name}_precision_at_base_rate_{rate:g}"] = (
                    float(rate * recall / denominator) if denominator == denominator and denominator > 0 else nan)
        return out


def bootstrap_weights(n: int, n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Row counts for `n_boot` resamples with replacement, as a (n_boot, n) uint8 array."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, n), dtype=np.uint8)
    for b in range(n_boot):
        out[b] = np.bincount(rng.integers(0, n, n), minlength=n).clip(max=255)
    return out


def percentile_interval(values: Sequence[float]) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if len(values) < 10:
        return None, None
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def none_if_nan(value: float) -> float | None:
    return None if value != value else float(value)


def evaluate_scores(y: np.ndarray, raw: np.ndarray, operating: Mapping[str, object],
                    weights: np.ndarray | None = None, mask: np.ndarray | None = None) -> dict:
    """Every metric for one scorer on one set of rows, each with a 95% bootstrap interval.

    `weights` is a `bootstrap_weights` array for the whole dataset. `mask`
    restricts the rows (a credibility split); the same resamples are used, so
    a subgroup's interval reflects resampling the whole dataset.
    Returns `{"rows": ..., "metrics": {name: {"value", "ci_low", "ci_high"}}, "confusion": {...}}`.
    """
    y = np.asarray(y).astype(int)
    raw = np.asarray(raw, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        y, raw = y[mask], raw[mask]
        weights = weights[:, mask] if weights is not None else None
    thresholds = {name: float(operating["thresholds"][name]["raw_score"]) for name in ("block", "warn")}
    result = {"rows": int(len(y)), "legitimate": int((y == 0).sum()), "phishing": int((y == 1).sum())}
    if len(y) == 0:
        return {**result, "metrics": {}, "confusion": {}}
    curve = Curve(y, raw, thresholds)
    point = curve.stats(np.ones(len(y)))
    boots = [curve.stats(w) for w in weights] if weights is not None else []
    metrics = {}
    for name, value in point.items():
        if name in ("n_pos", "n_neg"):
            continue
        low, high = percentile_interval([b[name] for b in boots]) if boots else (None, None)
        metrics[name] = {"value": none_if_nan(value), "ci_low": low, "ci_high": high}
    verdict = M.verdicts(raw, operating["thresholds"])
    confusion = {
        cls: {v: int(((y == label) & (verdict == v)).sum()) for v in ("Allow", "Warn", "Block")}
        for label, cls in D.CLASS_NAMES.items()
    }
    return {**result, "metrics": metrics, "confusion": confusion}


# --------------------------------------------------------------------------- #
# False positives per URL and per registered domain
# --------------------------------------------------------------------------- #

def fpr_by_domain(y: np.ndarray, raw: np.ndarray, domains: Sequence[str], operating: Mapping[str, object],
                  n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED,
                  points: Sequence[str] = ("block", "warn")) -> dict:
    """False-positive rate on the legitimate rows, two ways, with domain-resampled intervals.

    per_url      flagged legitimate URLs / legitimate URLs. Domains with many URLs dominate.
    per_domain   the rate within each registered domain, averaged over domains. Each domain counts once.

    The interval resamples registered domains, not rows, because URLs from one
    domain are not independent: a model tends to get all of them right or wrong together.
    """
    y = np.asarray(y).astype(int)
    legit = y == 0
    frame = pd.DataFrame({"domain": np.asarray(domains, dtype=object)[legit], "raw": np.asarray(raw)[legit]})
    out = {"legitimate_rows": int(legit.sum()), "registered_domains": int(frame["domain"].nunique())}
    rng = np.random.default_rng(seed)
    n_domains = out["registered_domains"]
    draws = rng.integers(0, n_domains, size=(n_boot, n_domains))
    for name in points:
        flagged = frame["raw"] >= float(operating["thresholds"][name]["raw_score"])
        per = flagged.groupby(frame["domain"]).agg(["sum", "size"])
        fp, n = per["sum"].to_numpy(dtype=np.float64), per["size"].to_numpy(dtype=np.float64)
        rate = fp / n
        micro = fp[draws].sum(axis=1) / n[draws].sum(axis=1)
        macro = rate[draws].mean(axis=1)
        out[name] = {
            "per_url": {"value": float(fp.sum() / n.sum()), "ci_low": float(np.percentile(micro, 2.5)),
                        "ci_high": float(np.percentile(micro, 97.5))},
            "per_domain": {"value": float(rate.mean()), "ci_low": float(np.percentile(macro, 2.5)),
                           "ci_high": float(np.percentile(macro, 97.5))},
            "domains_with_any_false_positive": int((fp > 0).sum()),
            "share_of_false_positives_in_top_10_domains":
                float(np.sort(fp)[::-1][:10].sum() / fp.sum()) if fp.sum() else 0.0,
        }
    return out


# --------------------------------------------------------------------------- #
# Running the whole evaluation
# --------------------------------------------------------------------------- #

SPLITS = {
    "url_shape": ("bare_homepage", {"bare homepage": True, "has a path": False}),
    "registered_domain": ("domain_seen", {"seen in training": True, "unseen in training": False}),
    "original_query": ("had_query", {"had a query string": True, "no query string": False}),
}


def evaluate_dataset(df: pd.DataFrame, scores: pd.DataFrame, groups: pd.DataFrame,
                     scorers: Mapping[str, Mapping[str, object]], *, splits: bool = True,
                     domain_fpr: bool = False, pages_only: bool = False) -> dict:
    """All results for one dataset: overall, the three credibility splits, and the optional extras."""
    y = df["label"].to_numpy()
    weights = bootstrap_weights(len(df))
    out = {}
    for name, scorer in scorers.items():
        raw, operating = scores[name].to_numpy(), scorer["operating"]
        entry = {"all": evaluate_scores(y, raw, operating, weights)}
        if splits:
            entry["splits"] = {
                split: {label: evaluate_scores(y, raw, operating, weights, groups[column].to_numpy() == value)
                        for label, value in values.items()}
                for split, (column, values) in SPLITS.items()
            }
        if domain_fpr:
            entry["false_positive_rate_by_domain"] = fpr_by_domain(
                y, raw, groups["registered_domain"].to_numpy(), operating)
        if pages_only:
            entry["sensitivity_pages_only"] = evaluate_scores(
                y, raw, operating, weights, groups["resource_kind"].to_numpy() == "page")
        out[name] = entry
    return out


def robustness(phishing: pd.DataFrame, scorers: Mapping[str, Mapping[str, object]],
               strengths: Sequence[int] = (0, 1, 2, 3, 5, 8), sample: int = 5000, seed: int = BOOTSTRAP_SEED) -> dict:
    """Detection rate of the two models on phishing URLs after 0 to 8 random character edits.

    Uses a random sample of phishing URLs from the evaluation sample. Each URL
    is mutated by `src.mutate.mutate_url`; the frozen thresholds are applied
    unchanged. The interval is a bootstrap over the sampled URLs.
    """
    from src import mutate

    rows = phishing.sample(min(sample, len(phishing)), random_state=seed)
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(rows), size=(N_BOOTSTRAP, len(rows)))
    out = {"sampled_phishing_urls": int(len(rows)), "strengths": list(strengths), "models": {}}
    for name in ("lgbm", "cnn"):
        out["models"][name] = {"label": scorers[name]["label"], "warn": [], "block": []}
    for strength in strengths:
        mutated = mutate.mutate_many(rows["url"].tolist(), strength, seed)
        frame = pd.DataFrame({"url": mutated, "url_full": mutated})
        for name in ("lgbm", "cnn"):
            raw = scorers[name]["raw"](frame)
            for point in ("warn", "block"):
                hit = (raw >= float(scorers[name]["operating"]["thresholds"][point]["raw_score"])).astype(np.float64)
                boot = hit[resamples].mean(axis=1)
                out["models"][name][point].append({
                    "strength": int(strength), "recall": float(hit.mean()),
                    "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
                })
    return out


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def fmt(metric: Mapping[str, float | None] | None, percent: bool = True, digits: int = 1, ci: bool = True) -> str:
    """"93.1% (92.7-93.5)" or "0.981 (0.980-0.982)"; "n/a" when undefined."""
    if not metric or metric.get("value") is None:
        return "n/a"
    scale, unit = (100.0, "%") if percent else (1.0, "")
    text = f"{metric['value'] * scale:.{digits}f}{unit}"
    if ci and metric.get("ci_low") is not None:
        text += f" ({metric['ci_low'] * scale:.{digits}f}-{metric['ci_high'] * scale:.{digits}f})"
    return text


def headline_table(results: Mapping[str, object], datasets: Mapping[str, str],
                   scorers: Sequence[str] = tuple(SCORERS)) -> pd.DataFrame:
    """One row per dataset and scorer with the main metrics, formatted with intervals."""
    rows = []
    for key, title in datasets.items():
        for name in scorers:
            m = results[key][name]["all"]["metrics"]
            rows.append({
                "Dataset": title, "Scorer": SCORERS[name][0],
                "ROC-AUC": fmt(m["roc_auc"], False, 3), "PR-AUC": fmt(m["pr_auc"], False, 3),
                "Detection at 0.1% FPR": fmt(m["tpr_at_fpr_0.001"]),
                "FPR at 95% detection": fmt(m["fpr_at_tpr_0.95"]),
                "Block: detection": fmt(m["block_recall"]), "Block: FPR": fmt(m["block_fpr"], digits=2),
                "Warn: detection": fmt(m["warn_recall"]), "Warn: FPR": fmt(m["warn_fpr"]),
                "Warn: precision": fmt(m["warn_precision"]), "Warn: F1": fmt(m["warn_f1"], False, 3),
            })
    return pd.DataFrame(rows).set_index(["Dataset", "Scorer"])


def split_table(results: Mapping[str, object], dataset: str, split: str,
                scorers: Sequence[str] = ("lgbm", "cnn")) -> pd.DataFrame:
    """A credibility split for one dataset: one row per scorer and subgroup."""
    rows = []
    for name in scorers:
        for label, entry in results[dataset][name]["splits"][split].items():
            m = entry["metrics"]
            rows.append({
                "Scorer": SCORERS[name][0], "Subgroup": label,
                "legitimate": entry["legitimate"], "phishing": entry["phishing"],
                "ROC-AUC": fmt(m.get("roc_auc"), False, 3), "PR-AUC": fmt(m.get("pr_auc"), False, 3),
                "Block: detection": fmt(m.get("block_recall")), "Block: FPR": fmt(m.get("block_fpr"), digits=2),
                "Warn: detection": fmt(m.get("warn_recall")), "Warn: FPR": fmt(m.get("warn_fpr")),
            })
    return pd.DataFrame(rows).set_index(["Scorer", "Subgroup"])


def write_json(obj: Mapping[str, object], path: Path) -> Path:
    return M.save_json(obj, path)


#: The files the main evaluation scores, whose hashes it records. The review
#: thresholds are added later by notebook 05 and are not part of it.
EVALUATED_FILES = (
    "models/cnn.onnx", "models/cnn_operating.json", "models/lgbm.txt", "models/lgbm_full_urls.txt",
    "models/lgbm_full_urls_operating.json", "models/lgbm_operating.json", "models/lgbm_phiusiil.txt",
    "models/lgbm_phiusiil_operating.json", "src/features.py",
)
#: The parts of metrics.json that the post hoc results were computed from.
POST_HOC_DEPENDS_ON = ("scorers", "datasets", "results", "model_hashes")


def write_metrics(metrics: Mapping[str, object], path: Path | None = None) -> tuple[Path, str]:
    """Write the main evaluation to metrics.json without losing valid post hoc results.

    notebooks/05_post_hoc.ipynb adds a `post_hoc` key computed from the main
    evaluation. It is kept when everything it was computed from is unchanged,
    and dropped otherwise, so the file never pairs post hoc numbers with a
    different evaluation. Returns the path and a sentence saying which happened.
    """
    path = Path(path or REPORTS_DIR / "metrics.json")
    new = json.loads(json.dumps(metrics))
    old = M.load_json(path) if path.exists() else {}
    if "post_hoc" not in old:
        status = "No post hoc results were saved; run notebooks/05_post_hoc.ipynb to add them."
    elif all(old.get(key) == new.get(key) for key in POST_HOC_DEPENDS_ON):
        new["post_hoc"] = old["post_hoc"]
        status = "The saved post hoc results were kept: the evaluation they were computed from is unchanged."
    else:
        status = ("The saved post hoc results were dropped because the evaluation changed; "
                  "run notebooks/05_post_hoc.ipynb again.")
    return M.save_json(new, path), status


# --------------------------------------------------------------------------- #
# Charts (PNG, 200 dpi). Colours are the first five slots of the project
# palette in their fixed order; each scorer keeps its colour in every chart.
# --------------------------------------------------------------------------- #

SCORER_COLORS = {"lgbm": "#2a78d6", "cnn": "#eb6834", "lgbm_full_urls": "#1baf7a",
                 "lgbm_phiusiil": "#eda100", "template_rule": "#e87ba4"}
SHORT_LABELS = {"lgbm": "Model A: LightGBM", "cnn": "Model B: CNN", "lgbm_full_urls": "LightGBM, full URLs",
                "lgbm_phiusiil": "LightGBM trained on PhiUSIIL", "template_rule": "Template rule"}


def plot_model_comparison(results: Mapping[str, object], datasets: Mapping[str, str], path: Path,
                          metric: str = "roc_auc", title: str = "ROC-AUC by dataset and model"):
    """Grouped bars: one group per dataset, one bar per scorer, with 95% intervals."""
    import matplotlib.pyplot as plt

    names = list(SCORERS)
    fig, ax = plt.subplots(figsize=(2.75 * len(datasets) + 1.2, 4.6))
    fig.patch.set_facecolor(S.SURFACE)
    S.style_axis(ax)
    width = 0.15
    for g, key in enumerate(datasets):
        for j, name in enumerate(names):
            m = results[key][name]["all"]["metrics"][metric]
            if m["value"] is None:
                continue
            x = g + (j - (len(names) - 1) / 2) * (width + 0.02)
            ax.bar(x, m["value"], width=width, color=SCORER_COLORS[name], label=SHORT_LABELS[name] if g == 0 else None)
            if m["ci_low"] is not None:
                ax.plot([x, x], [m["ci_low"], m["ci_high"]], color=S.INK, linewidth=1)
            # three decimals, as the results and the briefing quote them (0.995, not 0.99)
            ax.text(x, max(m["value"], m["ci_high"] or 0) + 0.012, f"{m['value']:.3f}", ha="center", va="bottom",
                    fontsize=7, color=S.INK_2)
    ax.axhline(0.5, color=S.AXIS, linewidth=1)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(list(datasets.values()), fontsize=9.5, color=S.INK)
    ax.set_xlim(-0.55, len(datasets) - 0.45)
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5\n(chance)", "0.75", "1.0"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=9, labelcolor=S.INK_2)
    ax.set_title(title, loc="left", fontsize=12.5, color=S.INK, fontweight="bold", pad=12)
    fig.tight_layout()
    return S.save_figure(fig, path)


def plot_curves(scored: Mapping[str, tuple], datasets: Mapping[str, str], path: Path):
    """ROC curves (top row, log-scaled false-positive rate) and precision-recall curves (bottom row)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, axes = plt.subplots(2, len(datasets), figsize=(4.5 * len(datasets), 8.2), squeeze=False)
    fig.patch.set_facecolor(S.SURFACE)
    for col, (key, title) in enumerate(datasets.items()):
        y, scores = scored[key]
        top, bottom = axes[0, col], axes[1, col]
        S.style_axis(top, "both"), S.style_axis(bottom, "both")
        for name in SCORERS:
            raw, color = scores[name].to_numpy(), SCORER_COLORS[name]
            if name == "template_rule":                    # one operating point, not a curve
                flagged = raw >= 0.5
                tpr, fpr = flagged[y == 1].mean(), flagged[y == 0].mean()
                precision = (flagged & (y == 1)).sum() / max(flagged.sum(), 1)
                top.plot([max(fpr, 1e-4)], [tpr], "o", ms=8, color=color, mec=S.SURFACE, mew=2,
                         label=SHORT_LABELS[name])
                bottom.plot([tpr], [precision], "o", ms=8, color=color, mec=S.SURFACE, mew=2)
                continue
            fpr, tpr, _ = roc_curve(y, raw)
            top.plot(np.maximum(fpr, 1e-4), tpr, color=color, linewidth=2, label=SHORT_LABELS[name])
            precision, recall, _ = precision_recall_curve(y, raw)
            bottom.plot(recall, precision, color=color, linewidth=2)
        top.set_xscale("log")
        top.set_xlim(1e-4, 1)
        top.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
        top.set_xticklabels(["0.01%", "0.1%", "1%", "10%", "100%"])
        top.axvline(TARGET_FPR, color=S.AXIS, linewidth=1)
        top.set_ylim(0, 1.02), bottom.set_ylim(0, 1.02), bottom.set_xlim(0, 1)
        top.set_title(title, loc="left", fontsize=11, color=S.INK, pad=8)
        top.set_xlabel("False-positive rate (log scale; the line marks the 0.1% target)", fontsize=9, color=S.INK_2)
        bottom.set_xlabel("Detection rate (recall)", fontsize=9, color=S.INK_2)
        for ax in (top, bottom):
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        bottom.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        bottom.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    axes[0, 0].set_ylabel("ROC: detection rate", fontsize=9.5, color=S.INK_2)
    axes[1, 0].set_ylabel("Precision-recall: precision", fontsize=9.5, color=S.INK_2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9.5, labelcolor=S.INK_2)
    fig.suptitle("ROC and precision-recall curves", x=0.01, ha="left", fontsize=13, color=S.INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return S.save_figure(fig, path)


def plot_confusion(results: Mapping[str, object], datasets: Mapping[str, str], path: Path,
                   scorers: Sequence[str] = ("lgbm", "cnn")):
    """Verdict by true class: one panel per model and dataset, shaded by share of the class."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue_ramp", S.BLUE_RAMP)
    verdicts = ("Allow", "Warn", "Block")
    fig, axes = plt.subplots(len(scorers), len(datasets), figsize=(3.9 * len(datasets), 2.55 * len(scorers) + 0.7),
                             squeeze=False)
    fig.patch.set_facecolor(S.SURFACE)
    for r, name in enumerate(scorers):
        for c, (key, title) in enumerate(datasets.items()):
            ax = axes[r, c]
            confusion = results[key][name]["all"]["confusion"]
            counts = np.array([[confusion[cls][v] for v in verdicts] for cls in ("legitimate", "phishing")],
                              dtype=float)
            share = counts / counts.sum(axis=1, keepdims=True)
            ax.imshow(share, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            for i in range(2):
                for j in range(3):
                    ax.text(j, i, f"{share[i, j]:.1%}\n{int(counts[i, j]):,}", ha="center", va="center", fontsize=9,
                            color="#ffffff" if share[i, j] > 0.5 else S.INK)
            ax.set_xticks(range(3))
            ax.set_xticklabels(verdicts, fontsize=9, color=S.INK_2)
            ax.set_yticks(range(2))
            ax.set_yticklabels(["legitimate", "phishing"] if c == 0 else ["", ""], fontsize=9, color=S.INK_2)
            ax.tick_params(length=0)
            for x in (0.5, 1.5):
                ax.axvline(x, color=S.SURFACE, linewidth=2)
            ax.axhline(0.5, color=S.SURFACE, linewidth=2)
            for side in ax.spines.values():
                side.set_visible(False)
            ax.set_title(f"{SHORT_LABELS[name]}\n{title}", loc="left", fontsize=9.5, color=S.INK, pad=6)
    fig.suptitle("Verdict by true class (share of each class, and count)", x=0.01, ha="left", fontsize=12.5,
                 color=S.INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return S.save_figure(fig, path)


def plot_robustness(result: Mapping[str, object], path: Path):
    """Detection rate against the number of random character edits, at both thresholds."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    fig.patch.set_facecolor(S.SURFACE)
    titles = {"warn": "Warned or blocked", "block": "Blocked"}
    positions = range(len(result["strengths"]))
    for ax, point in zip(axes, ("warn", "block")):
        S.style_axis(ax)
        recalls = {name: [p["recall"] for p in series[point]] for name, series in result["models"].items()}
        for name, series in result["models"].items():
            ax.fill_between(positions, [p["ci_low"] for p in series[point]], [p["ci_high"] for p in series[point]],
                            color=SCORER_COLORS[name], alpha=0.10, linewidth=0)
            ax.plot(positions, recalls[name], color=SCORER_COLORS[name], linewidth=2, marker="o", ms=7, mec=S.SURFACE,
                    mew=2, label=SHORT_LABELS[name])
        # End labels: one shared label where the two lines coincide, otherwise one above and one below.
        for i in (0, len(result["strengths"]) - 1):
            values = sorted(((recalls[name][i], name) for name in recalls), reverse=True)
            if values[0][0] - values[-1][0] < 0.04:
                ax.annotate(f"both {values[0][0]:.0%}", (i, values[-1][0]), textcoords="offset points",
                            xytext=(0, -16), ha="center", fontsize=8.5, color=S.INK_2)
            else:
                ax.annotate(f"{values[0][0]:.0%}", (i, values[0][0]), textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=8.5, color=S.INK_2)
                ax.annotate(f"{values[-1][0]:.0%}", (i, values[-1][0]), textcoords="offset points", xytext=(0, -16),
                            ha="center", fontsize=8.5, color=S.INK_2)
        ax.set_xticks(list(positions))
        ax.set_xticklabels([str(s) for s in result["strengths"]])
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Random character edits applied to each phishing URL", fontsize=9, color=S.INK_2)
        ax.set_title(titles[point], loc="left", fontsize=11, color=S.INK, pad=8)
    axes[0].set_ylabel("Share of phishing URLs still caught", fontsize=9, color=S.INK_2)
    axes[0].legend(loc="lower left", frameon=False, fontsize=9, labelcolor=S.INK_2)
    fig.suptitle("Detection after random edits to the URL", x=0.01, ha="left", fontsize=12.5, color=S.INK,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return S.save_figure(fig, path)


def plot_shap_summary(booster, X: np.ndarray, path: Path, max_display: int = 15):
    """SHAP summary (beeswarm) for the LightGBM model: each dot is one URL's contribution from one feature."""
    import matplotlib.pyplot as plt
    import shap
    from matplotlib.colors import LinearSegmentedColormap

    import warnings

    # SHAP warns that its LightGBM output format changed (handled below) and that NumPy's global seed is set
    # (set on purpose by data.set_seeds); neither affects the result.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")
        warnings.filterwarnings("ignore", message="The NumPy global RNG was seeded")
        explainer = shap.TreeExplainer(booster)
        values = explainer.shap_values(X)
        values = values[1] if isinstance(values, list) else values
        cmap = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#c3c2b7", "#e34948"])
        plt.figure(figsize=(8.6, 6.4), facecolor=S.SURFACE)
        shap.summary_plot(values, X, feature_names=list(F.FEATURE_NAMES), max_display=max_display, show=False,
                          cmap=cmap, plot_size=None, color_bar_label="Feature value (low to high)")
    fig, ax = plt.gcf(), plt.gca()
    ax.set_facecolor(S.SURFACE)
    ax.set_xlabel("Contribution to the raw score (positive pushes towards phishing)", fontsize=9.5, color=S.INK_2)
    ax.tick_params(colors=S.INK_2, labelsize=9)
    ax.set_title("What drives Model A's scores (SHAP values)", loc="left", fontsize=12.5, color=S.INK,
                 fontweight="bold", pad=12)
    fig.tight_layout()
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_abs)
    ranking = [{"feature": F.FEATURE_NAMES[i], "mean_abs_shap": float(mean_abs[i])} for i in order]
    return S.save_figure(fig, path), ranking
