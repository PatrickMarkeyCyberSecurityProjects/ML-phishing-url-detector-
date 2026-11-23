"""Training, calibration, thresholds and export for both models.

    Model A   LightGBM on the 31 lexical features of `src.features`
    Model B   character-level CNN (PyTorch), exported to ONNX

Every model outputs a *raw score*: the log-odds (margin) before the sigmoid.
Thresholds are set on the raw score, and an isotonic calibrator maps it to the
score shown to a user. Both are fitted on the validation set only and saved as
JSON, so the app applies them with NumPy and needs neither scikit-learn nor torch.

A calibrated score is not a real-world probability: the class mix of the
validation set was set by how the data was collected (docs/design_decisions.md).

torch is imported only inside the CNN functions, so the LightGBM half of this
module works without it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src import features as F

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
SEED = 42

#: The two operating points, fixed before training (docs/problem_and_threat_model.md).
BLOCK_FPR = 0.001      # "block": at most 0.1% of legitimate validation URLs are blocked
WARN_TPR = 0.95        # "warn": at least 95% of phishing validation URLs get a warning or a block

TRACKS = {"main": True, "full": False}     # track name -> strip_query


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def track_urls(df: pd.DataFrame, track: str) -> pd.Series:
    """The URL column a track reads: `url` (query removed) or `url_full`."""
    return df["url"] if TRACKS[track] else df["url_full"]


def lexical_features(df: pd.DataFrame, track: str = "main") -> np.ndarray:
    """The (n, 31) feature matrix of a harmonized frame for one track."""
    return F.feature_matrix(track_urls(df, track), strip_query=TRACKS[track])


def char_ids(df: pd.DataFrame, track: str = "main", max_len: int = F.DEFAULT_MAX_LEN,
             chunk: int = 50_000) -> np.ndarray:
    """Character ids of a harmonized frame as a uint8 array of shape (n, max_len).

    The vocabulary has 97 ids, so uint8 holds them; the model casts to int64.
    """
    urls = track_urls(df, track).tolist()
    out = np.zeros((len(urls), max_len), dtype=np.uint8)
    for start in range(0, len(urls), chunk):
        out[start:start + chunk] = F.encode_batch(urls[start:start + chunk], max_len, TRACKS[track])
    return out


# --------------------------------------------------------------------------- #
# Calibration and thresholds (NumPy at prediction time)
# --------------------------------------------------------------------------- #

def fit_isotonic(raw_scores: Sequence[float], labels: Sequence[int]) -> dict:
    """Fit an isotonic calibrator and return it as breakpoints.

    The result `{"x": [...], "y": [...]}` reproduces scikit-learn's prediction
    with `np.interp(raw, x, y)`, which is what `apply_calibration` does.
    """
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(raw_scores, dtype=np.float64), np.asarray(labels, dtype=np.float64))
    return {"x": [float(v) for v in iso.X_thresholds_], "y": [float(v) for v in iso.y_thresholds_]}


def apply_calibration(calibration: Mapping[str, Sequence[float]], raw_scores) -> np.ndarray:
    """Map raw scores to calibrated scores in [0, 1] by linear interpolation."""
    return np.interp(np.asarray(raw_scores, dtype=np.float64), calibration["x"], calibration["y"])


def choose_thresholds(raw_scores: Sequence[float], labels: Sequence[int],
                      block_fpr: float = BLOCK_FPR, warn_tpr: float = WARN_TPR) -> dict:
    """Choose the "block" and "warn" thresholds on raw scores. Validation data only.

    block   the lowest threshold whose false-positive rate is at most `block_fpr`
    warn    the highest threshold whose detection rate is at least `warn_tpr`

    A URL is blocked when its raw score is >= block, warned when it is >= warn
    (and below block), and allowed otherwise. If the model is strong enough
    that warn would lie above block, the warn band is empty.
    """
    from sklearn.metrics import roc_curve

    y = np.asarray(labels).astype(int)
    s = np.asarray(raw_scores, dtype=np.float64)
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    n_neg, n_pos = int((y == 0).sum()), int((y == 1).sum())

    # roc_curve lists thresholds from high to low; point 0 is "nothing flagged".
    ok = np.flatnonzero(fpr <= block_fpr)
    i_block = int(ok[-1])
    if i_block == 0:
        raise ValueError("No threshold meets the block false-positive target.")
    i_warn = int(np.flatnonzero(tpr >= warn_tpr)[0])

    def point(i: int, target_name: str, target: float) -> dict:
        return {
            "raw_score": float(thr[i]),
            target_name: target,
            "validation_fpr": float(fpr[i]),
            "validation_tpr": float(tpr[i]),
            "validation_false_positives": int(round(fpr[i] * n_neg)),
            "validation_true_positives": int(round(tpr[i] * n_pos)),
        }

    return {
        "block": point(i_block, "target_max_fpr", block_fpr),
        "warn": point(i_warn, "target_min_tpr", warn_tpr),
        "validation_rows": {"legitimate": n_neg, "phishing": n_pos},
        "rule": "Block if raw >= block.raw_score; Warn if raw >= warn.raw_score; otherwise Allow",
    }


def verdicts(raw_scores, thresholds: Mapping[str, Mapping[str, float]]) -> np.ndarray:
    """"Block", "Warn" or "Allow" for each raw score."""
    s = np.asarray(raw_scores, dtype=np.float64)
    out = np.full(s.shape, "Allow", dtype=object)
    out[s >= thresholds["warn"]["raw_score"]] = "Warn"
    out[s >= thresholds["block"]["raw_score"]] = "Block"
    return out


def build_operating_file(name: str, track: str, raw_val, y_val, extra: Mapping[str, object] | None = None) -> dict:
    """Calibrator and thresholds for one model, from validation scores."""
    calibration = fit_isotonic(raw_val, y_val)
    thresholds = choose_thresholds(raw_val, y_val)
    for point in ("block", "warn"):
        thresholds[point]["calibrated_score"] = float(
            apply_calibration(calibration, [thresholds[point]["raw_score"]])[0]
        )
    return {
        "model": name,
        "track": track,
        "strip_query": TRACKS[track],
        "raw_score": "log-odds (margin) before the sigmoid",
        "fitted_on": "validation set only",
        "note": "Calibrated scores are not real-world probabilities: the validation class mix "
                "was set by how the data was collected.",
        "thresholds": thresholds,
        "calibration": calibration,
        **dict(extra or {}),
    }


def save_json(obj: Mapping[str, object], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Model A: LightGBM
# --------------------------------------------------------------------------- #

LGBM_FIXED = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "bagging_freq": 1,
    "seed": SEED,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 8,
    "verbosity": -1,
}
LGBM_MAX_ROUNDS = 1500
LGBM_EARLY_STOPPING = 50


def _lgbm_params(trial_params: Mapping[str, object], y: np.ndarray) -> dict:
    """Fixed settings, tuned settings and the class weight for the rows being fitted."""
    n_pos = int(y.sum())
    return {**LGBM_FIXED, **trial_params, "scale_pos_weight": float((len(y) - n_pos) / max(n_pos, 1))}


def tune_lgbm(X: np.ndarray, y: np.ndarray, *, n_trials: int = 20, n_folds: int = 3,
              sample_size: int = 100_000, seed: int = SEED) -> dict:
    """Tune LightGBM with Optuna: `n_trials` trials of `n_folds`-fold cross-validation.

    Runs on a stratified sample of `sample_size` rows of the fitting data. Each
    fold trains with early stopping on its own held-out part, so tuning never
    sees the validation set. The objective is the mean PR-AUC (average
    precision) across folds. Class weights rebalance the classes.

    Returns the best parameters, the number of boosting rounds (mean best
    iteration of the best trial) and a table of all trials.
    """
    import lightgbm as lgb
    import optuna
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold, train_test_split

    y = np.asarray(y).astype(int)
    if sample_size < len(y):
        idx, _ = train_test_split(np.arange(len(y)), train_size=sample_size, stratify=y, random_state=seed)
        idx = np.sort(idx)
    else:
        idx = np.arange(len(y))
    Xs, ys = X[idx], y[idx]
    folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(Xs, ys))

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }
        scores, rounds = [], []
        for train_idx, held_idx in folds:
            booster = lgb.train(
                _lgbm_params(params, ys[train_idx]),
                lgb.Dataset(Xs[train_idx], ys[train_idx], feature_name=list(F.FEATURE_NAMES)),
                num_boost_round=LGBM_MAX_ROUNDS,
                valid_sets=[lgb.Dataset(Xs[held_idx], ys[held_idx])],
                feval=lambda pred, data: ("ap", average_precision_score(data.get_label(), pred), True),
                callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING, first_metric_only=True, verbose=False)],
            )
            rounds.append(int(booster.best_iteration))
            scores.append(float(average_precision_score(
                ys[held_idx], booster.predict(Xs[held_idx], num_iteration=booster.best_iteration, raw_score=True))))
        trial.set_user_attr("rounds", rounds)
        trial.set_user_attr("fold_scores", scores)
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    trials = pd.DataFrame([
        {"trial": t.number, "cv_pr_auc": t.value, "mean_rounds": float(np.mean(t.user_attrs["rounds"])), **t.params}
        for t in study.trials
    ])
    return {
        "params": dict(best.params),
        "n_rounds": int(round(float(np.mean(best.user_attrs["rounds"])))),
        "cv_pr_auc": float(best.value),
        "cv_fold_pr_auc": [float(v) for v in best.user_attrs["fold_scores"]],
        "n_trials": n_trials, "n_folds": n_folds, "sample_rows": int(len(ys)),
        "trials": trials,
    }


def fit_lgbm(X: np.ndarray, y: np.ndarray, params: Mapping[str, object], n_rounds: int):
    """Fit the final LightGBM on all the given rows, with class weights."""
    import lightgbm as lgb

    y = np.asarray(y).astype(int)
    return lgb.train(
        _lgbm_params(params, y),
        lgb.Dataset(X, y, feature_name=list(F.FEATURE_NAMES)),
        num_boost_round=int(n_rounds),
    )


def lgbm_raw(booster, X: np.ndarray) -> np.ndarray:
    """Raw scores (log-odds) from a LightGBM booster."""
    return np.asarray(booster.predict(X, raw_score=True), dtype=np.float64)


def load_lgbm(path: Path):
    import lightgbm as lgb

    return lgb.Booster(model_file=str(path))


def train_lgbm_model(name: str, track: str, fit: pd.DataFrame, val: pd.DataFrame,
                     models_dir: Path = MODELS_DIR) -> dict:
    """Tune, fit, calibrate and save one LightGBM. Returns a summary for the notebook.

    Writes `<name>.txt` (the model) and `<name>_operating.json` (calibrator and
    thresholds, from `val` only).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    started = time.time()
    X_fit, y_fit = lexical_features(fit, track), fit["label"].to_numpy()
    X_val, y_val = lexical_features(val, track), val["label"].to_numpy()
    tuned = tune_lgbm(X_fit, y_fit)
    booster = fit_lgbm(X_fit, y_fit, tuned["params"], tuned["n_rounds"])
    raw_val = lgbm_raw(booster, X_val)

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{name}.txt"
    booster.save_model(str(model_path))
    training = {
        "algorithm": "LightGBM gradient-boosted trees",
        "features": list(F.FEATURE_NAMES),
        "fit_rows": int(len(y_fit)), "fit_phishing": int(y_fit.sum()),
        "tuned_params": tuned["params"], "n_rounds": tuned["n_rounds"],
        "fixed_params": {k: v for k, v in LGBM_FIXED.items()},
        "class_weight_scale_pos_weight": float((len(y_fit) - y_fit.sum()) / y_fit.sum()),
        "tuning": {"trials": tuned["n_trials"], "folds": tuned["n_folds"], "sample_rows": tuned["sample_rows"],
                   "objective": "mean PR-AUC across folds", "best_cv_pr_auc": tuned["cv_pr_auc"],
                   "best_cv_fold_pr_auc": tuned["cv_fold_pr_auc"]},
        "validation_roc_auc": float(roc_auc_score(y_val, raw_val)),
        "validation_pr_auc": float(average_precision_score(y_val, raw_val)),
        "seed": SEED,
    }
    operating = build_operating_file(name, track, raw_val, y_val, {"training": training})
    save_json(operating, models_dir / f"{name}_operating.json")
    return {"name": name, "track": track, "booster": booster, "operating": operating, "trials": tuned["trials"],
            "raw_val": raw_val, "y_val": y_val, "seconds": time.time() - started}


# --------------------------------------------------------------------------- #
# Model B: character-level CNN
# --------------------------------------------------------------------------- #

CNN_CONFIG = {
    "vocab_size": F.VOCAB_SIZE, "max_len": F.DEFAULT_MAX_LEN, "embedding_dim": 32,
    "kernel_widths": [3, 5, 7], "filters": 128, "dense": 64, "dropout": 0.3,
}
CNN_TRAINING = {
    "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 0.01, "batch_size": 512,
    "max_epochs": 30, "patience": 3, "early_stopping_metric": "PR-AUC on the early-stopping split",
    "early_stopping_share": 0.05,
}


def build_cnn(config: Mapping[str, object] = CNN_CONFIG):
    """The character CNN: embedding -> three parallel Conv1D -> global max-pool -> dense -> logit."""
    import torch
    from torch import nn

    class CharCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(config["vocab_size"], config["embedding_dim"], padding_idx=F.PAD_ID)
            self.convs = nn.ModuleList(
                nn.Conv1d(config["embedding_dim"], config["filters"], kernel_size=k, padding=k // 2)
                for k in config["kernel_widths"]
            )
            self.dropout = nn.Dropout(config["dropout"])
            self.dense = nn.Linear(config["filters"] * len(config["kernel_widths"]), config["dense"])
            self.out = nn.Linear(config["dense"], 1)

        def forward(self, ids):                      # ids: (batch, max_len) integer character ids
            x = self.embedding(ids.long()).transpose(1, 2)            # (batch, embedding, length)
            pooled = [torch.relu(conv(x)).amax(dim=2) for conv in self.convs]
            x = self.dropout(torch.cat(pooled, dim=1))
            x = self.dropout(torch.relu(self.dense(x)))
            return self.out(x).squeeze(1)            # raw score (log-odds)

    return CharCNN()


def set_torch_determinism(seed: int = SEED) -> dict:
    """Seed everything and switch on the deterministic settings PyTorch offers.

    GPU training is still not guaranteed bit-for-bit reproducible; the notebook
    measures the run-to-run difference instead of assuming it is zero.
    """
    import random

    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed, "cudnn.deterministic": True, "cudnn.benchmark": False,
        "use_deterministic_algorithms": "True (warn_only)",
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def cnn_raw(model, ids: np.ndarray, device: str = "cpu", batch_size: int = 4096) -> np.ndarray:
    """Raw scores (log-odds) from the PyTorch model."""
    import torch

    model = model.to(device).eval()
    out = np.empty(len(ids), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(ids[start:start + batch_size])).to(device)
            out[start:start + batch_size] = model(batch).float().cpu().numpy()
    return out.astype(np.float64)


def train_cnn(ids_train: np.ndarray, y_train: np.ndarray, ids_stop: np.ndarray, y_stop: np.ndarray, *,
              device: str, seed: int = SEED, training: Mapping[str, object] = CNN_TRAINING,
              config: Mapping[str, object] = CNN_CONFIG, log=print) -> tuple[object, list[dict]]:
    """Train the CNN with AdamW and early stopping. Returns (best model, per-epoch history).

    `ids_stop`, `y_stop` is the early-stopping split, taken from inside the
    fitting data. Training stops when its PR-AUC has not improved for
    `patience` epochs, and the weights of the best epoch are restored.
    `config` is the architecture; the default is the shipped one.
    """
    import copy

    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from torch import nn

    set_torch_determinism(seed)
    model = build_cnn(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                  weight_decay=training["weight_decay"])
    n_pos = float(y_train.sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y_train) - n_pos) / n_pos, device=device))

    x_all = torch.from_numpy(np.ascontiguousarray(ids_train)).to(device)          # uint8, stays on the device
    y_all = torch.from_numpy(np.asarray(y_train, dtype=np.float32)).to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch = int(training["batch_size"])

    best_score, best_state, bad_epochs, history = -1.0, None, 0, []
    for epoch in range(1, int(training["max_epochs"]) + 1):
        started = time.time()
        model.train()
        order = torch.randperm(len(x_all), generator=generator).to(device)
        total = 0.0
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_all[idx]), y_all[idx])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        raw = cnn_raw(model, ids_stop, device)
        row = {"epoch": epoch, "train_loss": total / len(order),
               "stop_pr_auc": float(average_precision_score(y_stop, raw)),
               "stop_roc_auc": float(roc_auc_score(y_stop, raw)),
               "seconds": time.time() - started}
        history.append(row)
        log(f"epoch {epoch:2d}  loss {row['train_loss']:.4f}  early-stopping PR-AUC {row['stop_pr_auc']:.5f}  "
            f"ROC-AUC {row['stop_roc_auc']:.5f}  {row['seconds']:.1f}s")
        if row["stop_pr_auc"] > best_score:
            best_score, bad_epochs = row["stop_pr_auc"], 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad_epochs += 1
            if bad_epochs >= int(training["patience"]):
                break
    model.load_state_dict(best_state)
    return model.cpu().eval(), history


def export_onnx(model, path: Path, max_len: int = F.DEFAULT_MAX_LEN) -> Path:
    """Export the CNN to ONNX. Input `ids` int64 (batch, max_len); output `raw_score` float32 (batch,)."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    example = torch.zeros((2, max_len), dtype=torch.int64)
    torch.onnx.export(
        model, (example,), str(path), input_names=["ids"], output_names=["raw_score"],
        dynamic_axes={"ids": {0: "batch"}, "raw_score": {0: "batch"}}, opset_version=17, dynamo=False,
    )
    return path


def onnx_session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def onnx_raw(session, ids: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Raw scores (log-odds) from the exported ONNX model, on CPU."""
    out = np.empty(len(ids), dtype=np.float64)
    for start in range(0, len(ids), batch_size):
        batch = np.ascontiguousarray(ids[start:start + batch_size]).astype(np.int64)
        out[start:start + batch_size] = session.run(["raw_score"], {"ids": batch})[0]
    return out


def sigmoid(raw) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    return np.where(raw >= 0, 1.0 / (1.0 + np.exp(-np.abs(raw))), np.exp(-np.abs(raw)) / (1.0 + np.exp(-np.abs(raw))))


def onnx_parity(model, session, ids: np.ndarray) -> dict:
    """Largest difference between PyTorch (CPU) and ONNX Runtime outputs on the same inputs."""
    torch_raw, ort_raw = cnn_raw(model, ids, "cpu"), onnx_raw(session, ids)
    return {
        "rows": int(len(ids)),
        "max_abs_diff_probability": float(np.max(np.abs(sigmoid(torch_raw) - sigmoid(ort_raw)))),
        "max_abs_diff_raw_score": float(np.max(np.abs(torch_raw - ort_raw))),
        "tolerance_probability": 1e-5,
    }


# --------------------------------------------------------------------------- #
# Hashes
# --------------------------------------------------------------------------- #

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_hashes(paths: Sequence[Path], out_path: Path | None = None) -> dict:
    """SHA-256 and size of each file, merged into models/hashes.json."""
    out_path = Path(out_path) if out_path else MODELS_DIR / "hashes.json"
    hashes = load_json(out_path) if out_path.exists() else {}
    for path in paths:
        path = Path(path)
        hashes[path.relative_to(ROOT).as_posix() if path.is_absolute() and ROOT in path.parents else path.name] = {
            "sha256": sha256_file(path), "bytes": int(path.stat().st_size),
        }
    save_json(dict(sorted(hashes.items())), out_path)
    return hashes
