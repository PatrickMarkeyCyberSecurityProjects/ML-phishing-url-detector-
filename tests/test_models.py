"""Tests for src/models.py on small invented data. No dataset files are needed."""

import numpy as np
import pandas as pd
import pytest

from src import features as F
from src import models as M


def _frame(n=400, seed=0):
    """Invented URLs: 'phishing' ones are short hyphenated hosts, 'legitimate' ones have paths."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        if i % 2:
            url = f"http://secure-login-{rng.integers(1e6)}.example/{'a' * int(rng.integers(0, 3))}?id={i}"
            rows.append((url, 1))
        else:
            url = f"https://www.site{rng.integers(1e6)}.example/articles/{rng.integers(1e4)}/page-{i}?ref=home"
            rows.append((url, 0))
    raw = pd.DataFrame(rows, columns=["url_raw", "label"])
    raw["url_full"] = raw["url_raw"].map(F.normalize_url)
    raw["url"] = raw["url_raw"].map(lambda u: F.normalize_url(u, strip_query=True))
    return raw


# ------------------------------------------------------------------ inputs

def test_tracks_read_different_columns_and_keep_31_features():
    frame = _frame(20)
    main, full = M.lexical_features(frame, "main"), M.lexical_features(frame, "full")
    assert main.shape == full.shape == (20, 31)
    q = F.FEATURE_NAMES.index("query_len")
    assert (main[:, q] == 0).all() and (full[:, q] > 0).all()
    ids = M.char_ids(frame, "main", chunk=7)
    assert ids.shape == (20, 200) and ids.dtype == np.uint8
    assert np.array_equal(ids, F.encode_batch(frame["url"], strip_query=True).astype(np.uint8))
    assert not np.array_equal(ids, M.char_ids(frame, "full"))


# ------------------------------------------------------------------ calibration and thresholds

def test_isotonic_breakpoints_reproduce_scikit_learn():
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(1)
    raw = rng.normal(size=3000)
    y = (rng.random(3000) < 1 / (1 + np.exp(-2 * raw))).astype(int)
    cal = M.fit_isotonic(raw, y)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, y)
    probe = np.concatenate([rng.normal(size=500) * 3, [-99.0, 99.0]])
    assert np.allclose(M.apply_calibration(cal, probe), iso.predict(probe), atol=1e-12)
    calibrated = M.apply_calibration(cal, np.sort(probe))
    assert (np.diff(calibrated) >= 0).all() and calibrated.min() >= 0 and calibrated.max() <= 1


def test_thresholds_meet_their_targets_on_the_data_they_were_chosen_on():
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(20_000, int), np.ones(5_000, int)]
    raw = np.r_[rng.normal(-2, 1, 20_000), rng.normal(2, 1, 5_000)]
    t = M.choose_thresholds(raw, y)
    block, warn = t["block"]["raw_score"], t["warn"]["raw_score"]
    assert (raw[y == 0] >= block).mean() <= M.BLOCK_FPR
    assert (raw[y == 1] >= warn).mean() >= M.WARN_TPR
    # Each is the tightest threshold that meets its target.
    lower = np.sort(raw)[np.searchsorted(np.sort(raw), block) - 1]
    assert (raw[y == 0] >= lower).mean() > M.BLOCK_FPR
    higher = np.sort(raw)[np.searchsorted(np.sort(raw), warn, side="right")]
    assert (raw[y == 1] >= higher).mean() < M.WARN_TPR
    assert t["block"]["validation_false_positives"] == int((raw[y == 0] >= block).sum())
    assert t["validation_rows"] == {"legitimate": 20_000, "phishing": 5_000}


def test_verdicts_follow_the_thresholds():
    thresholds = {"block": {"raw_score": 3.0}, "warn": {"raw_score": -1.0}}
    assert M.verdicts([-5, -1, 0, 2.999, 3, 9], thresholds).tolist() == [
        "Allow", "Warn", "Warn", "Warn", "Block", "Block"]
    # A warn threshold above block leaves no warn band.
    empty_band = {"block": {"raw_score": 1.0}, "warn": {"raw_score": 2.0}}
    assert M.verdicts([0.5, 1.5, 2.5], empty_band).tolist() == ["Allow", "Block", "Block"]


def test_sigmoid_is_stable():
    assert np.allclose(M.sigmoid([0, 800, -800]), [0.5, 1.0, 0.0])
    assert np.allclose(M.sigmoid([1.5, -1.5]), [1 / (1 + np.exp(-1.5)), 1 / (1 + np.exp(1.5))])


# ------------------------------------------------------------------ LightGBM

def test_lgbm_tunes_fits_calibrates_and_saves(tmp_path):
    fit, val = _frame(600, seed=3), _frame(300, seed=4)
    X, y = M.lexical_features(fit), fit["label"].to_numpy()
    tuned = M.tune_lgbm(X, y, n_trials=2, n_folds=2, sample_size=400)
    assert tuned["sample_rows"] == 400 and len(tuned["trials"]) == 2 and tuned["n_rounds"] >= 1
    again = M.tune_lgbm(X, y, n_trials=2, n_folds=2, sample_size=400)
    assert again["params"] == tuned["params"] and again["cv_pr_auc"] == tuned["cv_pr_auc"]    # seeded

    booster = M.fit_lgbm(X, y, tuned["params"], tuned["n_rounds"])
    raw = M.lgbm_raw(booster, M.lexical_features(val))
    assert np.allclose(M.sigmoid(raw), booster.predict(M.lexical_features(val)))
    assert np.array_equal(raw, M.lgbm_raw(M.fit_lgbm(X, y, tuned["params"], tuned["n_rounds"]),
                                          M.lexical_features(val)))                         # deterministic
    assert booster.feature_name() == list(F.FEATURE_NAMES)

    operating = M.build_operating_file("m", "main", raw, val["label"].to_numpy())
    path = M.save_json(operating, tmp_path / "m_operating.json")
    back = M.load_json(path)
    assert back["strip_query"] is True and set(back["thresholds"]) >= {"block", "warn"}
    assert np.allclose(M.apply_calibration(back["calibration"], raw), M.apply_calibration(operating["calibration"], raw))
    booster.save_model(str(tmp_path / "m.txt"))
    assert np.array_equal(M.lgbm_raw(M.load_lgbm(tmp_path / "m.txt"), M.lexical_features(val)), raw)
    # pred_contrib (used by the app for explanations) sums to the raw score.
    contrib = booster.predict(M.lexical_features(val), pred_contrib=True)
    assert contrib.shape == (len(val), 32) and np.allclose(contrib.sum(axis=1), raw)


# ------------------------------------------------------------------ CNN and ONNX

def test_cnn_architecture_matches_the_design():
    torch = pytest.importorskip("torch")
    model = M.build_cnn()
    assert model.embedding.num_embeddings == F.VOCAB_SIZE and model.embedding.embedding_dim == 32
    assert [c.kernel_size[0] for c in model.convs] == [3, 5, 7]
    assert all(c.out_channels == 128 for c in model.convs)
    assert model.dense.in_features == 384 and model.dense.out_features == 64 and model.dropout.p == 0.3
    out = model.eval()(torch.zeros((3, 200), dtype=torch.uint8))
    assert out.shape == (3,)


def test_cnn_trains_and_onnx_export_matches_pytorch(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    fit, stop = _frame(600, seed=5), _frame(200, seed=6)
    training = dict(M.CNN_TRAINING, max_epochs=3, batch_size=64)
    model, history = M.train_cnn(M.char_ids(fit), fit["label"].to_numpy(), M.char_ids(stop),
                                 stop["label"].to_numpy(), device="cpu", training=training, log=lambda *_: None)
    assert 1 <= len(history) <= 3 and history[-1]["stop_roc_auc"] > 0.9

    path = M.export_onnx(model, tmp_path / "cnn.onnx")
    session = M.onnx_session(path)
    ids = np.vstack([M.char_ids(stop), np.zeros((1, 200), np.uint8), np.full((1, 200), 96, np.uint8)])
    parity = M.onnx_parity(model, session, ids)
    assert parity["max_abs_diff_probability"] < 1e-5
    assert M.onnx_raw(session, ids[:1]).shape == (1,)                     # batch size is dynamic
    assert M.onnx_raw(session, ids[:7], batch_size=3).shape == (7,)


def test_write_hashes(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"abc")
    hashes = M.write_hashes([a], tmp_path / "hashes.json")
    assert hashes["a.bin"] == {
        "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "bytes": 3}
    assert M.load_json(tmp_path / "hashes.json") == hashes
