"""Loading, harmonizing, deduplicating, splitting and profiling the datasets.

Label convention everywhere: 1 = phishing, 0 = legitimate.

    PhreshPhish train split   training and validation (validation = most recent 15%)
    PhreshPhish test split    later-in-time test, a pinned 50,000-row sample;
                              never used to fit, tune, calibrate or set thresholds
    Secondary (Mendeley)      unseen dataset from another collector; evaluation only
    PhiUSIIL (UCI id 967)     cautionary baseline only: its legitimate class is a
                              single URL template (see docs/design_decisions.md)

Every dataset passes through the same `harmonize` function, and every URL is
cleaned by `src.features.normalize_url`. Nothing in this module fetches a URL
from any dataset; it only reads the strings.

Each harmonized frame carries three spellings of a URL:
    url        main track: normalized, query string and fragment removed. This is
               the row's identity (deduplication and overlap removal use it) and
               what the deployed models see.
    url_full   normalized with the query kept. Used by one comparison model.
    url_raw    exactly as the dataset shipped it.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src import features as F
from src import style as S

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"

SEED = 42
#: Written into the manifest. 2 = PhreshPhish-trained design (docs/design_decisions.md).
DESIGN = 2

# ---- PhiUSIIL (cautionary baseline) ------------------------------------------
PHIUSIIL_UCI_ID = 967
UCI_API = "https://archive.ics.uci.edu/api/dataset"
PHIUSIIL_URL_COL = "URL"
PHIUSIIL_LABEL_COL = "label"
#: PhiUSIIL ships with 1 = legitimate, the opposite of this project. Flip it.
PHIUSIIL_LABEL_MAP = {1: 0, 0: 1}
#: Class counts after mapping, as documented by UCI (1 = phishing).
PHIUSIIL_EXPECTED_COUNTS = {0: 134_850, 1: 100_945}
#: "Donated on 3/3/2024" on the UCI page. PhiUSIIL gives no collection dates.

# ---- PhreshPhish (train split: training; test split: evaluation only) --------
PHRESHPHISH_REPO = "phreshphish/phreshphish"
#: Commit of the dataset repository that was read. Pinned so a rerun reads the
#: same files even if the repository changes.
PHRESHPHISH_REVISION = "eca603e0ecd4f698f8d4d226c088659e72681239"
#: The test loader reads only these files and has no option for another split.
PHRESHPHISH_TEST_GLOB = "data/test-*.parquet"
#: Read only by `load_phreshphish_train_raw`.
PHRESHPHISH_TRAIN_GLOB = "data/train-*.parquet"
#: Small columns only. The `html` column (most of the 9 GB) is never read.
PHRESHPHISH_COLUMNS = ("sha256", "url", "label", "date")
PHRESHPHISH_LABEL_MAP = {"phish": 1, "benign": 0}
#: Test-split class counts from the dataset card (1 = phishing).
PHRESHPHISH_EXPECTED_COUNTS = {0: 91_260, 1: 76_800}
#: Train-split class counts from the dataset card (1 = phishing).
PHRESHPHISH_TRAIN_EXPECTED_COUNTS = {0: 276_729, 1: 221_526}
#: The evaluation sample was drawn once (stratified, seed 42) and is pinned
#: by the `sha256` id of each row, so a rebuild selects exactly the same rows.
EVAL_IDS_PATH = ROOT / "frozen" / "phreshphish_eval_sha256.txt"
#: Share of the train split, taken from its most recent days, held out for
#: validation: calibration and thresholds only. Tuning and early stopping use
#: data from inside the fitting part.
VALIDATION_SHARE = 0.15

# ---- Mendeley Data files (secondary dataset and its checks) -------------------
MENDELEY_FILE_URL = "https://data.mendeley.com/public-files/datasets/{dataset}/files/{file}/file_downloaded"
MENDELEY_MAX_BYTES = 100 * 1024 * 1024
#: Wangchuk, "Phishing URL dataset", v1, published 2026-01-09, CC BY 4.0.
#: The SHA-256 is the one Mendeley publishes for the file.
MENDELEY_WANGCHUK = {
    "key": "wangchuk",
    "name": "Wangchuk, Phishing URL dataset, v1",
    "doi": "10.17632/3jddhy2f6s.1",
    "dataset": "3jddhy2f6s",
    "file_id": "575f34ca-98c8-42cd-affe-7dad6aaf17a2",
    "filename": "dataset2.csv",
    "sha256": "fccce1a99282b3cd8a1fe08fa83350fe44f7afc4c8889a6d0a94ef9c4d7897e3",
    "published": "2026-01-09",
    "licence": "CC BY 4.0",
    "documented_counts": {0: 94_919, 1: 54_807},
}
WANGCHUK_URL_COL = "URL"
WANGCHUK_LABEL_COL = "Label"
#: The file is read as text, so the label values are the strings "0" and "1".
WANGCHUK_LABEL_MAP = {"1": 1, "0": 0}
#: Kaitholikkal & B, "Phishing URL dataset", v1, published 2024-04-02, CC BY 4.0.
#: Only the phishing-only file is read, to date the secondary's phishing class.
MENDELEY_KAITHOLIKKAL_PHISHING = {
    "key": "kaitholikkal",
    "name": "Kaitholikkal & B, Phishing URL dataset, v1 (Phishing URLs.csv)",
    "doi": "10.17632/vfszbj9b36.1",
    "dataset": "vfszbj9b36",
    "file_id": "97e4b9fc-8c55-4579-ae80-d30740d00913",
    "filename": "Phishing URLs.csv",
    "sha256": "00762308b0347c899769487dac35ee0ab0664fd6f2259588ff7151faab776041",
    "published": "2024-04-02",
    "licence": "CC BY 4.0",
}
#: Ariyadasa et al., "Phishing Websites Dataset", v1. Fallback secondary: read
#: only if Wangchuk fails its gate, and then only index.sql (no HTML).
MENDELEY_ARIYADASA = {
    "key": "ariyadasa",
    "name": "Ariyadasa et al., Phishing Websites Dataset, v1 (index.sql)",
    "doi": "10.17632/n96ncsr5g4.1",
    "dataset": "n96ncsr5g4",
    "file_id": "dac80106-cc68-43c3-8810-96408c09fbbc",
    "filename": "index.sql",
    "sha256": "3de752bb74332c8f2c01cbc1355b6eda0e2a6d5bcf9a87fddf9b26b5bc8f1ccc",
    "published": "2021-11-16",
    "licence": "CC BY 4.0",
    "documented_counts": {0: 50_000, 1: 30_000},
}

ARIYADASA_URL_COL = "url"
ARIYADASA_LABEL_COL = "result"
#: The Mendeley description: "0 for legitimate and 1 for phishing". Read as text.
ARIYADASA_LABEL_MAP = {"1": 1, "0": 0}

#: File extensions that mark a URL as a resource a browser loads for a page, not
#: a page a person visits. Fixed before any model was trained; used only for a
#: sensitivity result, never as a model feature. Documents (pdf, doc) are not
#: listed: a person can be sent a link to one.
NON_PAGE_EXTENSIONS = frozenset(
    "js mjs css map json xml txt csv ico png jpg jpeg gif svg webp bmp avif woff woff2 ttf otf eot "
    "mp3 mp4 webm ogg wav avi mov zip gz tar rar 7z exe dmg apk msi iso bin swf".split()
)

_TEST_FILE_RE = re.compile(r"^test-[\w\-]+\.parquet$")
_TRAIN_FILE_RE = re.compile(r"^train-[\w\-]+\.parquet$")

CLASS_NAMES = {0: "legitimate", 1: "phishing"}


class ClassCountMismatch(ValueError):
    """The loaded class counts differ from the documented ones."""


# --------------------------------------------------------------------------- #
# Reproducibility helpers
# --------------------------------------------------------------------------- #

def set_seeds(seed: int = SEED) -> None:
    """Seed Python and NumPy, and PyTorch when it is already imported.

    LightGBM takes its seed as a model parameter, set where the model is built.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(df: pd.DataFrame) -> str:
    """SHA-256 of the (url, label) pairs in row order. Independent of file format."""
    digest = hashlib.sha256()
    for url, label in zip(df["url"], df["label"]):
        digest.update(f"{url}\t{int(label)}\n".encode("utf-8", "surrogatepass"))
    return digest.hexdigest()


def defang(url: object) -> str:
    """Make a URL unclickable for display: `hxxps://example[.]com/a`.

    Used only when printing sample rows. Data files keep the real strings.
    """
    text = url if isinstance(url, str) else ""
    text = re.sub(r"^(\s*)http", r"\1hxxp", text, count=1, flags=re.IGNORECASE)
    return text.replace(".", "[.]")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _https_get(url: str, timeout: int = 300) -> bytes:
    """Download a dataset file over HTTPS, verifying against certifi's CA bundle.

    `ucimlrepo` reads the UCI file with the operating system's certificate
    store, which on the development machine rejected UCI's certificate chain as
    expired. The certifi bundle verifies it, so this is used instead.
    Certificate checking stays on.
    """
    import ssl
    import urllib.request

    import certifi

    context = ssl.create_default_context(cafile=certifi.where())
    # Mendeley Data answers 403 to Python's default client string, so a generic
    # one is sent. It carries no personal or project information.
    request = urllib.request.Request(url, headers={"User-Agent": "dataset-downloader/1.0"})
    with urllib.request.urlopen(request, context=context, timeout=timeout) as response:
        return response.read()


def load_phiusiil_raw(raw_dir: Path = RAW_DIR, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Download PhiUSIIL from UCI and keep only the raw URL and the label.

    The file location comes from the UCI repository API for dataset 967, the
    same endpoint the `ucimlrepo` package uses.

    Returns (frame, source). `frame` has the two original columns, untouched.
    `source` records every column name in the downloaded file, so the notebook
    can show what was discarded. The precomputed and HTML-derived columns are
    dropped here and are never used as features.

    The result is cached under data/raw/phiusiil/, so later runs work offline.
    """
    out_dir = Path(raw_dir) / "phiusiil"
    cache = out_dir / "phiusiil_url_label.parquet"
    source_path = out_dir / "source.json"
    if cache.exists() and source_path.exists() and not refresh:
        return pd.read_parquet(cache), json.loads(source_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(_https_get(f"{UCI_API}?id={PHIUSIIL_UCI_ID}"))["data"]
    csv_path = out_dir / "data.csv"
    csv_path.write_bytes(_https_get(metadata["data_url"]))
    original = pd.read_csv(csv_path)
    source = {
        "uci_id": PHIUSIIL_UCI_ID,
        "name": metadata.get("name"),
        "data_url": metadata.get("data_url"),
        "documented_instances": metadata.get("num_instances"),
        "documented_features": metadata.get("num_features"),
        "csv_sha256": sha256_file(csv_path),
        "n_rows": int(len(original)),
        "n_columns": int(original.shape[1]),
        "columns": [str(c) for c in original.columns],
        "columns_kept": [PHIUSIIL_URL_COL, PHIUSIIL_LABEL_COL],
    }
    frame = original[[PHIUSIIL_URL_COL, PHIUSIIL_LABEL_COL]].copy()

    frame.to_parquet(cache, index=False)
    source_path.write_text(json.dumps(source, indent=2), encoding="utf-8")
    return frame, source


def assert_test_split_only(paths: Iterable[str]) -> None:
    """Refuse any PhreshPhish file that is not part of the test split."""
    paths = list(paths)
    if not paths:
        raise ValueError("No PhreshPhish test files were found.")
    for path in paths:
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        if not _TEST_FILE_RE.match(name):
            raise ValueError(
                f"Refusing to read {path!r}: only the PhreshPhish test split may be loaded."
            )


def assert_train_split_only(paths: Iterable[str]) -> None:
    """Refuse any PhreshPhish file that is not part of the train split."""
    paths = list(paths)
    if not paths:
        raise ValueError("No PhreshPhish train files were found.")
    for path in paths:
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        if not _TRAIN_FILE_RE.match(name):
            raise ValueError(
                f"Refusing to read {path!r}: this loader reads only the PhreshPhish train split."
            )


def _read_phreshphish_columns(fs, paths: Sequence[str], workers: int = 6) -> tuple[pd.DataFrame, list, list]:
    """Read the four small columns from each parquet file, in file order.

    Parquet stores columns separately, so range requests fetch `sha256`, `url`,
    `label` and `date` and never transfer the `html` column.
    """
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow as pa
    import pyarrow.parquet as pq

    def read_one(path: str):
        with fs.open(path, "rb") as fh:
            parquet = pq.ParquetFile(fh)
            table = parquet.read(columns=list(PHRESHPHISH_COLUMNS))
            info = {
                "file": path.rsplit("/", 1)[-1],
                "rows": int(parquet.metadata.num_rows),
                "bytes": int(fs.size(path)),
            }
            return table, info, list(parquet.schema_arrow.names)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(read_one, paths))
    frame = pa.concat_tables([r[0] for r in results]).to_pandas()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame, [r[1] for r in results], results[0][2]


def _load_phreshphish_split(split: str, glob: str, guard, cache_name: str, source_name: str,
                            raw_dir: Path, refresh: bool) -> tuple[pd.DataFrame, dict]:
    out_dir = Path(raw_dir) / "phreshphish"
    cache = out_dir / cache_name
    source_path = out_dir / source_name
    if cache.exists() and source_path.exists() and not refresh:
        return pd.read_parquet(cache), json.loads(source_path.read_text(encoding="utf-8"))

    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    base = f"datasets/{PHRESHPHISH_REPO}@{PHRESHPHISH_REVISION}"
    paths = sorted(fs.glob(f"{base}/{glob}"))
    guard(paths)
    frame, files, all_columns = _read_phreshphish_columns(fs, paths)

    source = {
        "repo": PHRESHPHISH_REPO,
        "revision": PHRESHPHISH_REVISION,
        "split": split,
        "files": files,
        "n_files": len(files),
        "bytes_in_repo": int(sum(f["bytes"] for f in files)),
        "n_rows": int(len(frame)),
        "columns": all_columns,
        "columns_kept": list(PHRESHPHISH_COLUMNS),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    source_path.write_text(json.dumps(source, indent=2), encoding="utf-8")
    return frame, source


def load_phreshphish_test_raw(raw_dir: Path = RAW_DIR, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Read four small columns of the PhreshPhish test split from Hugging Face.

    The split is about 9 GB because every row carries the page HTML. Only
    `sha256`, `url`, `label` and `date` are read; the HTML is never transferred.

    There is no argument that selects another split, and every file name is
    checked by `assert_test_split_only`. Rows returned here are for evaluation
    only: they are never used to fit, tune, calibrate or choose a threshold.

    Returns (frame, source) and caches both under data/raw/phreshphish/.
    """
    return _load_phreshphish_split(
        "test", PHRESHPHISH_TEST_GLOB, assert_test_split_only,
        "phreshphish_test_columns.parquet", "source.json", raw_dir, refresh,
    )


def load_phreshphish_train_raw(raw_dir: Path = RAW_DIR, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Read four small columns of the PhreshPhish TRAIN split from Hugging Face.

    This is the project's training data (see docs/design_decisions.md). The split
    is about 27 GB with HTML; only `sha256`, `url`, `label` and `date` are read.
    `date` and `sha256` are metadata and never model features.

    Like the test loader it has no argument that selects a split, and every file
    name is checked, here by `assert_train_split_only`.

    Returns (frame, source) and caches both under data/raw/phreshphish/.
    """
    return _load_phreshphish_split(
        "train", PHRESHPHISH_TRAIN_GLOB, assert_train_split_only,
        "phreshphish_train_columns.parquet", "source_train.json", raw_dir, refresh,
    )


def download_mendeley_file(spec: Mapping[str, object], raw_dir: Path = RAW_DIR, refresh: bool = False) -> Path:
    """Download one file of a Mendeley Data dataset and verify its SHA-256.

    `spec` is one of the MENDELEY_* dictionaries above. The file goes into its
    own directory, data/raw/<key>/, and is only ever parsed as text. The hash
    is compared with the one Mendeley publishes for that file version, so a
    changed or truncated download raises instead of being used.
    """
    out_dir = Path(raw_dir) / str(spec["key"])
    path = out_dir / str(spec["filename"])
    if not path.exists() or refresh:
        out_dir.mkdir(parents=True, exist_ok=True)
        url = MENDELEY_FILE_URL.format(dataset=spec["dataset"], file=spec["file_id"])
        payload = _https_get(url)
        if len(payload) > MENDELEY_MAX_BYTES:
            raise ValueError(f"{spec['filename']}: {len(payload):,} bytes is larger than expected.")
        path.write_bytes(payload)
    digest = sha256_file(path)
    if digest != spec["sha256"]:
        raise ValueError(
            f"{path}: SHA-256 {digest} does not match the published {spec['sha256']}."
        )
    return path


def _read_url_csv(path: Path) -> pd.DataFrame:
    """Read a CSV of URLs as text. Nothing is parsed as a number or a missing value."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8", encoding_errors="replace")


def load_wangchuk_raw(raw_dir: Path = RAW_DIR, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Wangchuk's "Phishing URL dataset" v1 from Mendeley Data: raw URL and label.

    The file has two columns, `URL` and `Label`. Its description documents
    94,919 legitimate and 54,807 phishing URLs, and the file holds 94,919 rows
    with Label 0 and 54,807 with Label 1, so Label 1 = phishing. That is already
    this project's convention; `harmonize` asserts the counts.

    This dataset is for evaluation only. Its legitimate class comes from a web
    crawl, not from verified pages, so the project calls it "presumed benign".
    """
    spec = MENDELEY_WANGCHUK
    path = download_mendeley_file(spec, raw_dir, refresh)
    frame = _read_url_csv(path)
    source = {
        "doi": spec["doi"],
        "file": spec["filename"],
        "file_sha256": sha256_file(path),
        "file_bytes": int(path.stat().st_size),
        "n_rows": int(len(frame)),
        "columns": [str(c) for c in frame.columns],
        "url_column": WANGCHUK_URL_COL,
        "label_column": WANGCHUK_LABEL_COL,
        "label_map": {str(k): int(v) for k, v in WANGCHUK_LABEL_MAP.items()},
        "label_direction": "Label 1 = phishing, Label 0 = legitimate (as shipped; matches the documented class counts)",
        "raw_label_counts": {str(k): int(v) for k, v in frame[WANGCHUK_LABEL_COL].value_counts().items()},
        "empty_urls": int((frame[WANGCHUK_URL_COL].str.strip() == "").sum()),
    }
    return frame, source


_SQL_ROW_RE = re.compile(
    r"^\((\d+), '((?:[^'\\]|\\.|'')*)', '((?:[^'\\]|\\.|'')*)', (\d+), '([^']*)'\)[,;]\s*$"
)
_SQL_ESCAPE_RE = re.compile(r"\\(.)|''", re.S)
_SQL_ESCAPES = {"0": "\0", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a", "b": "\b"}


def _sql_unescape(text: str) -> str:
    """Undo MySQL string escaping (backslash escapes and doubled quotes)."""
    return _SQL_ESCAPE_RE.sub(lambda m: "'" if m.group(1) is None else _SQL_ESCAPES.get(m.group(1), m.group(1)), text)


def parse_index_sql(text: str) -> pd.DataFrame:
    """Read the rows of a phpMyAdmin dump of the `index` table, as text.

    The dump is never executed. Each value row has the form
    `(rec_id, 'url', 'website', result, 'created_date')`. Every line that
    starts with "(" must parse, otherwise this raises, so no row can be lost
    silently. All columns are returned as strings.
    """
    rows, bad = [], []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.startswith("("):
            continue
        match = _SQL_ROW_RE.match(line)
        if match is None:
            bad.append(number)
            continue
        rec_id, url, website, result, created = match.groups()
        rows.append((rec_id, _sql_unescape(url), _sql_unescape(website), result, created))
    if bad:
        raise ValueError(f"index.sql: {len(bad)} value rows could not be parsed (first at line {bad[0]}).")
    return pd.DataFrame(rows, columns=["rec_id", "url", "website", "result", "created_date"])


def load_ariyadasa_raw(raw_dir: Path = RAW_DIR, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Ariyadasa et al., "Phishing Websites Dataset" v1: URL, label and download date.

    Only `index.sql` is read, never the HTML pages. Its `result` column is
    documented as 0 = legitimate, 1 = phishing, which is this project's
    convention. `created_date` is the date the page was downloaded.

    Evaluation only. The legitimate class is called "presumed benign".
    """
    spec = MENDELEY_ARIYADASA
    path = download_mendeley_file(spec, raw_dir, refresh)
    frame = parse_index_sql(path.read_text(encoding="utf-8", errors="replace"))
    frame["date"] = pd.to_datetime(frame["created_date"], errors="coerce")
    source = {
        "doi": spec["doi"],
        "file": spec["filename"],
        "file_sha256": sha256_file(path),
        "file_bytes": int(path.stat().st_size),
        "n_rows": int(len(frame)),
        "columns": ["rec_id", "url", "website", "result", "created_date"],
        "url_column": ARIYADASA_URL_COL,
        "label_column": ARIYADASA_LABEL_COL,
        "label_map": {str(k): int(v) for k, v in ARIYADASA_LABEL_MAP.items()},
        "label_direction": "result 1 = phishing, result 0 = legitimate (as documented on the Mendeley page)",
        "raw_label_counts": {str(k): int(v) for k, v in frame[ARIYADASA_LABEL_COL].value_counts().items()},
        "empty_urls": int((frame[ARIYADASA_URL_COL].str.strip() == "").sum()),
        "rec_id_max": int(frame["rec_id"].astype(int).max()),
        "date_min": str(frame["date"].min().date()),
        "date_max": str(frame["date"].max().date()),
    }
    return frame, source


def load_kaitholikkal_phishing_raw(raw_dir: Path = RAW_DIR, refresh: bool = False) -> pd.DataFrame:
    """The 54,807 phishing URLs published by Kaitholikkal & B on 2 April 2024.

    Read only to check where the secondary dataset's phishing class comes from.
    Returns a frame with `url_raw` and the normalized `url`. Never scored.
    """
    path = download_mendeley_file(MENDELEY_KAITHOLIKKAL_PHISHING, raw_dir, refresh)
    raw = _read_url_csv(path)["url"]
    return pd.DataFrame({"url_raw": raw, "url": [F.normalize_url(u) for u in raw]})


def resource_kind(url: object) -> str:
    """"robots.txt", "other non-page resource" or "page", from the URL path alone."""
    last = F.split_url(url).path.rsplit("/", 1)[-1].lower()
    if last == "robots.txt":
        return "robots.txt"
    if "." in last and last.rsplit(".", 1)[1] in NON_PAGE_EXTENSIONS:
        return "other non-page resource"
    return "page"


# --------------------------------------------------------------------------- #
# Harmonization: the one function both datasets pass through
# --------------------------------------------------------------------------- #

def harmonize(
    df: pd.DataFrame,
    *,
    url_col: str,
    label_col: str,
    label_map: Mapping[object, int],
    source: str,
    expected_counts: Mapping[int, int] | None = None,
    keep: Sequence[str] = (),
    strip_query: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Bring a raw dataset to the project's common format.

    Output columns: `url`, `url_full`, `url_raw` (as shipped), `label`
    (1 = phishing, 0 = legitimate), `source`, `had_query`, plus any columns
    named in `keep`.

    Steps:
      1. map labels with `label_map`; any label value not in the map raises;
      2. assert the class counts (against `expected_counts` when given);
      3. normalize every URL with `src.features.normalize_url`. `url_full`
         keeps the query string. `url` is the main-track form: with
         `strip_query=True` the query string and fragment are removed,
         otherwise it equals `url_full`. `had_query` records whether the
         shipped URL had a query string;
      4. drop rows whose URL is empty after normalization.

    Returns (frame, report). The report counts what each step changed.
    """
    raw_labels = df[label_col]
    unknown = sorted(set(raw_labels.dropna().unique()) - set(label_map), key=str)
    if unknown or raw_labels.isna().any():
        raise ValueError(f"{source}: label values not in the label map: {unknown or 'missing values'}")
    if set(label_map.values()) - {0, 1}:
        raise ValueError("label_map must map onto 0 (legitimate) and 1 (phishing)")

    label = raw_labels.map(label_map).astype("int8")
    raw_counts = {str(k): int(v) for k, v in raw_labels.value_counts().sort_index().items()}
    counts = {int(k): int(v) for k, v in label.value_counts().sort_index().items()}
    counts = {0: counts.get(0, 0), 1: counts.get(1, 0)}

    # Polarity check: the phishing count must equal the count of the raw values
    # that the map sends to 1. This fails if the map is applied the wrong way.
    phishing_raw = sum(int((raw_labels == k).sum()) for k, v in label_map.items() if v == 1)
    assert counts[1] == phishing_raw and counts[0] + counts[1] == len(df)
    if expected_counts is not None and counts != dict(expected_counts):
        raise ClassCountMismatch(
            f"{source}: class counts after mapping are {counts}, expected {dict(expected_counts)}"
        )

    url_raw = df[url_col].where(df[url_col].map(lambda v: isinstance(v, str)), "")
    changes = pd.DataFrame([F.normalization_changes(u, strip_query) for u in url_raw], index=df.index)
    full = [F.normalize_url(u) for u in url_raw]
    had_query = np.array([bool(F._split(u).query) for u in full], dtype=bool)
    out = pd.DataFrame(
        {
            "url": [F.normalize_url(u, strip_query) for u in url_raw] if strip_query else full,
            "url_full": full,
            "url_raw": url_raw,
            "label": label,
            "source": source,
            "had_query": had_query,
        },
        index=df.index,
    )
    for col in keep:
        out[col] = df[col]

    empty = out["url"] == ""
    out = out.loc[~empty].reset_index(drop=True)
    report = {
        "source": source,
        "rows_in": int(len(df)),
        "raw_label_counts": raw_counts,
        "class_counts_after_mapping": counts,
        "empty_urls_dropped": int(empty.sum()),
        "rows_out": int(len(out)),
        "urls_changed": int((out["url"] != out["url_raw"]).sum()),
        **{f"n_{name}": int(changes[name].sum()) for name in changes.columns},
        "scheme_added_by_class": {
            CLASS_NAMES[k]: int(changes.loc[label == k, "scheme_added"].sum()) for k in (0, 1)
        },
        "root_slash_added_by_class": {
            CLASS_NAMES[k]: int(changes.loc[label == k, "root_slash_added"].sum()) for k in (0, 1)
        },
        "strip_query": bool(strip_query),
        "had_query_by_class": {CLASS_NAMES[k]: int(had_query[(label == k).to_numpy()].sum()) for k in (0, 1)},
        "had_query_share_by_class": {
            CLASS_NAMES[k]: round(float(had_query[(label == k).to_numpy()].mean()), 6) if (label == k).any() else 0.0
            for k in (0, 1)
        },
        "query_removed_by_class": {
            CLASS_NAMES[k]: int(changes.loc[label == k, "query_removed"].sum()) for k in (0, 1)
        },
    }
    return out, report


def deduplicate(df: pd.DataFrame, tie_break: Sequence[str] = ()) -> tuple[pd.DataFrame, dict]:
    """Keep one row per normalized URL.

    A URL that appears with both labels is ambiguous, so every copy is dropped.
    For other duplicates the first row is kept after sorting by URL and then by
    the `tie_break` columns (for PhreshPhish, `date`, so the earliest sighting
    is kept). The result is sorted by URL.
    """
    labels_per_url = df.groupby("url")["label"].nunique()
    conflict_urls = labels_per_url.index[labels_per_url > 1]
    conflict = df["url"].isin(conflict_urls)
    kept = df.loc[~conflict].sort_values(["url", *tie_break], kind="mergesort")
    out = kept.drop_duplicates("url", keep="first").reset_index(drop=True)
    report = {
        "rows_in": int(len(df)),
        "conflicting_urls": int(len(conflict_urls)),
        "conflicting_rows_dropped": int(conflict.sum()),
        "duplicate_rows_dropped": int(len(kept) - len(out)),
        "duplicate_rows_dropped_by_class": class_counts(kept[kept.duplicated("url", keep="first")]),
        "rows_out": int(len(out)),
        "class_counts": class_counts(out),
    }
    return out, report


def collapse_report(df: pd.DataFrame) -> dict:
    """How many rows share their `url` with another row, without removing any.

    Used for the pinned evaluation sample: its membership is kept, so rows that
    become the same string once the query is removed are counted, not dropped.
    """
    sizes = df.groupby("url")["label"].agg(["size", "nunique"])
    shared = sizes[sizes["size"] > 1]
    conflict = shared[shared["nunique"] > 1]
    in_shared = df["url"].isin(shared.index)
    return {
        "rows": int(len(df)),
        "distinct_urls": int(len(sizes)),
        "rows_sharing_a_url": int(in_shared.sum()),
        "rows_sharing_a_url_by_class": class_counts(df[in_shared]),
        "urls_shared_by_several_rows": int(len(shared)),
        "surplus_rows": int(len(df) - len(sizes)),
        "urls_with_conflicting_labels": int(len(conflict)),
        "rows_with_conflicting_labels": int(df["url"].isin(conflict.index).sum()),
    }


def drop_on_or_after(df: pd.DataFrame, date: object, date_col: str = "date") -> tuple[pd.DataFrame, dict]:
    """Remove rows dated on or after `date`, so the frame ends strictly before it."""
    cut = pd.Timestamp(date).normalize()
    late = df[date_col].dt.normalize() >= cut
    out = df.loc[~late].reset_index(drop=True)
    report = {
        "cut_date": str(cut.date()),
        "rows_in": int(len(df)),
        "rows_dropped": int(late.sum()),
        "dropped_by_class": class_counts(df[late]),
        "rows_out": int(len(out)),
        "date_max_after": str(out[date_col].max().date()) if len(out) else None,
    }
    return out, report


def remove_cross_dataset_overlap(primary: pd.DataFrame, secondary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove from `primary` every URL that also appears in `secondary`.

    Two URLs match when they share `src.features.canonical_key`, which ignores
    the scheme, a leading `www.` and trailing slashes. That is stricter than an
    exact match, so the same page cannot sit in training under one spelling and
    in the evaluation set under another. Exact matches are counted separately.
    """
    secondary_exact = set(secondary["url"])
    key_to_label = dict(zip(secondary["url"].map(F.canonical_key), secondary["label"]))
    primary_keys = primary["url"].map(F.canonical_key)
    exact = primary["url"].isin(secondary_exact)
    loose = primary_keys.isin(key_to_label.keys())
    out = primary.loc[~loose].reset_index(drop=True)
    removed = primary.loc[loose]
    report = {
        "rows_in": int(len(primary)),
        "exact_matches": int(exact.sum()),
        "loose_matches": int(loose.sum()),
        "removed": int(loose.sum()),
        "removed_by_class": class_counts(removed),
        # Same page, opposite label in the other dataset.
        "removed_with_different_label": int(
            (primary_keys[loose].map(key_to_label) != removed["label"]).sum()
        ),
        "rows_out": int(len(out)),
        "class_counts": class_counts(out),
    }
    return out, report


def load_eval_ids(path: Path = EVAL_IDS_PATH) -> list[str]:
    """The `sha256` ids of the pinned PhreshPhish evaluation sample."""
    return Path(path).read_text(encoding="ascii").split()


def select_by_ids(df: pd.DataFrame, ids: Sequence[str], id_col: str = "sha256") -> pd.DataFrame:
    """Rows of `df` whose id is in `ids`. Raises unless every id is found exactly once.

    Used to rebuild the evaluation sample from the pinned id list. Sorted by URL.
    """
    wanted = set(ids)
    if len(wanted) != len(ids):
        raise ValueError("The id list contains duplicates.")
    out = df[df[id_col].isin(wanted)]
    if len(out) != len(wanted) or not out[id_col].is_unique:
        raise ValueError(f"Expected {len(wanted):,} rows from the id list, found {len(out):,}.")
    return out.sort_values("url", kind="mergesort").reset_index(drop=True)


def date_range(df: pd.DataFrame, date_col: str = "date") -> dict:
    """Earliest and latest date, number of distinct days and number of rows."""
    days = df[date_col].dt.normalize()
    return {
        "rows": int(len(df)),
        "date_min": str(days.min().date()),
        "date_max": str(days.max().date()),
        "distinct_days": int(days.nunique()),
    }


def temporal_split(
    df: pd.DataFrame, val_share: float = VALIDATION_SHARE, date_col: str = "date"
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Hold out the most recent days of `df` for validation.

    The cut falls on a day boundary: validation is every row dated on or after
    the latest day D for which those rows make up at least `val_share` of the
    frame. No day is shared between the two parts, so validation is strictly
    later than training. Both parts are sorted by URL.

    Returns ({"train": ..., "val": ...}, report).
    """
    if not 0 < val_share < 1:
        raise ValueError("val_share must be between 0 and 1")
    days = df[date_col].dt.normalize()
    if days.isna().any():
        raise ValueError("Every row needs a date for a temporal split.")
    per_day = days.value_counts().sort_index(ascending=False)
    from_latest = per_day.cumsum()
    cutoff = from_latest.index[int(np.searchsorted(from_latest.to_numpy(), val_share * len(df)))]
    is_val = days >= cutoff
    parts = {
        "train": df.loc[~is_val].sort_values("url", kind="mergesort").reset_index(drop=True),
        "val": df.loc[is_val].sort_values("url", kind="mergesort").reset_index(drop=True),
    }
    if parts["train"].empty:
        raise ValueError("The temporal split left no training rows.")
    report = {
        "requested_val_share": val_share,
        "val_share": round(len(parts["val"]) / len(df), 6),
        "cutoff_date": str(cutoff.date()),
        "rule": "validation = rows dated on or after cutoff_date; training = rows before it",
        "train": {**date_range(parts["train"], date_col), **class_counts(parts["train"])},
        "val": {**date_range(parts["val"], date_col), **class_counts(parts["val"])},
    }
    return parts, report


def split_70_15_15(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Stratified 70/15/15 split into train, val and test.

    Rows are sorted by URL first, so the split depends only on the set of URLs
    and the seed, not on the order the source file happened to have.
    """
    from sklearn.model_selection import train_test_split

    ordered = df.sort_values("url", kind="mergesort").reset_index(drop=True)
    train, rest = train_test_split(ordered, test_size=0.30, stratify=ordered["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=seed)
    return {
        name: part.sort_values("url", kind="mergesort").reset_index(drop=True)
        for name, part in (("train", train), ("val", val), ("test", test))
    }


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["label"].value_counts()
    return {"legitimate": int(counts.get(0, 0)), "phishing": int(counts.get(1, 0))}


def save_processed(
    frames: Mapping[str, pd.DataFrame],
    out_dir: Path = PROCESSED_DIR,
    extra: Mapping[str, object] | None = None,
) -> dict:
    """Write each frame to `<name>.parquet` and a manifest.json beside them.

    The manifest holds row and class counts and two hashes per file, plus
    whatever is passed in `extra` (step reports, seed, source revision). It has
    no timestamps, so two runs on the same inputs give an identical manifest.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, frame in frames.items():
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        counts = class_counts(frame)
        files[name] = {
            "file": path.name,
            "rows": int(len(frame)),
            **counts,
            "phishing_share": round(counts["phishing"] / len(frame), 6) if len(frame) else 0.0,
            "content_sha256": content_hash(frame),
            "file_sha256": sha256_file(path),
        }
    manifest = {"design": DESIGN, "label_convention": "1 = phishing, 0 = legitimate", "files": files,
                **(extra or {})}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return manifest


def load_processed(name: str, out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Read one processed file by name, for example "phreshphish_train"."""
    return pd.read_parquet(Path(out_dir) / f"{name}.parquet")


def load_manifest(out_dir: Path = PROCESSED_DIR) -> dict:
    return json.loads((Path(out_dir) / "manifest.json").read_text(encoding="utf-8"))


def split_counts_table(manifest: Mapping[str, object]) -> pd.DataFrame:
    """Rows per saved file and per class, read from the manifest."""
    rows = [
        {
            "file": info["file"],
            "rows": info["rows"],
            "legitimate (0)": info["legitimate"],
            "phishing (1)": info["phishing"],
            "phishing share": info["phishing_share"],
        }
        for info in manifest["files"].values()
    ]
    return pd.DataFrame(rows).set_index("file")


# --------------------------------------------------------------------------- #
# Profiling: structure, features, artefact checks
# --------------------------------------------------------------------------- #

def feature_frame(urls: Iterable[object]) -> pd.DataFrame:
    """`src.features.extract_features` for every URL, as a DataFrame."""
    return pd.DataFrame([F.extract_features(u) for u in urls])


def structure_frame(urls: Iterable[object]) -> pd.DataFrame:
    """`src.features.describe_url` for every URL, as a DataFrame."""
    return pd.DataFrame([F.describe_url(u) for u in urls])


def raw_scheme(url: object) -> str:
    """Scheme of a URL exactly as shipped, lowercased, or "none" when it has none."""
    match = F._SCHEME_RE.match(url.strip()) if isinstance(url, str) else None
    return match.group(1).lower() if match else "none"


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Add structural columns and the lexical features to a harmonized frame.

    Two columns describe the URL as the dataset shipped it (`url_raw` when the
    frame has it), because normalization hides both facts:
    `raw_ends_at_host` is True when nothing followed the host, not even a
    slash, and `raw_scheme` is the scheme as written, or "none". When the frame
    has `had_query`, `has_query` reports that (the shipped URL had a query
    string) even though `url` no longer does. Features are computed from `url`.
    """
    urls = df["url"].tolist()
    structure = structure_frame(urls)
    feats = feature_frame(urls).drop(columns=["tld"])
    raw = df["url_raw"] if "url_raw" in df.columns else df["url"]
    slash_added = np.array([F.normalization_changes(u)["root_slash_added"] for u in raw], dtype=bool)
    structure["raw_ends_at_host"] = slash_added & structure["is_bare_homepage"].to_numpy(dtype=bool)
    structure["raw_scheme"] = [raw_scheme(u) for u in raw]
    structure["resource_kind"] = [resource_kind(u) for u in urls]
    if "had_query" in df.columns:
        # `has_query` then describes the shipped URL, not the main-track form.
        structure["has_query"] = df["had_query"].to_numpy(dtype=bool)
    return pd.concat([df.reset_index(drop=True), structure, feats], axis=1)


def add_homepage_template(profiled: pd.DataFrame) -> pd.DataFrame:
    """Add three fixed descriptions of URL shape, used to measure a collection artefact.

    nothing_after_host           as shipped, the URL ends at the host: no slash, path,
                                 query or fragment
    homepage_template            as shipped, exactly `https://www.<host>`
    homepage_template_any_slash  `https://www.<host>` or `https://www.<host>/`. After
                                 normalization these are one URL, so this is the form
                                 the template rule takes on normalized URLs

    None of these is a model feature.
    """
    out = profiled.copy()
    https_www = out["is_https"].eq(1) & out["has_www"]
    out["nothing_after_host"] = out["raw_ends_at_host"]
    out["homepage_template"] = https_www & out["nothing_after_host"]
    out["homepage_template_any_slash"] = https_www & out["is_bare_homepage"]
    return out


def markdown_table(rows: Sequence[Sequence[object]], headers: Sequence[str]) -> str:
    """A GitHub-flavoured markdown table, for display in the notebook."""
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Charts (PNG, 200 dpi). Colours are the first two slots of a palette checked
# for colour-blind separation; classes are also named by direct labels.
# --------------------------------------------------------------------------- #

_CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}


def plot_length_ecdf(
    profiled: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    path: Path,
    xmin: float = 0,
    legend_loc: str = "lower right",
):
    """Cumulative distribution of a length column by class, one panel per dataset.

    Each curve shows the share of a class whose value is at or below x. With
    `xmin=0` the x axis is linear up to 1 and logarithmic beyond, so zero-length
    values show. A class whose values are all equal draws as one vertical step
    followed by a flat line at 100%.
    """
    import matplotlib.pyplot as plt

    sources = list(dict.fromkeys(profiled["source"]))
    xmax = 1000
    ticks = [t for t in (0, 1, 10, 100, 1000) if t >= xmin]
    fig, axes = plt.subplots(1, len(sources), figsize=(5.4 * len(sources), 4.2), sharey=True)
    fig.patch.set_facecolor(S.SURFACE)
    for ax, source in zip(np.atleast_1d(axes), sources):
        S.style_axis(ax)
        part = profiled[profiled["source"] == source]
        for label in (0, 1):
            values = np.sort(part.loc[part["label"] == label, column].to_numpy())
            share = np.arange(1, len(values) + 1) / len(values)
            # Start at 0% and carry the last level to the right edge.
            xs = np.concatenate(([values[0]], values, [max(xmax, values[-1])]))
            ys = np.concatenate(([0.0], share, [1.0]))
            ax.step(xs, ys, where="post", color=_CLASS_COLORS[label], linewidth=2,
                    label=f"{CLASS_NAMES[label]} (n = {len(values):,})")
        if xmin <= 0:
            ax.set_xscale("symlog", linthresh=1, linscale=0.5)
            ax.set_xlim(-0.12, xmax)
        else:
            ax.set_xscale("log")
            ax.set_xlim(xmin, xmax)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:,}" for t in ticks])
        ax.minorticks_off()
        ax.set_ylim(0, 1.03)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_title(source, loc="left", fontsize=11, color=S.INK, pad=8)
        ax.set_xlabel(xlabel, fontsize=9, color=S.INK_2)
        ax.legend(loc=legend_loc, frameon=False, fontsize=9, labelcolor=S.INK_2)
    np.atleast_1d(axes)[0].set_ylabel("Share of class at or below this length", fontsize=9, color=S.INK_2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=S.INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=S.SURFACE)
    return fig


def plot_share_bars(profiled: pd.DataFrame, column: str, title: str, path: Path):
    """Share of URLs with a True/1 value in `column`, by class within each dataset."""
    import matplotlib.pyplot as plt

    sources = list(dict.fromkeys(profiled["source"]))
    shares = profiled.groupby(["source", "label"])[column].mean()
    fig, ax = plt.subplots(figsize=(max(6.4, 2.3 * len(sources) + 1.4), 3.8))
    fig.patch.set_facecolor(S.SURFACE)
    S.style_axis(ax)
    width = 0.34
    for i, source in enumerate(sources):
        for label, offset in ((0, -width / 2 - 0.01), (1, width / 2 + 0.01)):
            value = float(shares.get((source, label), 0.0))
            ax.bar(i + offset, value, width=width, color=_CLASS_COLORS[label],
                   label=CLASS_NAMES[label] if i == 0 else None)
            ax.text(i + offset, value + 0.02, f"{value:.1%}", ha="center", va="bottom",
                    fontsize=9, color=S.INK)
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([str(name).replace(" ", "\n", 1) if len(sources) > 3 else name for name in sources],
                       fontsize=10, color=S.INK)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=S.INK_2, ncol=2)
    ax.set_title(title, loc="left", fontsize=12.5, color=S.INK, fontweight="bold", pad=12)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=S.SURFACE)
    return fig
