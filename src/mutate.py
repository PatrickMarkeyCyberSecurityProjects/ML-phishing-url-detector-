"""Mutation of URLs for two robustness tests.

Random edits are the first half of this module. Targeted edits,
chosen to look legitimate, are the second half; they were added after the test
results had been seen (docs/design_decisions.md, decision 23).

Random edits. The question they answer: how quickly does detection fall when a phishing URL
is changed a little, the way a new registration of a near-identical name would
change it? `mutate_url(url, strength, rng)` applies `strength` random edits to
the host name and path. The edits are random, not chosen to beat the models,
so this measures brittleness and not the best an attacker could do.

Edits (one is drawn at random for each step):
    lookalike   replace a letter with a similar-looking character (o -> 0, l -> 1 ...)
    substitute  replace a letter or digit with a random one
    insert      insert a random letter or digit
    delete      remove a letter or digit
    swap        exchange two neighbouring characters

The scheme, the dots, the top-level domain and the slashes are never touched,
so a mutated URL keeps its shape (same scheme, same number of host labels and
path segments).
"""

from __future__ import annotations

import numpy as np

from src import features as F

LOOKALIKES = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "4", "s": "5", "b": "8", "g": "9", "t": "7", "z": "2",
              "0": "o", "1": "l", "3": "e", "5": "s"}
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
EDITS = ("lookalike", "substitute", "insert", "delete", "swap")


def editable_positions(url: str) -> list[int]:
    """Indexes of the letters and digits that may be edited.

    These are the characters of the host (not its last label, the top-level
    domain) and of the path. An IP-address host is left alone.
    """
    normalized = F.normalize_url(url)
    parts = F.split_url(normalized)
    after_scheme = len(parts.scheme) + 3
    end = F._AUTHORITY_END_RE.search(normalized, after_scheme)
    path_start = end.start() if end else len(normalized)
    positions = []
    start = normalized.find(parts.host, after_scheme, path_start) if parts.host else -1
    if start != -1 and not F.host_is_ip(parts.host):
        last_dot = parts.host.rfind(".")
        stop = start + (last_dot if last_dot != -1 else len(parts.host))       # leave the top-level domain alone
        positions += [i for i in range(start, stop) if normalized[i].isascii() and normalized[i].isalnum()]
    path_end = path_start + len(parts.path)
    positions += [i for i in range(path_start, path_end) if normalized[i].isascii() and normalized[i].isalnum()]
    return positions


def mutate_url(url: str, strength: int, rng: np.random.Generator) -> str:
    """Apply `strength` random edits to a URL. `strength=0` returns it normalized."""
    current = F.normalize_url(url, strip_query=F.MAIN_STRIP_QUERY)
    for _ in range(int(strength)):
        positions = editable_positions(current)
        if not positions:
            break
        i = int(positions[int(rng.integers(len(positions)))])
        edit = EDITS[int(rng.integers(len(EDITS)))]
        ch = current[i]
        if edit == "lookalike":
            new = LOOKALIKES.get(ch.lower(), ALPHABET[int(rng.integers(len(ALPHABET)))])
            current = current[:i] + new + current[i + 1:]
        elif edit == "substitute":
            current = current[:i] + ALPHABET[int(rng.integers(len(ALPHABET)))] + current[i + 1:]
        elif edit == "insert":
            current = current[:i] + ALPHABET[int(rng.integers(len(ALPHABET)))] + current[i:]
        elif edit == "delete":
            if len(positions) > 1:               # never delete the last editable character
                current = current[:i] + current[i + 1:]
        elif edit == "swap":
            if i + 1 in positions:
                current = current[:i] + current[i + 1] + current[i] + current[i + 2:]
        current = F.normalize_url(current, strip_query=F.MAIN_STRIP_QUERY)
    return current


def mutate_many(urls, strength: int, seed: int = 42) -> list[str]:
    """Mutate every URL at one strength, reproducibly."""
    rng = np.random.default_rng([seed, int(strength)])
    return [mutate_url(u, strength, rng) for u in urls]


# --------------------------------------------------------------------------- #
# Targeted edits: chosen to look legitimate, not random.
#
# Added after the test results had been seen (docs/design_decisions.md,
# decision 23). The edits and the five paths below were written into that
# document before the test was run, and must not be changed afterwards.
# --------------------------------------------------------------------------- #

TARGETED_EDITS = {
    "www": "add a www. prefix to the host name",
    "https": "switch the scheme to https",
    "path": "append a benign-looking path",
    "all": "all three together",
}
BENIGN_PATHS = (
    "/about-us",
    "/en/support/contact",
    "/blog/2025/03/company-news",
    "/products/category/item-details.html",
    "/news/articles/annual-report-2024.html",
)


def _pieces(url: object) -> tuple[str, str, str, str, str] | None:
    """(scheme, userinfo with its '@', host, port with its ':', everything after) of the main-track URL."""
    normalized = F.normalize_url(url, strip_query=F.MAIN_STRIP_QUERY)
    match = F._SCHEME_RE.match(normalized)
    if match is None:
        return None
    after = normalized[match.end():]
    end = F._AUTHORITY_END_RE.search(after)
    authority, rest = (after[: end.start()], after[end.start():]) if end else (after, "")
    prefix, host, suffix = F._split_authority(authority)
    return match.group(1), prefix, host, suffix, rest


def targeted_edit(url: object, edit: str, benign_path: str = BENIGN_PATHS[0]) -> str:
    """Apply one targeted edit to a URL and return the main-track form of the result.

    www     `www.` is put in front of the host, unless the host already begins
            with `www.` or is an IP address
    https   the scheme becomes `https`
    path    `benign_path` is appended to the existing path, after removing any
            trailing "/" so that no "//" is created
    all     the three together
    """
    if edit not in TARGETED_EDITS:
        raise ValueError(f"Unknown targeted edit: {edit!r}")
    pieces = _pieces(url)
    if pieces is None:
        return ""
    scheme, prefix, host, suffix, rest = pieces
    if edit in ("www", "all") and host and not host.startswith("www.") and not F.host_is_ip(host):
        host = "www." + host
    if edit in ("https", "all"):
        scheme = "https"
    if edit in ("path", "all"):
        cut = min((i for i in (rest.find("?"), rest.find("#")) if i != -1), default=len(rest))
        rest = rest[:cut].rstrip("/") + benign_path + rest[cut:]
    return F.normalize_url(f"{scheme}://{prefix}{host}{suffix}{rest}", strip_query=F.MAIN_STRIP_QUERY)


def benign_path_draw(n: int, seed: int = 42) -> np.ndarray:
    """Which of the five benign paths each of `n` URLs gets. The same draw serves `path` and `all`."""
    return np.random.default_rng(seed).integers(0, len(BENIGN_PATHS), size=n)


def targeted_many(urls, edit: str, seed: int = 42) -> list[str]:
    """Apply one targeted edit to every URL, reproducibly."""
    urls = list(urls)
    draw = benign_path_draw(len(urls), seed)
    return [targeted_edit(u, edit, BENIGN_PATHS[int(i)]) for u, i in zip(urls, draw)]
