"""URL normalization and feature extraction: the single source for the project.

Training notebooks, evaluation and the deployed app all import this module, so a
URL is turned into model inputs by exactly the same code at training time and at
serving time. It needs only numpy and the standard library, because the app image
must stay small.

Nothing here fetches a URL. Every function analyses the string only.

Two tracks
    main   the query string and fragment are removed before anything else
           (`strip_query=True`). The deployed models and the app use this track.
    full   the URL is kept whole (`strip_query=False`). Used for one labelled
           comparison model and for data analysis.
The training data's legitimate URLs had their query parameters removed by the
collector, so in that data "has a query" marks phishing only because of how it
was collected. The main track removes the query from every URL so that no model
can learn it. See docs/design_decisions.md.

Public functions
    normalize_url(url, strip_query)      -> str         cleaned URL
    extract_features(url, strip_query)   -> dict        31 lexical features for LightGBM
    encode_chars(url, 200, strip_query)  -> np.ndarray  character ids for the CNN
    feature_matrix(urls, strip_query)    -> np.ndarray  (n, 31) features, in FEATURE_NAMES order
    encode_batch(urls, 200, strip_query) -> np.ndarray  (n, 200) character ids
    strip_scheme_www(url)                -> str         scheme and `www.` removed
    describe_url(url)                    -> dict        structural parts, for analysis only
"""

from __future__ import annotations

import codecs
import ipaddress
import math
import re
from collections import Counter
from typing import NamedTuple

import numpy as np

# --------------------------------------------------------------------------- #
# Fixed lists. These were written from general knowledge before any PhreshPhish
# row was inspected, and they are not fitted to either dataset.
# --------------------------------------------------------------------------- #

#: Scheme assumed when a URL arrives without one, chosen before any data was profiled.
DEFAULT_SCHEME = "https"

#: The main track removes the query string and fragment. Training, evaluation
#: and the app all pass this value, so they cannot disagree.
MAIN_STRIP_QUERY = True

#: Words that ask for credentials. Counted only outside the registered domain.
CREDENTIAL_WORDS = ("login", "verify", "secure", "account", "update")

#: Frequently impersonated brands. Counted only outside the registered domain,
#: so `paypal.com` scores 0 and `paypal.com.example.net` scores 1.
BRAND_WORDS = (
    "paypal", "microsoft", "office365", "outlook", "onedrive", "sharepoint",
    "apple", "icloud", "amazon", "netflix", "facebook", "instagram", "whatsapp",
    "google", "gmail", "linkedin", "dropbox", "docusign", "adobe", "wellsfargo",
    "bankofamerica", "citibank", "hsbc", "santander", "dhl", "fedex", "usps",
    "coinbase", "binance", "metamask", "yahoo", "ebay", "twitter", "telegram",
    "steam", "roblox", "spotify",
)

#: Long-established generic top-level domains.
COMMON_TLDS = frozenset({"com", "org", "net", "edu", "gov", "mil", "int"})

#: Second-level labels that act as part of the public suffix under a country
#: code, for example `co.uk` or `com.au`. This replaces a public-suffix list
#: (which would be an extra dependency that downloads data). It is approximate.
_CC_SECOND_LEVEL = frozenset(
    {"co", "com", "net", "org", "gov", "edu", "ac", "or", "ne", "go", "mil", "nom", "sch"}
)

#: Numeric model features, in a fixed order. `extract_features` also returns
#: `tld` (a string) for analysis; it is not a model input.
FEATURE_NAMES = (
    "url_len", "host_len", "path_len", "query_len",
    "n_digits", "n_letters", "n_special",
    "digit_ratio", "letter_ratio", "special_ratio", "host_digit_ratio",
    "url_entropy", "host_entropy",
    "subdomain_depth", "path_depth", "n_query_params",
    "n_dots", "n_hyphens", "n_host_hyphens", "n_pct_encoded",
    "tld_len", "tld_is_common", "tld_is_cc",
    "host_is_ip", "has_at", "has_double_slash", "has_port", "has_punycode",
    "is_https",
    "n_brand_words", "n_cred_words",
)

# Character vocabulary for the CNN: 0 = padding, 1 = unknown (anything that is
# not printable ASCII, including decoded punycode), 2..96 = ASCII 32..126.
PAD_ID = 0
UNK_ID = 1
_ASCII_LO, _ASCII_HI = 32, 126
_CHAR_OFFSET = _ASCII_LO - 2
VOCAB_SIZE = (_ASCII_HI - _ASCII_LO + 1) + 2
DEFAULT_MAX_LEN = 200

_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*)://")
_AUTHORITY_END_RE = re.compile(r"[/?#\\]")
_PUNYCODE_LABEL_RE = re.compile(r"^xn--[a-z0-9\-]+$")
_IPV4_PART_RE = re.compile(r"^(?:0x[0-9a-f]*|[0-9]+)$")
_PCT_RE = re.compile(r"%[0-9A-Fa-f]{2}")
_DROP_TAB_NEWLINE = {ord("\t"): None, ord("\r"): None, ord("\n"): None}
_DIGITS = "0123456789"


class UrlParts(NamedTuple):
    """The pieces of a normalized URL. Every field is a string, possibly empty."""

    scheme: str
    userinfo: str
    host: str
    port: str
    path: str
    query: str
    fragment: str


_EMPTY_PARTS = UrlParts("", "", "", "", "", "", "")


# --------------------------------------------------------------------------- #
# Parsing and normalization
# --------------------------------------------------------------------------- #

def _split_authority(authority: str) -> tuple[str, str, str]:
    """Split `user@host:port` into (prefix, host, suffix).

    `prefix` keeps the trailing '@' and `suffix` keeps the leading ':', so the
    three pieces always join back to the input.
    """
    at = authority.rfind("@")
    prefix = authority[: at + 1] if at != -1 else ""
    hostport = authority[at + 1:]
    if hostport.startswith("["):  # IPv6 literal
        end = hostport.find("]")
        if end != -1:
            return prefix, hostport[: end + 1], hostport[end + 1:]
        return prefix, hostport, ""
    colon = hostport.rfind(":")
    if colon != -1:
        tail = hostport[colon + 1:]
        if tail == "" or (tail.isascii() and tail.isdigit()):
            return prefix, hostport[:colon], hostport[colon:]
    return prefix, hostport, ""


def _decode_punycode_label(label: str) -> str:
    """Decode one `xn--` host label to Unicode, or return it unchanged.

    The label is left alone when decoding fails or when the result could change
    how the URL parses on a second pass (pure ASCII, whitespace, control
    characters, or another `xn--` prefix). That keeps normalization idempotent.
    """
    if not _PUNYCODE_LABEL_RE.match(label):
        return label
    try:
        decoded = codecs.decode(label[4:].encode("ascii"), "punycode").lower()
    except Exception:  # malformed punycode: keep the label as given
        return label
    if not decoded or decoded.isascii() or decoded.startswith("xn--"):
        return label
    if any(ch.isspace() or not ch.isprintable() for ch in decoded):
        return label
    return decoded


def _normalize_host(host: str) -> tuple[str, bool]:
    """Lowercase a host and decode punycode labels. Returns (host, decoded?)."""
    lowered = host.lower()
    if lowered.startswith("["):
        return lowered, False
    labels = lowered.split(".")
    decoded = [_decode_punycode_label(label) for label in labels]
    return ".".join(decoded), decoded != labels


def _normalize(url: object, strip_query: bool = False) -> tuple[str, dict[str, bool]]:
    flags = {
        "whitespace_stripped": False,
        "scheme_added": False,
        "host_lowercased": False,
        "punycode_decoded": False,
        "query_removed": False,
        "root_slash_added": False,
    }
    if not isinstance(url, str):
        return "", flags
    s = url.strip().translate(_DROP_TAB_NEWLINE)
    flags["whitespace_stripped"] = s != url
    if not s:
        return "", flags

    match = _SCHEME_RE.match(s)
    if match is None:
        s = f"{DEFAULT_SCHEME}:{s}" if s.startswith("//") else f"{DEFAULT_SCHEME}://{s}"
        flags["scheme_added"] = True
        match = _SCHEME_RE.match(s)
    scheme = match.group(1).lower()
    after = s[match.end():]

    end = _AUTHORITY_END_RE.search(after)
    authority, rest = (after[: end.start()], after[end.start():]) if end else (after, "")
    prefix, host, suffix = _split_authority(authority)
    new_host, decoded = _normalize_host(host)
    flags["host_lowercased"] = host != host.lower()
    flags["punycode_decoded"] = decoded
    if strip_query:
        cut = min((i for i in (rest.find("?"), rest.find("#")) if i != -1), default=-1)
        if cut != -1:
            # rstrip: whitespace that sat before the "?" would otherwise be
            # stripped on a second pass, and normalization must be idempotent.
            rest = rest[:cut].rstrip()
            flags["query_removed"] = True
    # An empty path and "/" are the same address, so the path is always written.
    if rest == "" or rest[0] in "?#":
        rest = "/" + rest
        flags["root_slash_added"] = True
    return f"{scheme}://{prefix}{new_host}{suffix}{rest}", flags


def normalize_url(url: object, strip_query: bool = False) -> str:
    """Return the cleaned form of a URL.

    Steps, in order:
      1. strip surrounding whitespace and remove embedded tab and newline
         characters;
      2. if there is no `scheme://`, prepend `https://`;
      3. lowercase the scheme and the host;
      4. decode punycode (`xn--`) host labels to Unicode;
      5. write an empty path as "/", so `https://example.com` and
         `https://example.com/` are one URL (`https://example.com?a=1` becomes
         `https://example.com/?a=1`).
    Userinfo, port, query and fragment are left exactly as given, and so is any
    path that is not empty.

    With `strip_query=True` (the main track) everything from the first "?" or
    "#" onward is removed before step 5, so `https://example.com/a?x=1#top`
    becomes `https://example.com/a`.

    The function is idempotent, never raises, and returns "" for an empty or
    non-string input.
    """
    return _normalize(url, strip_query)[0]


def normalization_changes(url: object, strip_query: bool = False) -> dict[str, bool]:
    """Which normalization steps changed this URL. Used for reporting only."""
    return _normalize(url, strip_query)[1]


def _split(u: str) -> UrlParts:
    """Split an already-normalized URL into its parts."""
    match = _SCHEME_RE.match(u)
    if match is None:
        return _EMPTY_PARTS
    scheme = match.group(1)
    after = u[match.end():]
    end = _AUTHORITY_END_RE.search(after)
    authority, rest = (after[: end.start()], after[end.start():]) if end else (after, "")
    prefix, host, suffix = _split_authority(authority)
    port = suffix[1:] if suffix.startswith(":") and suffix[1:].isdigit() else ""
    rest, _, fragment = rest.partition("#")
    path, _, query = rest.partition("?")
    return UrlParts(scheme, prefix[:-1], host, port, path, query, fragment)


def split_url(url: object) -> UrlParts:
    """Normalize a URL and split it into parts. Never raises."""
    return _split(normalize_url(url))


def host_is_ip(host: str) -> bool:
    """True when the host is an IP address rather than a name.

    Covers bracketed IPv6, dotted IPv4, and the single-number and hexadecimal
    IPv4 forms that browsers accept (for example `3232235777` or `0xC0A80101`).
    """
    if not host:
        return False
    if host.startswith("[") and host.endswith("]"):
        try:
            ipaddress.IPv6Address(host[1:-1].split("%")[0])
            return True
        except ValueError:
            return False
    parts = host.rstrip(".").lower().split(".")
    if not 1 <= len(parts) <= 4:
        return False
    return all(part and _IPV4_PART_RE.match(part) for part in parts)


def registered_domain(host: str) -> str:
    """Approximate registered domain of a host, using the standard library only.

    The last two labels, or the last three when the host ends in a country code
    preceded by a common second-level label (`example.co.uk`). An IP address or
    a single-label host is returned as is. Without a public-suffix list this is
    an approximation: hosting suffixes such as `github.io` count as the
    registered domain.
    """
    host = host.rstrip(".")
    if not host or host_is_ip(host):
        return host
    labels = [label for label in host.split(".") if label]
    if len(labels) <= 2:
        return ".".join(labels)
    tld, second = labels[-1], labels[-2]
    if len(tld) == 2 and tld.isascii() and tld.isalpha() and second in _CC_SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _tld(host: str) -> str:
    host = host.rstrip(".")
    if not host or host_is_ip(host) or "." not in host:
        return ""
    return host.rsplit(".", 1)[1]


def shannon_entropy(s: str) -> float:
    """Shannon entropy of a string in bits per character. 0.0 for ""."""
    n = len(s)
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def strip_scheme_www(url: object) -> str:
    """Ablation variant: the normalized URL without its scheme and leading `www.`.

    Every legitimate PhiUSIIL URL begins `https://www.`, which lets a model
    separate that dataset's classes on the prefix alone. Passing a URL through
    this function before `extract_features` or `encode_chars` removes the
    prefix: both then re-add the same default scheme to every URL, so the
    scheme and `www.` carry no information.
    """
    u = normalize_url(url)
    match = _SCHEME_RE.match(u)
    if match is None:
        return ""
    after = u[match.end():]
    end = _AUTHORITY_END_RE.search(after)
    authority, rest = (after[: end.start()], after[end.start():]) if end else (after, "")
    prefix, host, suffix = _split_authority(authority)
    if host.startswith("www.") and len(host) > 4:
        host = host[4:]
    return f"{prefix}{host}{suffix}{rest}"


def canonical_key(url: object) -> str:
    """Loose identity of a URL, for finding the same page across datasets.

    Ignores the scheme, a leading `www.` and trailing slashes, so
    `http://example.com` and `https://www.example.com/` share one key.
    """
    return strip_scheme_www(url).rstrip("/")


def describe_url(url: object) -> dict[str, object]:
    """Structural facts about a URL, for data analysis and leakage checks.

    These are not model features. `is_bare_homepage` is True when the URL has
    no path beyond "/", no query and no fragment. `has_path` is True when the
    path is longer than "/".
    """
    p = split_url(url)
    return {
        "scheme": p.scheme,
        "host": p.host,
        "registered_domain": registered_domain(p.host),
        "tld": _tld(p.host),
        "has_www": p.host.startswith("www."),
        "is_bare_homepage": p.path in ("", "/") and not p.query and not p.fragment,
        "has_path": p.path not in ("", "/"),
        "has_query": bool(p.query),
    }


# --------------------------------------------------------------------------- #
# Model inputs
# --------------------------------------------------------------------------- #

def _count_words(text: str, words: tuple[str, ...]) -> int:
    return sum(1 for word in words if word in text)


def extract_features(url: object, strip_query: bool = False) -> dict[str, object]:
    """Lexical features of one URL, computed from the string alone.

    With `strip_query=True` (the main track) the query string and fragment are
    removed first, so `query_len` and `n_query_params` are always 0.

    The URL is normalized first, so callers may pass raw or normalized URLs and
    get the same result. The returned dict always has the same keys:
    every name in FEATURE_NAMES (numeric and finite) plus `tld` (a string kept
    for analysis). An empty or non-string input gives all zeros.

    Lengths count characters of the normalized URL, so a URL with no path has
    `path_len` 1 (the "/" that normalization writes). Ratios are shares of the
    whole URL, except `host_digit_ratio`, which is a share of the host.
    `n_brand_words` and `n_cred_words` count distinct words from BRAND_WORDS and
    CREDENTIAL_WORDS found anywhere except the registered domain.
    """
    u = normalize_url(url, strip_query)
    p = _split(u)
    host = p.host
    n = len(u)

    n_digits = sum(u.count(d) for d in _DIGITS)
    n_letters = sum(map(str.isalpha, u))
    n_special = n - n_digits - n_letters
    host_digits = sum(host.count(d) for d in _DIGITS)

    is_ip = host_is_ip(host)
    reg = registered_domain(host)
    bare_host = host.rstrip(".")
    if is_ip or not reg:
        subdomain = ""
        subdomain_depth = 0
    else:
        subdomain = bare_host[: -len(reg)].rstrip(".") if bare_host.endswith(reg) else ""
        subdomain_depth = len([label for label in subdomain.split(".") if label])
    tld = _tld(host)

    after_authority = u[len(p.scheme) + 3:] if p.scheme else ""
    outside_domain = " ".join((p.userinfo, subdomain, p.path, p.query, p.fragment)).lower()

    return {
        "url_len": n,
        "host_len": len(host),
        "path_len": len(p.path),
        "query_len": len(p.query),
        "n_digits": n_digits,
        "n_letters": n_letters,
        "n_special": n_special,
        "digit_ratio": n_digits / n if n else 0.0,
        "letter_ratio": n_letters / n if n else 0.0,
        "special_ratio": n_special / n if n else 0.0,
        "host_digit_ratio": host_digits / len(host) if host else 0.0,
        "url_entropy": shannon_entropy(u),
        "host_entropy": shannon_entropy(host),
        "subdomain_depth": subdomain_depth,
        "path_depth": len([seg for seg in p.path.split("/") if seg]),
        "n_query_params": len([kv for kv in p.query.split("&") if kv]),
        "n_dots": u.count("."),
        "n_hyphens": u.count("-"),
        "n_host_hyphens": host.count("-"),
        "n_pct_encoded": len(_PCT_RE.findall(u)),
        "tld_len": len(tld),
        "tld_is_common": int(tld in COMMON_TLDS),
        "tld_is_cc": int(len(tld) == 2 and tld.isascii() and tld.isalpha()),
        "host_is_ip": int(is_ip),
        "has_at": int("@" in u),
        "has_double_slash": int("//" in after_authority),
        "has_port": int(bool(p.port)),
        "has_punycode": int((not host.isascii()) or "xn--" in host),
        "is_https": int(p.scheme == "https"),
        "n_brand_words": _count_words(outside_domain, BRAND_WORDS),
        "n_cred_words": _count_words(outside_domain, CREDENTIAL_WORDS),
        "tld": tld,
    }


def encode_chars(url: object, max_len: int = DEFAULT_MAX_LEN, strip_query: bool = False) -> np.ndarray:
    """Encode a URL as a fixed-length array of character ids for the CNN.

    The URL is normalized, cut to its first `max_len` characters and padded on
    the right with PAD_ID (0). Printable ASCII maps to 2..96. Every other
    character, including decoded punycode, maps to UNK_ID (1).
    Returns an int64 array of shape (max_len,).
    """
    out = np.full(max_len, PAD_ID, dtype=np.int64)
    u = normalize_url(url, strip_query)[:max_len]
    if not u:
        return out
    # surrogatepass: a lone surrogate in user input must not raise here.
    codes = np.frombuffer(u.encode("utf-32-le", "surrogatepass"), dtype="<u4").astype(np.int64)
    printable = (codes >= _ASCII_LO) & (codes <= _ASCII_HI)
    out[: len(codes)] = np.where(printable, codes - _CHAR_OFFSET, UNK_ID)
    return out



def feature_matrix(urls, strip_query: bool = False) -> np.ndarray:
    """Features of many URLs as a float64 array of shape (n, 31), in FEATURE_NAMES order."""
    rows = [extract_features(u, strip_query) for u in urls]
    return np.array([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float64).reshape(
        len(rows), len(FEATURE_NAMES)
    )


def encode_batch(urls, max_len: int = DEFAULT_MAX_LEN, strip_query: bool = False) -> np.ndarray:
    """Character ids of many URLs as an int64 array of shape (n, max_len)."""
    urls = list(urls)
    out = np.full((len(urls), max_len), PAD_ID, dtype=np.int64)
    for i, u in enumerate(urls):
        out[i] = encode_chars(u, max_len, strip_query)
    return out
