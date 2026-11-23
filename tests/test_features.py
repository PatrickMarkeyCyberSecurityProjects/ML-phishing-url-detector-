"""Tests for src/features.py: normalization, lexical features, character encoding.

Every URL here is invented. Hosts use reserved example domains or documentation
IP ranges.
"""

import math
import random
import time

import numpy as np
import pytest

from src import features as F
from src.features import (
    FEATURE_NAMES,
    PAD_ID,
    UNK_ID,
    canonical_key,
    describe_url,
    encode_chars,
    extract_features,
    host_is_ip,
    normalize_url,
    registered_domain,
    shannon_entropy,
    split_url,
    strip_scheme_www,
)

NORMAL = "https://www.example.com/path/to/page?x=1&y=2#frag"


# ------------------------------------------------------------------ normalization

def test_normalize_lowercases_scheme_and_host_only():
    assert normalize_url("HTTP://WWW.Example.COM/Path?Q=A") == "http://www.example.com/Path?Q=A"


def test_normalize_strips_whitespace_and_embedded_newlines():
    assert normalize_url("  \thttps://example.com/a\n b \r\n") == "https://example.com/a b"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("example.com/login", "https://example.com/login"),
        ("//example.com/login", "https://example.com/login"),
        ("example.com:8080/x", "https://example.com:8080/x"),
        ("www.example.com/r?u=http://other.example", "https://www.example.com/r?u=http://other.example"),
    ],
)
def test_missing_scheme_gets_https(raw, expected):
    assert normalize_url(raw) == expected


def test_missing_scheme_gives_same_features_as_explicit_https():
    assert extract_features("example.com/login") == extract_features("https://example.com/login")
    assert np.array_equal(encode_chars("example.com/login"), encode_chars("https://example.com/login"))


def test_host_port_is_not_read_as_a_scheme():
    parts = split_url("example.com:8080/x")
    assert (parts.scheme, parts.host, parts.port, parts.path) == ("https", "example.com", "8080", "/x")


def test_existing_scheme_is_kept():
    assert normalize_url("http://example.com") == "http://example.com/"
    assert normalize_url("ftp://example.com/f") == "ftp://example.com/f"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://example.com", "https://example.com/"),
        ("https://example.com/", "https://example.com/"),
        ("example.com", "https://example.com/"),
        ("https://example.com?a=1", "https://example.com/?a=1"),
        ("https://example.com#top", "https://example.com/#top"),
        ("https://example.com:8080", "https://example.com:8080/"),
        ("https://example.com/a", "https://example.com/a"),
        ("https://example.com//", "https://example.com//"),
    ],
)
def test_empty_path_is_written_as_a_slash(raw, expected):
    assert normalize_url(raw) == expected


def test_empty_path_and_slash_are_the_same_url_to_both_models():
    for a, b in (("https://www.example.com", "https://www.example.com/"),
                 ("http://example.com?a=1", "http://example.com/?a=1")):
        assert extract_features(a) == extract_features(b)
        assert np.array_equal(encode_chars(a), encode_chars(b))
        assert canonical_key(a) == canonical_key(b)


@pytest.mark.parametrize("bad", ["", "   ", "\t\n", None, 123, float("nan"), b"http://example.com"])
def test_empty_and_non_string_inputs_normalize_to_empty(bad):
    assert normalize_url(bad) == ""


def test_normalization_changes_flags():
    flags = F.normalization_changes(" Example.com ")
    assert flags == {
        "whitespace_stripped": True,
        "scheme_added": True,
        "host_lowercased": True,
        "punycode_decoded": False,
        "query_removed": False,
        "root_slash_added": True,
    }
    assert not any(F.normalization_changes("https://example.com/A").values())
    assert not any(F.normalization_changes("https://example.com/").values())
    assert F.normalization_changes("https://example.com?a=1")["root_slash_added"]


def test_normalize_is_idempotent_on_fuzzed_input():
    rng = random.Random(42)
    pieces = [
        "http", "HTTPS", "://", "//", "/", "?", "#", "@", ":", "[", "]", "\\", ".", "-", "%20",
        " ", "\t", "\n", "xn--", "XN--", "xn--80ak6aa92e", "xn--mnchen-3ya", "xn--a", "www.",
        "example", "COM", "8080", "::1", "é", "中", " ", "​", "a", "Z", "0", "=", "&",
    ]
    for _ in range(5000):
        raw = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 12)))
        once = normalize_url(raw)
        assert normalize_url(once) == once, repr(raw)
        # Feature extraction must accept anything normalization produces.
        assert set(extract_features(raw)) == set(FEATURE_NAMES) | {"tld"}


# ------------------------------------------------------------------ main track: query removed

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://example.com/a?x=1&y=2", "https://example.com/a"),
        ("https://example.com/a#top", "https://example.com/a"),
        ("https://example.com/a?x=1#top", "https://example.com/a"),
        ("https://example.com/a#top?x=1", "https://example.com/a"),
        ("https://example.com?x=1", "https://example.com/"),
        ("example.com/login?next=http://other.example/", "https://example.com/login"),
        ("https://example.com/a?", "https://example.com/a"),
        ("https://example.com/a", "https://example.com/a"),
        ("https://user@example.com:8080/a/b?x", "https://user@example.com:8080/a/b"),
    ],
)
def test_strip_query_removes_query_and_fragment(raw, expected):
    assert normalize_url(raw, strip_query=True) == expected
    assert normalize_url(expected, strip_query=True) == expected          # idempotent
    assert normalize_url(normalize_url(raw), strip_query=True) == expected


def test_strip_query_is_off_by_default_and_flagged():
    assert normalize_url("https://example.com/a?x=1") == "https://example.com/a?x=1"
    assert F.normalization_changes("https://example.com/a?x=1", strip_query=True)["query_removed"]
    assert not F.normalization_changes("https://example.com/a?x=1")["query_removed"]
    assert not F.normalization_changes("https://example.com/a", strip_query=True)["query_removed"]
    assert F.MAIN_STRIP_QUERY is True


def test_main_track_features_ignore_the_query():
    with_query = "http://paypal.com.example.net/login?session=abc123&next=https://x.example//y#frag"
    without = "http://paypal.com.example.net/login"
    a, b = extract_features(with_query, strip_query=True), extract_features(without, strip_query=True)
    assert a == b == extract_features(without)
    assert a["query_len"] == 0 and a["n_query_params"] == 0 and a["has_double_slash"] == 0
    assert np.array_equal(encode_chars(with_query, strip_query=True), encode_chars(without))
    full = extract_features(with_query)
    assert full["query_len"] > 0 and full["n_query_params"] == 2 and full["has_double_slash"] == 1
    assert list(a) == list(full)                                          # same 31 names in both tracks


def test_strip_query_idempotent_on_fuzzed_input():
    rng = random.Random(7)
    pieces = ["http", "://", "/", "?", "#", "@", ":", ".", "a", "b=1", "&", "www.", "example", "com", "%3F", " "]
    for _ in range(3000):
        raw = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 12)))
        once = normalize_url(raw, strip_query=True)
        assert normalize_url(once, strip_query=True) == once, repr(raw)
        assert "?" not in once and "#" not in once
        assert normalize_url(once) == once                                # nothing left for the full track to change


def test_feature_matrix_and_encode_batch_match_the_single_url_functions():
    urls = ["https://www.example.com/a?x=1", "example.org", "", None, "http://192.0.2.1:8080/x#y"]
    for strip in (False, True):
        matrix = F.feature_matrix(urls, strip_query=strip)
        assert matrix.shape == (5, len(FEATURE_NAMES)) and matrix.dtype == np.float64
        for row, url in zip(matrix, urls):
            single = extract_features(url, strip_query=strip)
            assert row.tolist() == [float(single[name]) for name in FEATURE_NAMES]
        batch = F.encode_batch(urls, strip_query=strip)
        assert batch.shape == (5, 200) and batch.dtype == np.int64
        for row, url in zip(batch, urls):
            assert np.array_equal(row, encode_chars(url, strip_query=strip))
    assert F.feature_matrix([], strip_query=True).shape == (0, len(FEATURE_NAMES))
    assert F.encode_batch([], max_len=10).shape == (0, 10)


# ------------------------------------------------------------------ punycode

def test_punycode_host_is_decoded_and_flagged():
    url = "https://xn--mnchen-3ya.example/a"
    assert normalize_url(url) == "https://münchen.example/a"
    assert extract_features(url)["has_punycode"] == 1


def test_punycode_is_decoded_when_uppercase():
    assert normalize_url("https://XN--MNCHEN-3YA.example") == "https://münchen.example/"


def test_unicode_host_typed_directly_matches_its_punycode_form():
    assert extract_features("https://münchen.example/a") == extract_features(
        "https://xn--mnchen-3ya.example/a"
    )


def test_malformed_punycode_does_not_crash_and_stays_flagged():
    url = "https://xn--999999999.example/"
    assert normalize_url(url) == url
    assert extract_features(url)["has_punycode"] == 1


def test_punycode_in_path_is_not_decoded():
    url = "https://example.com/xn--mnchen-3ya"
    assert normalize_url(url) == url
    assert extract_features(url)["has_punycode"] == 0


# ------------------------------------------------------------------ IP hosts

@pytest.mark.parametrize(
    "host",
    ["192.0.2.10", "3221225994", "0xC0000202", "0xc0.0x00.0x02.0x0a", "[2001:db8::1]", "[::1]"],
)
def test_ip_hosts_are_detected(host):
    f = extract_features(f"http://{host}/login")
    assert f["host_is_ip"] == 1
    assert f["subdomain_depth"] == 0
    assert f["tld"] == "" and f["tld_len"] == 0


@pytest.mark.parametrize("host", ["example.com", "123.example.com", "1.2.3.4.5", "192.0.2.com", "[nope]", ""])
def test_names_are_not_ip_hosts(host):
    assert host_is_ip(host) is False


def test_ipv6_host_with_port():
    parts = split_url("http://[2001:db8::1]:8443/a")
    assert (parts.host, parts.port, parts.path) == ("[2001:db8::1]", "8443", "/a")
    assert extract_features("http://[2001:db8::1]:8443/a")["has_port"] == 1


def test_malformed_ipv6_does_not_raise():
    assert extract_features("http://[2001:db8::1/a")["host_is_ip"] == 0
    assert split_url("http://[").host == "["


# ------------------------------------------------------------------ features: normal URL

def test_normal_url_known_values():
    f = extract_features(NORMAL)
    assert f["url_len"] == len(NORMAL) == 49
    assert f["host_len"] == len("www.example.com")
    assert f["path_len"] == len("/path/to/page")
    assert f["query_len"] == len("x=1&y=2")
    assert f["n_digits"] == 2
    # https(5) www(3) example(7) com(3) path(4) to(2) page(4) x(1) y(1) frag(4)
    assert f["n_letters"] == 34
    # "://"(3), 2 dots, 3 slashes, and ? & = = #
    assert f["n_special"] == 13
    assert f["digit_ratio"] == pytest.approx(2 / 49)
    assert f["letter_ratio"] + f["digit_ratio"] + f["special_ratio"] == pytest.approx(1.0)
    assert f["subdomain_depth"] == 1
    assert f["path_depth"] == 3
    assert f["n_query_params"] == 2
    assert f["n_dots"] == 2
    assert f["n_hyphens"] == 0
    assert f["tld"] == "com"
    assert (f["tld_len"], f["tld_is_common"], f["tld_is_cc"]) == (3, 1, 0)
    assert f["is_https"] == 1
    for flag in ("host_is_ip", "has_at", "has_double_slash", "has_port", "has_punycode"):
        assert f[flag] == 0, flag
    assert f["n_brand_words"] == 0 and f["n_cred_words"] == 0


def test_bare_homepage_has_only_the_root_path():
    f = extract_features("https://www.example.com")
    assert (f["path_len"], f["path_depth"], f["query_len"]) == (1, 0, 0)
    assert f["url_len"] == len("https://www.example.com/")
    for url in ("https://www.example.com", "https://www.example.com/"):
        d = describe_url(url)
        assert d["is_bare_homepage"] is True and d["has_path"] is False and d["has_query"] is False
    assert describe_url("https://www.example.com/a")["is_bare_homepage"] is False
    assert describe_url("https://www.example.com/a")["has_path"] is True
    query_only = describe_url("https://www.example.com?a=1")
    assert query_only["is_bare_homepage"] is False
    assert query_only["has_path"] is False and query_only["has_query"] is True


def test_flags_fire():
    f = extract_features("http://user@sub-a.example.co.uk:8080/a//b/%2e%2E?x=1")
    assert f["has_at"] == 1
    assert f["has_port"] == 1
    assert f["has_double_slash"] == 1
    assert f["is_https"] == 0
    assert f["n_pct_encoded"] == 2
    assert f["n_host_hyphens"] == 1
    assert f["tld"] == "uk" and f["tld_is_cc"] == 1 and f["tld_is_common"] == 0
    assert f["subdomain_depth"] == 1  # sub-a . example.co.uk


def test_double_slash_after_scheme_only():
    assert extract_features("https://example.com/a/b")["has_double_slash"] == 0
    assert extract_features("https://example.com/r?u=https://other.example")["has_double_slash"] == 1


def test_host_digit_ratio():
    assert extract_features("https://a1b2.example")["host_digit_ratio"] == pytest.approx(2 / 12)


@pytest.mark.parametrize(
    "host, expected",
    [
        ("example.com", "example.com"),
        ("www.example.com", "example.com"),
        ("a.b.example.co.uk", "example.co.uk"),
        ("shop.example.com.au", "example.com.au"),
        ("example.co", "example.co"),
        ("localhost", "localhost"),
        ("192.0.2.10", "192.0.2.10"),
        ("example.com.", "example.com"),
        ("", ""),
    ],
)
def test_registered_domain(host, expected):
    assert registered_domain(host) == expected


def test_entropy():
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("ab") == pytest.approx(1.0)
    assert shannon_entropy("abcd") == pytest.approx(2.0)
    f = extract_features(NORMAL)
    assert f["url_entropy"] == pytest.approx(shannon_entropy(NORMAL))
    assert f["host_entropy"] == pytest.approx(shannon_entropy("www.example.com"))


# ------------------------------------------------------------------ brand and credential words

def test_brand_inside_registered_domain_is_not_counted():
    assert extract_features("https://www.paypal.com/")["n_brand_words"] == 0
    assert extract_features("https://secure-paypal.example/")["n_cred_words"] == 0


def test_brand_and_credential_words_outside_registered_domain_are_counted():
    f = extract_features("http://paypal.com.account-check.example/Login/verify?next=update")
    assert f["n_brand_words"] == 1          # paypal, in the subdomain
    assert f["n_cred_words"] == 3           # login, verify, update; "account" is in the domain


def test_words_are_counted_once_each_and_case_insensitively():
    f = extract_features("https://example.com/LOGIN/login/Login?apple=1&APPLE=2&amazon=3")
    assert f["n_cred_words"] == 1
    assert f["n_brand_words"] == 2


def test_all_credential_words():
    f = extract_features("https://example.com/login/verify/secure/account/update")
    assert f["n_cred_words"] == 5


# ------------------------------------------------------------------ shape of the output

@pytest.mark.parametrize(
    "url",
    ["", None, "   ", "a", "https://", "http://[", "://", "?", "#", "@", "https://example.com",
     "http://192.0.2.10", "https://xn--mnchen-3ya.example", "\ud800", NORMAL],
)
def test_keys_are_constant_and_values_finite(url):
    f = extract_features(url)
    assert list(f) == list(FEATURE_NAMES) + ["tld"]
    assert isinstance(f["tld"], str)
    for name in FEATURE_NAMES:
        value = f[name]
        assert isinstance(value, (int, float)) and not isinstance(value, bool), name
        assert math.isfinite(value) and value >= 0, name


@pytest.mark.parametrize("empty", ["", "  ", None])
def test_empty_input_gives_all_zeros(empty):
    f = extract_features(empty)
    assert all(f[name] == 0 for name in FEATURE_NAMES)
    assert f["tld"] == ""


def test_feature_names_are_unique_and_frozen_at_31():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    assert len(FEATURE_NAMES) == 31          # frozen: see docs/design_decisions.md


def test_extraction_is_deterministic_and_idempotent_under_normalization():
    assert extract_features(NORMAL) == extract_features(NORMAL)
    raw = "  HTTP://WWW.Example.com/A  "
    assert extract_features(raw) == extract_features(normalize_url(raw))


def test_very_long_url():
    url = "https://example.com/" + "a1-" * 40_000 + "?q=" + "%41" * 5_000
    assert len(url) > 100_000
    start = time.perf_counter()
    f = extract_features(url)
    codes = encode_chars(url)
    elapsed = time.perf_counter() - start
    assert f["url_len"] == len(url)
    assert f["n_pct_encoded"] == 5_000
    assert f["n_hyphens"] == 40_000
    assert codes.shape == (200,) and (codes != PAD_ID).all()
    assert elapsed < 2.0


# ------------------------------------------------------------------ ablation helpers

def test_strip_scheme_www():
    assert strip_scheme_www("https://www.example.com/a") == "example.com/a"
    assert strip_scheme_www("http://example.com/a") == "example.com/a"
    assert strip_scheme_www("https://www2.example.com") == "www2.example.com/"
    assert strip_scheme_www("") == ""


def test_stripped_variant_hides_scheme_and_www_from_the_models():
    a = extract_features(strip_scheme_www("http://example.com/a"))
    b = extract_features(strip_scheme_www("https://www.example.com/a"))
    assert a == b


def test_canonical_key_ignores_scheme_www_and_trailing_slash():
    assert canonical_key("http://example.com") == canonical_key("https://www.example.com/")
    assert canonical_key("https://example.com/a") != canonical_key("https://example.com/b")


# ------------------------------------------------------------------ character encoding

def test_encode_chars_shape_dtype_and_padding():
    codes = encode_chars("https://example.com")
    assert codes.shape == (200,)
    assert codes.dtype == np.int64
    n = len("https://example.com/")
    assert (codes[:n] >= 2).all() and (codes[n:] == PAD_ID).all()


def test_encode_chars_values():
    codes = encode_chars("https://a.b/ ~", max_len=16)
    assert codes[0] == ord("h") - 30
    assert codes[12] == 2                     # space, the lowest printable character
    assert codes[13] == F.VOCAB_SIZE - 1      # '~', the highest
    assert codes.max() < F.VOCAB_SIZE


def test_encode_chars_truncates_to_the_head():
    url = "https://example.com/" + "a" * 500
    codes = encode_chars(url, max_len=50)
    assert codes.shape == (50,)
    assert np.array_equal(codes, encode_chars(url[:50], max_len=50))


def test_encode_chars_non_ascii_is_unknown():
    codes = encode_chars("https://münchen.example")
    assert codes[len("https://m")] == UNK_ID
    assert (encode_chars("https://xn--mnchen-3ya.example") == codes).all()


@pytest.mark.parametrize("empty", ["", None, "  "])
def test_encode_chars_empty_is_all_padding(empty):
    assert (encode_chars(empty) == PAD_ID).all()


def test_encode_chars_respects_max_len():
    assert encode_chars("https://example.com", max_len=5).tolist() == [
        ord(c) - 30 for c in "https"
    ]


def test_encode_chars_survives_lone_surrogate():
    assert encode_chars("https://example.com/\ud800")[len("https://example.com/")] == UNK_ID
