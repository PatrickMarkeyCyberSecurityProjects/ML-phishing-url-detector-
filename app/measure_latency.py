"""Send 100 invented URLs to a running copy of the app and record how long each takes.

    docker run --rm -d -p 7860:7860 --name phishing-url-scorer phishing-url-scorer
    python app/measure_latency.py http://localhost:7860 "docker image phishing-url-scorer on the development machine"

Writes reports/latency.json, or the file named as a third argument (use one for
a hosted Space, so the local measurement is kept). The 100 URLs are sent twice, once per client:
    plain HTTP       Gradio's two-step HTTP API over one kept-alive connection (httpx)
    gradio_client    the Python client library, used as its documentation shows
For each URL the round trip is timed by the client (HTTP request, Gradio queue,
scoring, response), and the app reports two times of its own:
    scoring_ms       normalization, features and both models
    explanation_ms   Model A's feature contributions (pred_contrib)

The URLs are invented and sit on names reserved for documentation. A development
tool: it needs gradio_client, which is installed with gradio, and is not part of
the deployed image.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N_URLS = 100
WARM_UP = 5
SEED = 42


def invented_urls(n: int = N_URLS, seed: int = SEED) -> list[str]:
    """`n` invented URLs of mixed shapes on example.com, example.net, example.org and 192.0.2.0/24."""
    rng = np.random.default_rng(seed)
    schemes = ("https://", "http://", "")
    subdomains = ("", "www.", "docs.", "shop.", "secure-login.", "account-verify.", "mail.", "cdn.static.",
                  "paypal.com.account-check.", "support-update.", "my.portal.")
    domains = ("example.com", "example.net", "example.org")
    paths = ("", "/", "/about/team", "/guides/getting-started/installation.html", "/products/item-48213",
             "/signin/", "/verify/login.php", "/wp-admin/update/index.html", "/news/2025/03/annual-report.html",
             "/a/b/c/d/e/f/index.html", "/account/session/8f2a91c7d0/confirm", "/blog/company-news")
    queries = ("", "", "", "?ref=newsletter&utm_source=mail", "?session=8f2a", "#top")
    urls = []
    for i in range(n):
        if i % 20 == 19:
            host = f"192.0.2.{int(rng.integers(1, 255))}"
        else:
            host = subdomains[int(rng.integers(len(subdomains)))] + domains[int(rng.integers(len(domains)))]
        urls.append(schemes[int(rng.integers(len(schemes)))] + host + paths[int(rng.integers(len(paths)))]
                    + queries[int(rng.integers(len(queries)))])
    return urls


def summary(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()),
            "max_ms": float(a.max())}


def _timed(call, urls: list[str]) -> list[dict]:
    for url in urls[:WARM_UP]:                                    # not counted
        call(url)
    rows = []
    for url in urls:
        started = time.perf_counter()
        result = call(url)
        elapsed = (time.perf_counter() - started) * 1000.0
        if not result.get("ok"):
            raise SystemExit(f"The app rejected {url!r}: {result.get('error')}")
        rows.append({"round_trip_ms": elapsed, "scoring_ms": result["scoring_ms"],
                     "explanation_ms": result["explanation_ms"], "in_app_total_ms": result["latency_ms"],
                     "bands": [result["models"]["lgbm"]["band"], result["models"]["cnn"]["band"]]})
    return rows


def measure(base_url: str, environment: str) -> dict:
    import httpx
    from gradio_client import Client

    urls = invented_urls()
    api = base_url.rstrip("/") + "/gradio_api/call/score"
    with httpx.Client(timeout=60) as http:
        def plain(url: str) -> dict:
            event = http.post(api, json={"data": [url]}).json()["event_id"]
            with http.stream("GET", f"{api}/{event}") as stream:
                body = "".join(stream.iter_text())
            data = next(line[len("data: "):] for line in reversed(body.splitlines()) if line.startswith("data: "))
            return json.loads(data)[0]

        rows = _timed(plain, urls)
    client = Client(base_url, verbose=False)
    library_rows = _timed(lambda url: client.predict(url, api_name="/score"), urls)
    bands = {}
    for row in rows:
        for model, band in zip(("lgbm", "cnn"), row["bands"]):
            bands.setdefault(model, {}).setdefault(band, 0)
            bands[model][band] += 1
    return {
        "measured": time.strftime("%Y-%m-%d"),
        "environment": environment,
        "client": f"Python on the same machine ({platform.system()} {platform.release()}), outside the container",
        "base_url": base_url,
        "urls": len(rows),
        "warm_up_requests_not_counted": WARM_UP,
        "requests": "one at a time, in sequence",
        "note": "Invented URLs on reserved example names. Round trip includes HTTP and the Gradio queue; "
                "the in-app times are reported by the app itself, from the plain-HTTP run. Nothing here "
                "measures a hosted Space or more than one request at a time.",
        "round_trip": summary([r["round_trip_ms"] for r in rows]),
        "round_trip_gradio_client": summary([r["round_trip_ms"] for r in library_rows]),
        "in_app_total": summary([r["in_app_total_ms"] for r in rows]),
        "scoring_both_models": summary([r["scoring_ms"] for r in rows]),
        "explanation_pred_contrib": summary([r["explanation_ms"] for r in rows]),
        "bands_of_the_invented_urls": bands,
    }


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
    where = sys.argv[2] if len(sys.argv) > 2 else "not stated"
    out = measure(base, where)
    path = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else ROOT / "reports" / "latency.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    for key in ("round_trip", "round_trip_gradio_client", "in_app_total", "scoring_both_models",
                "explanation_pred_contrib"):
        print(f"{key:26s} median {out[key]['median_ms']:7.2f} ms   p95 {out[key]['p95_ms']:7.2f} ms")
    print("saved", path)
