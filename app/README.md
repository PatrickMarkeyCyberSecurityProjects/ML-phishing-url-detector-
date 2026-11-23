---
title: Phishing URL Scorer
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Scores the text of a URL. It never opens the link.
suggested_hardware: cpu-basic
tags:
  - phishing
  - url
  - security
  - lightgbm
  - onnx
datasets:
  - phreshphish/phreshphish
---

# Phishing URL scorer

Scores the **text of a URL** with two machine-learning models. Each model gives
one of three results: **Block** (score above the threshold set for a 0.1%
false-positive rate on validation), **Review** (above the 1% threshold) or
**Allow** (below both; not proof that the URL is safe). The URL is never opened,
fetched or resolved.

- **Model A:** LightGBM on 31 features of the URL string.
- **Model B:** a character-level CNN, run with ONNX Runtime on CPU.

Both were trained on the PhreshPhish train split and frozen before any test set
was scored. The report gives the result line and a per-model detection table
(result, risk score out of 100, raw score), handling guidance, notes, URL
details, Model A's top five feature contributions, the measured rates of each
result, limitations and scoring time. It gives no combined verdict, and a risk
score is not a probability.

URL details break the URL into its parts (host, approximate registered domain,
top-level domain, subdomain levels, port, userinfo, path) and list observed
patterns. A copyable plain-text triage note gives the URL defanged, both
results, Model A's top factors, the SHA-256 of both model files and the
caveats. Nothing is looked up anywhere.

Defanged input (`hxxps://example[.]com`) and URLs wrapped in `<>` or quotes are
restored before scoring, and the page says so. The **Several URLs** tab scores
up to 50 URLs, one per line: each model's result and risk score per line, and
the same table as CSV text with the URLs defanged. Nothing is written to disk.
API endpoints: `/score` (one URL), `/score_batch` (one URL per line) and
`/analyse` (the page's HTML report).

## What each band did on later data

Measured on 50,000 PhreshPhish URLs collected after every training URL.

| Model | Band | Phishing URLs in this band | Legitimate URLs in this band |
|---|---|---|---|
| Model A | Block | 33.5% | 0.07% |
| Model A | Review | 35.7% | 1.08% |
| Model A | Allow | 30.9% | 98.85% |
| Model B | Block | 55.6% | 0.13% |
| Model B | Review | 28.9% | 1.21% |
| Model B | Allow | 15.5% | 98.66% |

The Block threshold was set on validation data for a 0.1% false-positive rate.
The Review threshold was set on validation data for a 1% false-positive rate,
and it was added after the test results had been seen.

## What this tool does not do

- **It never fetches the URL.** It cannot see the page, follow a redirect or a
  link shortener, or check a domain's age or reputation.
- **Query strings and fragments are ignored.** Everything from the first `?` or
  `#` is removed before scoring.
- **A bare homepage carries little signal.**
- **A score is not a probability.**
- **Small, ordinary-looking edits change the answer.** Adding `www.`, switching
  to `https` and appending an ordinary path cut the share of phishing links
  blocked from 33.5% to 2.3% for Model A and from 55.6% to 3.6% for Model B.
- **On other collectors' data the false-positive rates are higher.** The Review
  band flagged 12.54% (Model A) and 18.98% (Model B) of presumed-benign URLs in
  one external dataset.

It is a fast first filter for a graduate course project, not a security product
and not a gate to rely on alone.

The app does not write a submitted link to disk or to its logs and does not keep
a history of checks in the browser. Hugging Face may keep its own request logs.

## Data

Trained on [PhreshPhish](https://huggingface.co/datasets/phreshphish/phreshphish)
(Dalton et al., CC BY 4.0, for anti-phishing research only).

The animated background is adapted from the GradientASCII component of
[React Bits](https://reactbits.dev) (MIT + Commons Clause, Copyright (c) 2026
David Haz). It is decorative and makes no network requests.

## Use of generative AI

The app, its models and this card were built with Claude Code (Anthropic). The
project repository's AI appendix summarizes the tools, how they were used and
the decisions.
