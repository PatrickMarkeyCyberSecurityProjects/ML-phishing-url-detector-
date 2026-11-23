# Deployment

How the app is packaged, how it was tested locally, and the exact steps to put
it on a Hugging Face Docker Space. The local build and test were run on
2026-11-21 and their results are below. **The Space was published on
2026-11-23** from the owner's Hugging Face account: https://x39393902-phishing-url-scorer.hf.space (section 7a).

## 1. What is deployed

A Gradio app (`app/app.py`) in a Docker container on port 7860. It scores a URL
string with the two frozen models and never fetches the URL.

The Space needs its `Dockerfile` and `README.md` at the root of the Space
repository, with the app's files beside them. `app/build_space.py` copies
exactly these 14 files into `build/space/` (9.2 MB in total):

| File in the bundle | From | Why |
|---|---|---|
| `README.md` | `app/README.md` | The Space card, with the YAML header |
| `Dockerfile` | `app/Dockerfile` | The image |
| `requirements.txt` | `app/requirements.txt` | Every package pinned: the four the app imports and their dependencies |
| `app.py` | `app/app.py` | The app |
| `background.js` | `app/background.js` | The page's decorative animated background, put in the page head by `app.py` (ported from React Bits' GradientASCII, MIT + Commons Clause; no network use) |
| `src/__init__.py`, `src/features.py` | `src/` | The one feature module, as used in training |
| `models/lgbm.txt` | `models/` | Model A (8.0 MB, text) |
| `models/cnn.onnx` | `models/` | Model B (0.36 MB, binary) |
| `models/lgbm_operating.json`, `models/cnn_operating.json` | `models/` | Frozen block thresholds and calibrators |
| `models/review_operating.json` | `models/` | The review thresholds (added after the test results were seen) |
| `models/hashes.json` | `models/` | Recorded hashes of the files above |
| `reports/metrics.json` | `reports/` | The measured rates the app shows under "How reliable is this" |

The script refuses to build if a model file, an operating file or
`src/features.py` no longer matches `models/hashes.json`.

### Configuration

| Setting | Value | Where |
|---|---|---|
| Base image | `python:3.11-slim` | `Dockerfile` |
| System package | `libgomp1` (LightGBM needs it) | `Dockerfile` |
| User | `user`, uid 1000, not root | `Dockerfile` |
| Port | 7860 | `EXPOSE`, `GRADIO_SERVER_PORT`, `app_port` in the card |
| Python packages the app imports | `numpy==2.4.6`, `lightgbm==4.7.0`, `onnxruntime==1.32.0`, `gradio==6.32.0` | `requirements.txt` |
| Not installed | torch, scikit-learn, shap | checked in the built image |
| Telemetry | off (`GRADIO_ANALYTICS_ENABLED=False`) | `Dockerfile` |
| Secrets, tokens, environment variables to set | none | |
| Outbound network at run time | none needed | the app analyses strings only |
| Maximum input | 2,000 characters | `app.py` |

Every package in the image is pinned, dependencies included, to the versions
in the image that was tested, so a rebuild installs the same set. Among the
dependencies are pandas (Gradio needs it) and scipy (LightGBM needs it); the app
imports neither. The base image is the floating `python:3.11-slim` tag, so a
later rebuild may get a newer Python 3.11 patch release.

## 2. Build and run locally

Run from the repository root. Docker must be running.

```powershell
python app/build_space.py
docker build -t phishing-url-scorer build/space
docker run --rm -d -p 7860:7860 --name phishing-url-scorer phishing-url-scorer
```

Open http://localhost:7860. A link of the form
`http://localhost:7860/?url=<encoded URL>` fills the box and scores it.

```powershell
docker logs phishing-url-scorer      # "Running on local URL: http://0.0.0.0:7860"
docker stop phishing-url-scorer
```

Without Docker: `python app/app.py` from the repository root, in the project's
virtual environment.

## 3. What the local test showed (2026-11-21, repeated 2026-11-23)

First run on 2026-11-21. Repeated on 2026-11-23, after the page
was rewritten as a neutral report with URL details and a triage note, and
measured again the same day after the animated background was added and the
example buttons removed; the scores did not change, and the figures below are
from that last run. Measured once more later on 2026-11-23, after defanged input
was restored before scoring and the **Several URLs** tab (`/score_batch`) was
added; the scores of every well-formed URL are unchanged (the container tests
compare them with the notebooks' saved scores, for `/score` and `/score_batch`).

- **Build:** succeeded. `docker images` reports the image at 997 MB; 608 MB of
  that is the layer that installs the four packages and their dependencies.
- **Inside the container:** `uid=1000(user)`, Python 3.11.18, torch,
  scikit-learn and shap absent, and `pip freeze` equal to `requirements.txt`
  as a set. On 2026-11-23 one package, `packaging==26.3`, was found in the image
  but not in the file: it ships with the `python:3.11-slim` base image. It is now
  pinned in `requirements.txt` as well.
- **Start-up:** the page answered about 3 seconds after `docker run`.
- **100 invented URLs** sent one at a time by `app/measure_latency.py`
  (results in `reports/latency.json`):

| Measure | Median | 95th percentile |
|---|---|---|
| Full request, plain HTTP over one kept-alive connection | 76.1 ms | 79.8 ms |
| Full request, `gradio_client` Python library | 373.5 ms | 383.9 ms |
| Inside the app, everything | 28.3 ms | 30.4 ms |
| Inside the app, scoring with both models | 1.0 ms | 1.5 ms |
| Inside the app, Model A's feature breakdown (`pred_contrib`) | 27.1 ms | 29.3 ms |

  Read these with care. The client and the container were on the same machine
  (AMD Ryzen 7 7800X3D, Docker Desktop on Windows 11), requests were sent one at
  a time, and nothing here measures a hosted Space. The feature breakdown, not
  the models, is nearly all of the time inside the app. The `gradio_client`
  library adds about 300 ms per call on this machine over plain HTTP.
- **Tests:** `pytest` passes, including `tests/test_app.py`, which requires the
  app's scores to equal the scores the notebooks saved. With the container
  running, `PHISHING_APP_URL=http://localhost:7860 pytest tests/test_app.py`
  also checks the container itself over HTTP; it passed.

To repeat the measurement:

```powershell
python app/measure_latency.py http://localhost:7860 "local Docker container"
```

## 4. Before creating the Space: the account plan

The Hugging Face documentation read on 2026-11-21
(https://huggingface.co/docs/hub/spaces-overview) says:

> Gradio and Docker Spaces run on compute and require a paid plan to create: PRO
> for personal accounts, Team or Enterprise for organizations. Free personal
> accounts in good standing can still host up to 2 Gradio Spaces running on
> ZeroGPU.

So a Docker Space needs a PRO account (the pricing page lists it at $9 a
month). The CPU Basic hardware itself (2 vCPU, 16 GB) has no hourly cost. If the
account is on the free plan, see section 8.

## 5. Create the Space

1. Go to https://huggingface.co/new-space.
2. Owner: your account. Space name: for example `phishing-url-scorer`.
3. SDK: **Docker**. Template: blank.
4. Hardware: **CPU Basic**.
5. Visibility: public (the grader needs to open it).
6. Create the Space.

## 6. Push the bundle

Authentication is yours to set up: run `hf auth login` (install the CLI with
`pip install hf`) or add an SSH key at https://huggingface.co/settings/keys.
No token is stored in this repository and none should be.

### Option A: git

Replace `<user>` and `<space>`. The clone address is shown on the Space page
under "Clone repository".

```powershell
git clone https://huggingface.co/spaces/<user>/<space> ..\space-clone
python app/build_space.py ..\space-clone
cd ..\space-clone
git check-attr filter -- models/cnn.onnx models/lgbm.txt
git add -A
git status
git commit -m "Phishing URL scorer: app, models and measured rates"
git push
```

`build_space.py` keeps the clone's `.git` folder and `.gitattributes` and
replaces everything else, including the placeholder `README.md`.

About the two model files. `models/lgbm.txt` is 8.0 MB of text and
`models/cnn.onnx` is 0.36 MB of binary. The documentation says files over
10 MB should be tracked with `git-xet`; both are under that. The pages read did
not say whether a small binary file is accepted in a plain push. The
`git check-attr` line shows whether the Space's own `.gitattributes` already
routes `*.onnx` through LFS or Xet. If the push is rejected because of
`cnn.onnx`, run `git xet install` (or `git lfs install`), then
`git lfs track "*.onnx"`, `git add .gitattributes models/cnn.onnx`, commit and
push again.

### Option B: the `hf` command

```powershell
python app/build_space.py
hf upload <user>/<space> build/space . --repo-type space
```

The documentation pages read show `hf upload` with `--repo-type dataset`. The
`space` value follows the same pattern but was not shown on those pages.

## 7. Check the Space

1. Open the Space page. The "Building" log should show the same steps as the
   local build and end with `Running on local URL: http://0.0.0.0:7860`.
2. Before analysis the page shows only the URL box. Paste the invented
   URL `http://secure-login.account-verify.example.net/signin/`; it should show "Split result
   (no combined verdict)" (Model A: Review, Model B: Block).
3. Paste an empty box and a string longer than 2,000 characters. Each should
   give a one-line message.
4. Optionally measure the hosted app, keeping the local measurement:

```powershell
python app/measure_latency.py https://<user>-<space>.hf.space "Hugging Face Space, CPU Basic" reports/latency_space.json
```

A free-hardware Space goes to sleep when unused and takes a while to wake.
Open it a few minutes before the briefing. The screenshots in
`docs/screenshots/` are the fallback if it does not wake.

To update the Space later, repeat section 6. Each push rebuilds and restarts it.

## 7a. What the published Space showed (2026-11-23)

- Space `x39393902/phishing-url-scorer`, Docker SDK, CPU Basic, visibility
  *Protected*: the app is open to anyone at https://x39393902-phishing-url-scorer.hf.space, while the
  repository page needs a login (it answers 401 without one).
- Pushed with Option B (`hf upload ... --repo-type space`): 14 files, one commit.
  `cnn.onnx` was accepted.
- The build finished and the Space reached `RUNNING` about one minute after the upload.
- `PHISHING_APP_URL=https://x39393902-phishing-url-scorer.hf.space pytest tests/test_app.py`:
  107 passed, none skipped. The hosted app reproduces the notebooks' saved scores.
- Latency on Hugging Face hardware was not measured.

## 8. If the account cannot create a Docker Space

These are options, not steps already taken.

- **Upgrade to PRO** for the month of the briefing. Nothing in the project
  changes.
- **Demo from the local container** (section 2). The assignment asks for
  "Hugging Face Spaces (or similar)" and a packaged model; the Docker image
  meets the packaging requirement, but there would be no public URL.
- **A Gradio-SDK Space on the free ZeroGPU allowance.** The bundle's `app.py`
  and `requirements.txt` would work with a different card header (`sdk: gradio`,
  `sdk_version: 6.32.0`, `python_version: "3.11"`, `app_file: app.py`) and
  without the Dockerfile. This was not tested, and it would change a fixed
  design decision (a Docker container), so it would need to be decided first.

## 9. What has not been verified

- Whether `cnn.onnx` is accepted in a plain git push (the push used `hf upload`).
- Latency on Hugging Face hardware, and behaviour under more than one request
  at a time.
- A rebuild at a later date may use a newer `python:3.11-slim` base image.
