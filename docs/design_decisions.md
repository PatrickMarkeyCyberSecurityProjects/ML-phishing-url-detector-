# Design decisions

A dated record of what changed in the project's design, why, and what had been
seen of each dataset when each decision was made, so a reader can judge whether
any result was shaped by looking at test data.

- **Sections 1 to 5:** decided on 2026-11-21, before any model was trained. No
  model had been fitted, tuned or scored; `models/` was empty.
- **Section 6:** four additions decided on 2026-11-21 *after* the test results
  in `reports/results.md` had been seen (decisions 21 to 24). They are labelled
  post hoc wherever they appear.
- **Section 7:** errors found in the background research and how the project
  handles them.

Numbers come from `reports/phiusiil_template_rule.json`,
`reports/data_audit.json`, `reports/secondary_audit.json`,
`reports/secondary_audit_ariyadasa.json` and `notebooks/01_data.ipynb`.

## 1. The original design and what broke it

The project was first designed around PhiUSIIL (UCI id 967) as the training set,
with the PhreshPhish test split as the secondary dataset. The intended finding
was the gap between the two.

The first data audit showed that the gap would have been an artefact of
PhiUSIIL, not a measurement:

- Every legitimate PhiUSIIL URL is exactly `https://www.<host>`, with no path.
- A rule with no learning in it, "legitimate if the URL is exactly
  `https://www.<host>`, otherwise phishing", is right for 99.73% of PhiUSIIL's
  235,795 rows.
- In PhreshPhish the relationship is reversed: legitimate URLs are mostly pages
  with paths, and phishing URLs are the ones that are often bare homepages.

A model trained on PhiUSIIL learns the collection template. It cannot score a
real link, because real legitimate links have paths.

## 2. The design used

| Role | Dataset |
|---|---|
| Training | PhreshPhish train split (`url`, `label`, `date`, `sha256` only) |
| Validation | The most recent days of the train split, at least 15% of rows |
| Later-in-time test | A pinned 50,000-row sample of the PhreshPhish test split |
| Unseen secondary dataset | Wangchuk, "Phishing URL dataset" v1 (Mendeley, DOI 10.17632/3jddhy2f6s.1) |
| Second external check | Ariyadasa et al., "Phishing Websites Dataset" v1 (Mendeley, DOI 10.17632/n96ncsr5g4.1) |
| Cautionary baselines | The one-line template rule, and a LightGBM trained on PhiUSIIL |

The feature list is frozen at the 31 names written before any PhreshPhish row
was examined.

## 3. What had been seen of each dataset when the design changed

| Dataset | What had been looked at | What that means |
|---|---|---|
| PhiUSIIL | Fully profiled by class. No model fitted | Baseline only |
| PhreshPhish test split | The 50,000-row sample was profiled **by class**: URL shape shares, single-feature AUCs, sample rows. No model was fitted on it or scored against it | The move to PhreshPhish, the slash rule (decision 4) and the query rule (decision 8) were made after seeing class-level shape statistics of this sample. It was never used to fit, tune, calibrate or choose a threshold, but it is not a sample the designer knew nothing about |
| PhreshPhish train split | Not yet downloaded | |
| Wangchuk | Only the Mendeley description and file listing | Its audit was read before it was accepted. No model existed |
| Ariyadasa | Only the Mendeley description | Same as Wangchuk |

The sample's membership is pinned by the `sha256` id of each of its 50,000 rows
in `frozen/phreshphish_eval_sha256.txt`, so a rebuild selects exactly those rows.

## 4. Decisions made before any model was trained (2026-11-21)

### Data roles and rules

1. **Train on the PhreshPhish train split; keep PhiUSIIL only as a cautionary
   baseline.** Evidence: section 1.
2. **Validation is the most recent 15% of the train split by date**, cut on a day
   boundary. It is used only to fit the calibrator and choose thresholds.
   Tuning and early stopping use data from inside the fitting part.
3. **The feature list stays frozen at 31.**
4. **An empty path and `/` are the same URL.** PhiUSIIL never writes the final
   slash on a legitimate homepage and PhreshPhish always does, so the spelling
   alone separated classes. `normalize_url` writes an empty path as `/`. Chosen
   after the first look at the evaluation sample.
5. **The gates are direction-free.** A feature with AUC 0.03 separates the
   classes as well as one with 0.97, so each gate tests max(AUC, 1 - AUC).

### Gate 1 and the boundary day

6. **Drop the 681 train rows dated 2025-09-08.** The train split ends on the day
   the test split begins, although the paper says the test set "lies strictly in
   the future relative to the train set" (Section 4.1). After the drop (413
   legitimate, 268 phishing rows), training data ends on 2025-09-07.
7. **Remove validation URLs that loosely match a fitting URL.** 50 removed.

### Query strings

8. **The models are trained and scored with the query string and fragment
   removed** (`normalize_url(url, strip_query=True)`, in training, in every
   evaluation set and in the app).

   Evidence: in the train split as shipped, 0.006% of legitimate URLs have a
   query string, against 7.4% of phishing URLs; in the test split, 0.06% against
   24.0%. The PhreshPhish paper (arXiv 2507.10854v2, Appendix A.2, "Handling
   PII") states the cause:

   > "We focused our PII removal efforts on the benign class only as this is the
   > class that is most likely to contain sensitive information."
   >
   > "Query parameters in the URL are often used to pass sensitive information
   > such as user IDs, session tokens, and other PII to the server and are
   > therefore removed from the URL."

   The dataset card does not mention it. So "has a query string" marks phishing
   only because the collector removed queries from the other class, and a model
   trained on the URLs as shipped would flag ordinary links that carry a query.

   The coding tool (Claude Code) had recommended keeping the query and
   reporting it as a known artefact. The author overruled the coding tool's
   recommendation, for the reason above. That decision followed a review with a
   second AI assistant (the Claude app), which recommended removing the query.
   It was made after seeing the query shares of both the train split and the
   evaluation sample, by class.
9. **The feature list stays at 31 names.** The two query features (`query_len`,
   `n_query_params`) are constant at 0.
10. **Deduplication and train-against-test overlap removal use the
    query-stripped URL.** 3,331 duplicate train rows removed (3,277 phishing),
    one URL dropped for carrying both labels, and 39 train URLs removed for
    matching a test-split URL.
11. **The evaluation sample keeps its 50,000 pinned rows.** After query removal,
    852 rows share their URL with another row (236 URLs; 843 of the rows
    phishing), leaving 49,384 distinct URLs. No URL has both labels. The rows are
    kept and counted.
12. **One extra LightGBM is trained on full URLs** as a labelled comparison, to
    measure how much the artefact inflates scores. It is not deployed.
13. **All 31 features were scanned for the same one-sided pattern** (non-zero for
    under 0.5% of one class and over 5% of the other). On full URLs the scan
    flags exactly `query_len` and `n_query_params`; with the query removed it
    flags nothing.

### External datasets

14. **Wangchuk is the unseen secondary dataset.** It passed Gate 2: documented
    counts recovered (94,919 and 54,807); 3.1% of presumed-benign URLs are bare
    homepages (limit 50%); highest single shape-feature AUC 0.709 (limit 0.95).
    It is described as **a different-source test whose phishing URLs predate the
    training data, with a presumed-benign class of 547 registered domains**:
    - 99.96% of its phishing URLs are in the list published under DOI
      10.17632/vfszbj9b36.1 on 2024-04-02;
    - 38,493 rows are duplicates once the query is removed and 13,128 URLs also
      appear in PhreshPhish training or validation data, leaving 98,085 rows
      (59,162 presumed benign, 38,923 phishing);
    - 7.7% of its presumed-benign URLs are `robots.txt` files and 0.7% other
      non-page resources. Every row stays in the headline; a sensitivity result
      excludes them, using a list of file extensions fixed before any model was
      trained.
15. **Ariyadasa is a second external check.** Only `index.sql` is read. It passed
    the same gate: counts as documented (50,000 and 30,000); 14.0% bare
    homepages; highest AUC 0.716. Collected from 2020-12-01 to 2021-11-01, about
    three years before the training data. 309 URLs that also appear in the
    training data are removed, leaving 76,169 rows (48,477 presumed benign,
    27,692 phishing) across 27,399 presumed-benign registered domains.
16. **The legitimate class of both external datasets is called "presumed
    benign".** Nobody verified those pages.

### Collection months and what a score means

17. **No legitimate rows exist for November 2024 to April 2025**, which hold
    63,745 phishing rows. The fitting data is 47.6% phishing and validation 25.0%.
    The date is not a feature, so this does not leak the label, but the
    calibrator is fitted on validation, whose class mix is an accident of the
    collection schedule. **Calibrated scores are not real-world probabilities.**
    Thresholds are set on rates within each class, which do not depend on the
    class mix.

### Evaluation

18. **Three credibility splits for every metric:** bare homepage versus a path;
    registered domain seen versus unseen in training; original URL had a query
    string or not.
19. **For the external datasets, the false-positive rate is reported per URL and
    averaged per registered domain**, with intervals resampled by domain.

### Training hardware

20. **The CNN is trained on the full fitting set on a GPU** (an NVIDIA RTX 4080
    SUPER, with the CUDA build of torch 2.15.0). The deployed app runs the ONNX
    export on CPU.

## 5. What this design can and cannot claim

- **Later in time:** only the PhreshPhish evaluation sample, which comes from the
  same collector as the training data.
- **Different source:** Wangchuk and Ariyadasa. Both are *older* than the
  training data, so they test transfer to another collector, not to the future.
- **Not claimed:** performance on live traffic. Every class mix here is set by
  how the data was collected.

## 6. Additions made after the test results were seen (2026-11-21, post hoc)

Decisions 21 to 23 were decided after the main evaluation had been read: the
headline table, the credibility splits, the feature-contribution ranking
(`path_len` first) and the random-edit result. They are computed in
`notebooks/05_post_hoc.ipynb`, saved in `reports/metrics.json` under the key
`post_hoc`, and written in `reports/results.md` under their own heading.
Decision 24 was decided later the same day, after a review of the finished
project; it is computed in `notebooks/03_cnn.ipynb` and saved in its own file.
None is a pre-specified result. None changes the deployed models, the
calibrators, the block and warn thresholds or the main results table.

21. **Report the frozen warn threshold's false-positive rate on validation.** The
    warn threshold was pre-specified as "catch at least 95% of validation
    phishing". The main evaluation reported how many legitimate URLs it flags on
    the test sets but not on validation, where it was set. The validation figure
    shows whether the 95%-detection point was already unusable before any test.
22. **Add a "review" operating point: the threshold giving a 1% false-positive
    rate on validation.**
    - Rule: the lowest raw-score threshold at which at most 1% of legitimate
      validation URLs are flagged; the block rule with 1% in place of 0.1%,
      computed by the same function on the same validation scores.
    - The 1% figure was named after the test results had been seen. One value
      was named; no other value is tried on any test set. Each test set is
      scored with it once.
    - It is saved in its own file, `models/review_operating.json`.
    - The original warn point is described as what it was: the pre-specified
      target, which proved unreachable at a usable false-positive rate for a
      URL-only model.
    - The app shows three bands per model: Block (the frozen 0.1% threshold),
      Review (this threshold) and Allow.
23. **A targeted robustness test.** The random-edit test did not lower
    detection, which says nothing about edits chosen to look legitimate. The
    threat model named such edits before any model existed
    (`docs/problem_and_threat_model.md`, section 4.3). **The edits below were
    written here before the test was run**, and it was run once.

    Input: every phishing row of the PhreshPhish evaluation sample, as its
    query-stripped URL.

    | Edit | What it does |
    |---|---|
    | `www` | Put `www.` in front of the host name, unless it already begins with `www.` or is an IP address |
    | `https` | Set the scheme to `https` |
    | `path` | Append a benign-looking path to the existing path, after removing a trailing `/` |
    | `all` | The three edits together |

    The appended path is one of these five, drawn for each URL with seed 42; the
    `all` edit uses the same draw: `/about-us`, `/en/support/contact`,
    `/blog/2025/03/company-news`, `/products/category/item-details.html`,
    `/news/articles/annual-report-2024.html`.

    Reported: for each model, the share of these phishing URLs at or above the
    block and review thresholds, before and after each edit, with paired 95%
    bootstrap intervals (1,000 resamples, seed 42); how many URLs each edit
    changed; and the result for each path.

    Limits, stated in advance: the edits were written from the threat model, not
    found by querying the models, so the result is a lower bound on what an
    attacker who probes the scorer could achieve. Whether a phishing page would
    still load at the edited address is not tested.

24. **A CNN tuning check (post hoc; decided 2026-11-21 after the overnight
    review, written here before it was run).** LightGBM's settings were searched
    with Optuna; the CNN's were fixed in advance and never searched. This check
    asks whether the fixed settings were a poor choice. It is decided after the
    test results had been seen, so it is labelled post hoc wherever it appears.

    - **Grid:** the shipped architecture (`src.models.CNN_CONFIG`) with the
      learning rate at 0.0003, 0.001 and 0.003, and dropout at 0.3 and 0.5:
      six runs. The shipped settings (0.001, 0.3) are one of them. Dropout
      changes both dropout layers. Everything else is as shipped: AdamW with
      weight decay 0.01, batch 512, class weights, at most 30 epochs, early
      stopping after 3 epochs without improvement on the same early-stopping
      split (the most recent 5% of the fitting data), seed 42, the same GPU and
      deterministic settings.
    - **Data:** the same fitting data and early-stopping split as notebook 03.
    - **Measured:** validation PR-AUC only, from the PyTorch model, on the same
      validation set (the most recent weeks of the train split). No test set is
      scored with these models, nothing is calibrated and no threshold is set.
    - **"Among the best", fixed now:** the shipped settings count as among the
      best if their validation PR-AUC is within 0.005 of the best of the six.
    - **Whatever the result, the deployed CNN is kept.** Choosing a different
      model now, after the test results have been seen, would let those results
      influence the model and bias every test number. None of the six models is
      saved as a deployable model, exported or used by the app.
    - **Reported** in `notebooks/03_cnn.ipynb` (a new last section), saved in
      a new file, reports/cnn_tuning_check.json, and summarized in one sentence in
      `reports/results.md`.

## 7. Corrections to the background research

The project began from an AI-assisted research report. A review before the
build, and the data itself, found the errors below.

| What the research said | What is true | What the project does |
|---|---|---|
| Models score about 99% in-dataset but near chance across datasets (Cantone et al., 2024) | Cantone et al. is a network-intrusion study | Cited only as an analogy |
| An expected 20-40 point cross-dataset drop, credited to Hannousse & Yahiouche | The citation is a different paper | Not attributed; treated as a weak expectation |
| "One replication saw F1 fall from 0.997 to 0.641" | The source is an individual's GitHub repository | Quoted with that caveat |
| BEC losses of $3.05B as the headline | Much BEC involves no URL; the IC3 phishing/spoofing figure is $215.8M | BEC is presented as the downstream crime, not money a URL scorer saves |
| "Phishing is the top breach vector (IBM 2025)" | Verizon's DBIR 2026 ranks exploitation first | Stated as IBM's finding, with Verizon's ranking alongside |
| Industry figures quoted as fact | Most come from secondary sources | Each names its source. On 2026-11-21 they were checked against the originals where these could be opened; the wording of four was corrected |
| A 30% breach-prevention rate | It has no source | Not used; the business case gives no breach or dollar figure |
| The app shows SHAP values | The deployed image must not need `shap` | The app uses LightGBM's own `pred_contrib` |
| PhiUSIIL's legitimate class is "homepage-biased" | Stronger: every legitimate URL is `https://www.<host>` | Section 1 |
| PhreshPhish benign URLs are "real visited pages" | Webroot browsing telemetry plus Google results for targeted brands; query strings removed for privacy | Decision 8 |
| PhreshPhish is "~44% phishing" | The test split is 45.7% phishing | Reported as measured |
| Ariyadasa has 83,275 instances | `index.sql` holds 80,000 rows | Decision 15 |
| Exact registered domains need a public-suffix list | That adds a dependency that downloads data | Standard library only; suffixes like `co.uk` are approximated |
