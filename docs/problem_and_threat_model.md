# Problem Identification and Threat Model

## 1. The problem

A phishing link takes the person who clicks it to a page that harvests
credentials or delivers malware. The defender has to decide whether a link is
dangerous before the click, often for a URL nobody has seen before. This project
builds a scorer that makes that decision from the URL string alone, in
milliseconds, without fetching the page.

The problem is large and well measured:

- The FBI's Internet Crime Complaint Center (IC3) logged 191,561
  phishing/spoofing complaints in 2025, and reported phishing/spoofing losses
  tripled to $215.8M, from $70.0M in 2024 (FBI IC3 2025 Internet Crime Report,
  pp. 26-27 [17]; also reported by SpyCloud [1] and Brandefense [2]).
- The human element was involved in 62% of breaches (Verizon DBIR 2026, as
  reported by Help Net Security [3]; the report itself is behind a form).
- Phishing was the most common initial vector in IBM's 2025 study, at 16% of
  breaches (IBM Cost of a Data Breach 2025 [18]; also Northdoor [4]). Verizon's 2026
  report ranks vulnerability exploitation first at 31% [5], so phishing is one
  of the leading entry paths, not the only one.
- Attackers have automated the writing. The volume of AI-assisted text in
  malicious email doubled (DBIR 2026, as reported by Mimecast [6]), and IC3
  recorded 22,364 complaints tagged as AI-related, with $893M in losses (IC3
  2025 report, p. 39 [17]; also dmarcian [7]).

## 2. Impact if left unsolved

A phished credential is rarely the end of the incident. It is the entry point
for account takeover, business email compromise (BEC) and ransomware.

- BEC cost the victims who complained to IC3 $3,046,598,558 across 24,768
  complaints in 2025 (FBI IC3 2025 report, pp. 26-27 [17]; also DeepStrike [8]).
  IC3 takes complaints from more than 200 countries, so this is not a US-only
  figure. Not all BEC starts with a link, so this is the scale of
  the downstream crime that stolen credentials feed. It is not the amount a URL
  scorer would save.
- The average breach cost $4.99M globally (IBM 2026 [9]) and $10.22M in the US
  (IBM 2025 [18]; also DBTA [10]).

## 3. Current solutions and where they fall short

| Control | How it works | Why it falls short |
|---|---|---|
| Blocklists and reputation feeds | Block URLs and domains already reported as malicious | Reactive. A new URL is unlisted until someone reports it, and domains and hosting are close to free and disposable, so attackers rotate faster than lists update. |
| Sandboxes (detonation) | Fetch and render the page in an isolated browser and observe it | Takes seconds to minutes per URL, too slow to sit inline on every link. Cloaking defeats it: the page shows benign content to scanners and the phish to the victim. |
| User awareness training | Teach people to spot and report lures | Improvement plateaus, and the cues taught (poor grammar, odd formatting) are the ones AI-written lures remove. Social engineering on mobile succeeds 40% more often than email phishing (DBIR 2026 [5]; reported elsewhere as click rates in phishing tests), and on a phone the URL is barely visible. |
| Commercial ML-based email and URL security (Microsoft Defender for Office 365, Proofpoint, Google, Abnormal AI) | Vendor-trained models score messages and links inline, drawing on telemetry from the vendor's whole customer base | Closed. The models and their evaluation cannot be inspected or reproduced, so a buyer cannot tell how they perform on unfamiliar data. According to this project's background research, no vendor publishes cross-dataset results. |

The gap is a zero-hour, pre-click decision. A model that scores the URL string
can run on every link in under 10 ms and needs no prior report (measured later:
1.0 ms for both models inside the app's container, `reports/latency.json`). PhiUSIIL's
own authors report 99.79% accuracy on it with a pre-trained model, and 99.24%
with one trained only incrementally [11]. Scores like that make the problem
look solved. Section 4.3 explains why it is not.

## 4. Threat model

### 4.1 Threat actor

Financially motivated phishing crews and their suppliers, the
phishing-as-a-service kit operators.

- **Capabilities:** LLM-written lures, bulk domain registration, cloaking,
  hosting on legitimate platforms, and the ability to query any public detector
  repeatedly to see what passes.
- **Resources:** low cost and high scale. A campaign needs a kit subscription
  and a batch of cheap domains, so a blocked URL costs the attacker almost
  nothing to replace.
- **Motivation:** money, through credential resale, BEC payouts and the sale of
  initial access to ransomware operators.

Nation-state actors running tailored, low-volume spear phishing are out of
scope. They can afford infrastructure this system cannot distinguish from
legitimate sites.

### 4.2 Targets

Enterprise and SaaS credentials, MFA session tokens, banking logins and
payment-approval workflows. The people hit first are employees and finance
teams. Consumers and the brands being impersonated are hit second.

### 4.3 Evasion against this system

The system has two models: Model A, LightGBM on 31 lexical features, and
Model B, a character-level CNN. Both see only the URL string, with its query
string and fragment removed, which defines the attack surface.

| Evasion method | Why it works here | How this project tests it |
|---|---|---|
| Compromised or abused legitimate hosting | The host is a real, benign domain. Lexical features carry little or no signal, and neither model can see page content. This is the system's largest structural blind spot. | **Tested indirectly.** Every result is split by whether the registered domain was seen in training. On later data Model A blocks 45.4% of phishing on seen domains and 24.0% on unseen ones (Model B: 66.1% and 47.2%). No test isolates phishing hosted on a compromised legitimate site |
| Homoglyph and internationalized (IDN, punycode) domains | A lookalike character keeps the URL visually identical while changing the string the models read | **Partly tested.** The random-edit test includes ASCII look-alike swaps (`o` to `0`, `l` to `1`) among its edits; after eight random edits blocking did not fall (Model A 34.0% to 37.6%, Model B 55.9% to 56.3%). Unicode homoglyphs and punycode hosts were not tested |
| Short, brand-like URLs | Length, digit and special-character features look benign; little is left to score | **Tested by a split, not by an attack.** Every result is split by bare homepage versus a URL with a path. On later data the review point flags 7.22% (Model A) and 6.86% (Model B) of legitimate bare homepages, against 0.54% and 0.79% of legitimate URLs with a path. Shortening phishing URLs to see whether they escape was not tested |
| Mimicking benign feature patterns | An attacker who probes the scorer can shape a URL (HTTPS, a `www` prefix, a common TLD, a long path) to match what the model learned as legitimate | **Tested, after the test results were seen (post hoc).** A `www.` prefix, `https` and a benign-looking path were applied to all 22,851 phishing URLs of the later test; the edits were written down before the run. Together they cut blocking from 33.5% to 2.3% (Model A) and from 55.6% to 3.6% (Model B). Changing the TLD was not tested, and the edits were not found by probing the models, so an attacker could do better. SHAP ranks `path_len` first among Model A's features |
| Hiding the payload in the query string | The models ignore everything after `?` or `#`, so a phishing URL that differs from a benign one only there scores the same | **Not detectable by design**; stated as a limitation in the results and in the app. Results are split by whether the original URL had a query string |
| Redirect chains and URL shorteners | The scored string is the shortener or redirector. The destination is never seen, because the system does not fetch | **Not tested.** Stated as a limitation in the results and in the app |
| Concept drift | Kits, hosting habits and TLD choices change, so a model trained on older URLs decays | **Tested once, not tracked over time.** The PhreshPhish evaluation sample (2025-09-08 to 2025-12-16) was collected after every training row. Model A's ROC-AUC was 0.890 on the six validation weeks after training and 0.957 on the evaluation sample, so quality swings between periods. The two external datasets are older than the training data and test a different source instead. No retraining schedule was built or tested |
| Feed poisoning | Training labels come from public feeds an attacker can submit to | **Not tested.** Mitigated only by training on a fixed, published dataset |

The figures in the last column are read from `reports/metrics.json` and are
also in `reports/results.md`. The targeted test and the review point were added
after the test results were seen; `docs/design_decisions.md`, section 6, records
them.

A second risk comes from the data, not the attacker. PhiUSIIL's legitimate
class is made of homepages: this project measured that every legitimate URL in
it is exactly `https://www.<host>` (`reports/phiusiil_template_rule.json`, written by
`notebooks/01_data.ipynb`, section 1), and
one replication reports the same [13]. A model can score near-perfectly by
learning how the dataset was collected, then fail elsewhere.
Reported cross-dataset results include an XGBoost model with an F1 of 69.37%
on PhiUSIIL when tested across datasets [14], and, in one replication, an F1 of
0.997 in-dataset on PhiUSIIL against 0.641 when a model trained on another
dataset is tested on PhiUSIIL [13]. The same pattern is
documented in network intrusion detection (Cantone et al., 2024 [15]).

Because of that template, a one-line rule with no learning is right for
99.73% of the dataset (`reports/phiusiil_template_rule.json`). The project therefore does not train
on PhiUSIIL. It trains on the PhreshPhish train split, whose legitimate URLs
are pages people visited, and reports three tests: a later sample from
PhreshPhish, and two datasets from other collectors (Wangchuk, and Ariyadasa
et al.). A model trained on PhiUSIIL is kept only as a cautionary baseline.
The training data has collection artefacts of its own, most importantly that
its collector removed query strings from legitimate URLs only; these are
recorded in `docs/design_decisions.md`.

### 4.4 Operational context

**Where it sits.** The scorer is one layer in the email gateway, secure web
proxy or browser, placed after the blocklist lookup (a cheap check for known
bad URLs) and before the sandbox (an expensive check for uncertain ones). It
blocks above a high threshold, sends a middle band to review and allows the
rest. It complements blocklists and sandboxes and does not replace them.

As built, the middle band is "Review": a threshold set on validation for a 1%
false-positive rate, added after the test results were seen. The band specified
here originally, a warning that catches 95% of phishing, flagged 65.9%
(Model A) and 71.5% (Model B) of legitimate URLs on validation and is not used
in the app (`docs/design_decisions.md`, decision 22).

**In an analyst's workflow.** When a user reports a link, or a Review result
lands in the queue, the analyst pastes the link into the app. The app shows
both results, Model A's feature contributions, and the URL split into its
parts. It also gives a copyable triage note for the ticket, with the
URL defanged, both results, the model versions and the caveats. The note is
decision support, not a verdict. It supports the decision to escalate to a
sandbox, a reputation lookup or the user, and it does not replace those checks.
The app performs no lookups of its own, so no threat-intelligence result
appears in it that was not actually obtained.

**False negative.** A phishing link reaches the user. If they submit
credentials, the organization faces account takeover and possibly a breach
averaging $4.99M (IBM 2026 [9]). Other layers (MFA, sandboxing, user reporting)
still stand between a miss and a breach, so one miss is serious but not
automatically catastrophic.

**False positive.** A legitimate business link is blocked. Each one costs a
help-desk ticket and a delayed task, and repeated ones teach users to ignore
warnings and push administrators to switch the control off. Legitimate URLs
also vastly outnumber phishing ones, at realistic phishing rates of 0.05% to 5%
(the base rates of the PhreshPhish benchmarks [19]; also [16]), so false positives dominate at any loose threshold. As an illustration,
a scorer with 95% detection and a 0.1% false-positive rate is right in about
91% of its alerts when 1% of URLs are phishing, and in about 49% when 0.1% are.

For that reason the false-positive rate is the gating metric. The target
operating point is a false-positive rate of 0.1% or lower, and the project
reports detection rate at that point alongside F1 and AUC.

## 5. What this system does not do

It does not read page content, follow redirects, check domain age or
reputation, or inspect the email carrying the link. It ignores the query string
and fragment of a URL. It cannot detect phishing on a compromised legitimate
site from the URL alone, and it does not address BEC that contains no link. Its
calibrated scores are not real-world probabilities, because the class mix of
its training data was set by how the data was collected.

## Sources

Figures were first taken from an AI-assisted research report, through the
secondary sources below. On 2026-11-21 each was checked against the original
where it could be opened. The IC3 and IBM
figures were confirmed in the original reports [17][18][9]; the Verizon 31% and
40% figures in Verizon's own release [5]. The Verizon 62% and "AI-assisted text
doubled" figures could not be checked against the DBIR itself, which is behind a
form, and rest on the secondary sources named. The PhiUSIIL accuracy figures [11]
were checked against the paper by the author on 2026-11-22. An upper figure of
99.97%, which the research had given, is not in the paper and has been removed.

1. SpyCloud, "FBI IC3 Report 2025: $20.9B in Cybercrime Losses." https://spycloud.com/blog/fbi-internet-crime-report-2025/
2. Brandefense, "FBI IC3 2025 Report: The Costliest Cybercrimes." https://brandefense.io/security-news/fbi-ic3-2025-costliest-cybercrimes/
3. Help Net Security, "Lessons for organizations from the Verizon 2026 Data Breach Investigations Report." https://www.helpnetsecurity.com/2026/05/25/lessons-from-verizon-dbir-2026-findings/
4. Northdoor, "Cost of a Data Breach 2025: IBM Global Report." https://www.northdoor.co.uk/insight/news/cost-of-a-data-breach-report-2025/
5. Verizon, "Breach entry point, 2026 DBIR finds." https://www.verizon.com/about/news/breach-industry-wide-dbir-finds
6. Mimecast, "Findings from the 2026 Verizon DBIR that should shape your defenses." https://www.mimecast.com/blog/2026-verizon-dbir-findings/
7. dmarcian, "2025 Internet Crime Report: Nearly $21 Billion Lost." https://dmarcian.com/fbi-internet-crime-report-2025/
8. DeepStrike, "Business Email Compromise Statistics 2026." https://deepstrike.io/blog/business-email-compromise-statistics
9. IBM, "Cost of a Data Breach Report 2026." https://www.ibm.com/reports/data-breach
10. Database Trends and Applications, "IBM's 'Cost of a Data Breach Report' Signals Need for Greater Security and Governance for AI." https://www.dbta.com/Editorial/News-Flashes/IBMs-%E2%80%98Cost-of-a-Data-Breach-Report-Signals-Need-for-Greater-Security-and-Governance-for-AI-170893.aspx
11. Prasad & Chandra, "PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning," Computers & Security, 2024. https://www.sciencedirect.com/science/article/abs/pii/S0167404823004558
12. Not cited. Wei, Nakayama & Sekiya, "Enhancing Generalization in Phishing URL Detection via a Fine-Tuned BERT-Based Multimodal Approach," IEEE Access 13, 2025, doi:10.1109/ACCESS.2025.3591843, was cited for PhiUSIIL's homepage bias until 2026-11-21. Its abstract does not mention it and the full text could not be opened, so it was removed as support. The number is kept so the other references keep theirs.
13. Shaurya311, "phishing-and-malware-detection" (GitHub repository; an individual replication, not peer reviewed). https://github.com/Shaurya311/phishing-and-malware-detection
14. Misiek & Hyla, "XGBoost-Based URL Phishing Detection Method With Cross-Dataset Validation," IEEE Access 14, 2026, doi:10.1109/ACCESS.2026.3672690 (the 69.37% F1 was read in a published review of the paper, not in the paper). https://www.researchgate.net/publication/401915874_XGBoost-based_URL_phishing_detection_method_with_cross-dataset_validation
15. Cantone, Marrocco & Bria, "On the Cross-Dataset Generalization of Machine Learning for Network Intrusion Detection," 2024. arXiv 2402.10974. A network-intrusion study, cited as an analogy.
16. Zhang, "Hunting for Good Phish Data: Why the Dataset You Use Matters" (base rates used in the PhreshPhish benchmarks; the page refused automated access on 2026-11-21). https://zhanghaolin66.medium.com/hunting-for-good-phish-data-why-the-dataset-you-use-matters-ebe3caf2ea6c
17. FBI Internet Crime Complaint Center, "2025 Internet Crime Report." https://www.ic3.gov/AnnualReport/Reports/2025_IC3Report.pdf
18. IBM, "Cost of a Data Breach Report 2025" (the full report, read in a copy hosted by a law firm; IBM's newsroom release of 2025-07-30 gives the same US figure). https://www.bakerdonelson.com/webfiles/Publications/20250822_Cost-of-a-Data-Breach-Report-2025.pdf
19. Dalton et al., "PhreshPhish: A Real-World, High-Quality, Large-Scale Phishing Website Dataset and Benchmark," arXiv 2507.10854: "benchmark datasets with five different base rates varying from 0.05% to 5%." https://arxiv.org/abs/2507.10854
