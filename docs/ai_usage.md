# AI Appendix

- **Claude, Fable 5.2 (Claude app):** background research, planning, and review
  of results.
- **ChatGPT, GPT-6.2 Astra with web search:** a survey of public phishing URL
  datasets.
- **Claude Code, Fable 5.2 (VS Code), the tool shown in class:** the code,
  tests, notebooks and project documents, used as generated after my review.

## How I used it

My first goal was to identify a real cybersecurity problem and find out which
public datasets could support it. AI research tools sped up that survey, and I
used two of them so I could compare their findings before choosing phishing URL
detection and the datasets.

AI was most useful for turning my ideas and research into a workable prototype
quickly: refining the plan, writing and testing code, and checking the data for
flaws I would have been slow to find by hand.

Early on I saw that one long conversation would use up the context window and my
usage allowance before the deadline. I restructured the work around context
engineering: a short project rules file read at the start of every session, one
stage per conversation, findings saved in the repository, and a review pause
after each stage.

AI also turned the training and evaluation output into summary tables and
charts, which let me see the results clearly and decide how to finalize the
project.

## Decisions and verification

AI tools proposed several of the design changes. Replacing the original
training dataset was recommended by the Claude app after Claude Code found the
flaw. On query strings the two tools disagreed: Claude Code recommended keeping
them and the Claude app recommended removing them, and I chose removal. I
reviewed the evidence and approved each one. I set dataset quality as the top
priority, commissioned the separate dataset survey, and chose to train on my
own GPU.

AI output was checked and sometimes wrong. The research report contained
citation errors that I corrected, and two claims by the coding tool were fixed
after testing.
