# LLM Judge Bias Report - Phase B

**Date:** 2026-08-27
**Judge:** local deterministic fallback; no `OPENAI_API_KEY` configured

## Results

The judge evaluated the 10 provided human-label questions by comparing each recorded model answer (A) with that question's ground truth (B), then repeated the comparison after swapping positions.

| Measure | Result |
|---|---:|
| Total judged | 10 |
| Position bias count | 0 |
| Position bias rate | 0.0% |
| Verbosity bias | 88.9% |
| Decisive cases | 9 |
| B wins + B longer | 8 |
| Cohen's kappa | -0.206897 |

## Interpretation

The negative kappa means this lightweight lexical judge disagrees with the human labels more often than the chance-adjusted baseline. The high verbosity rate is also a warning that the longer ground-truth answer is frequently preferred, so the current fallback is not suitable as a production quality judge. Position bias was not observed in these swaps, but this result should be treated cautiously because only 10 pairs were evaluated.

The swap-and-average implementation correctly converts the second pass back into the original A/B coordinate system and returns `tie` on disagreement. Before production use, configure the OpenAI judge, add timeouts and JSON validation, judge multiple independent candidate pairs, and retain human review for close or safety-sensitive cases.
