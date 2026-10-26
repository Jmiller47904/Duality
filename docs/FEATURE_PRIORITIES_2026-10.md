# October 2026 feature priorities

Ranked against current source, open work and recent CI on October 26, 2026.
Effort and risk are relative estimates. This PR is a reviewable increment, not a release.

| Rank | Candidate | Value | Effort | Risk | Reason |
| --- | --- | --- | --- | --- | --- |
| 1 | Deterministic CPU memory-cost ablation (this PR) | High | Small | Low | Replace a universal overhead claim with same-host latency, parameter and retained-state measurements. |
| 2 | Trained memory-on/off recall and interference evaluation | High | Medium | Medium | Measure task accuracy and interference with controlled seeds; the CPU ablation does not make a quality claim. |
| 3 | Contradiction provenance ledger | Medium | Large | High | Track conflicting stored observations without pretending to establish truth. |
