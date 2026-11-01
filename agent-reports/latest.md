# Duality agent report

Fixed a reproducibility bug in the story-comprehension report: V1 statistics now include `burn_in_active`, computed using the same configured strength or token threshold as the forward gate. Added regression coverage and updated the V1 and benchmark documentation.

PR: [#21 — Report V1 burn-in state in attractor statistics](https://github.com/Jmiller47904/Duality/pull/21)

Checks completed locally: `compileall` and `git diff --check` passed. The focused pytest test could not start because pytest is unavailable; PyTorch is also unavailable. The costly Pythia benchmark was not run.

PR #21 is open at head `66542ff2`. GitHub reports mergeability unstable; CI and smoke checks are still running. It remains unmerged as required. No open issues or other PRs were found. No release published.