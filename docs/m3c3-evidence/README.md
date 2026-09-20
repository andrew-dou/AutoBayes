# M3c-3 implementation evidence — 2026-09-20 UTC

Decision commit `0f7ce72` precedes implementation on merged main `9755f1d`. The sole inherited fast-suite invocation passed **524 tests, 7 deselected**, exit 0, without timeout in **501.33 wall seconds** (pytest 490.58s). It includes existing CPU sampler fixtures; this is not a sampler-free run or feature-specific MCMC acceptance.

| Invocation | Outcome | Actual wall seconds |
| --- | --- | ---: |
| focused-1 | 62 passed, 1 failed: single-group describe notation; implementation repaired | 166.8117 |
| focused-2 | 194 passed, 2 failed: obsolete multiple-group refusal and exact missing-SD message; repaired | 149.8062 |
| focused-3 | 10 passed: repaired cases, name guards, example, records | 11.5881 |
| full-1 | 524 passed, 7 deselected, 7 warnings; existing CPU fixture sampling | 501.3325 |
| focused-4 | 16 passed: ModelSpec copying/serialization checks; mutable candidate later failed independent review | 15.4797 |
| focused-5 | 19 passed: immutable metadata repair, source/public mutation, copying/serialization, design/legacy | 15.3752 |

Focused total **359.06 / 900 seconds**; each cap 300 seconds. Full cap 900 seconds; exactly one invocation. No timeout, retry of the full suite, feature MCMC, GPU or third-party dependency change. AutoBayes's own version is 0.7.0. Warning locations are retained in full stdout; synthetic record diagnostics are not sampler acceptance evidence.

After the full run, static review found that storing `levels_by_group` as MappingProxyType would break ordinary dataclass copying/serialization and generated hashing. A copied dictionary and explicit hash passed 16 focused-4 checks, but independent review then reproduced mutation of a frozen specification and failed dictionary-key lookup through the public mapping. The original candidate and reproducer are preserved; see `review-metadata-regression/` and external `post-full-candidate/`. The final code owns immutable tuples behind a read-only mapping that reconstructs normally for deepcopy, pickle and dataclass `asdict`; 19 focused-5 checks include attempted public assignment and mutation of the caller’s source mapping/list, with stable hashes and dictionary lookup. Two formula doc/error strings were also clarified. Model mathematics and sampler settings did not change. The original full run and pre-full source state are preserved, not silently relabeled as a final-code full rerun.

Every run retains exact argv, HEAD/dirty status, input file hashes, environment overrides, child process group, stdout/stderr and exit receipt. `pre-full.diff` retains the exact tracked working changes; its single-space blank context lines are preserved even though `git diff --check` flags them when the patch is added as a file. The full source snapshot (including then-untracked new tests) is retained in `/private/tmp/autobayes-m3c3-implementation/full-source-snapshot/`; root owns durable external archival. Launch manifests describe their historical inputs and need not match later repairs. Do not execute the historical supervisor blindly: it uses exclusive attempt directories and refuses a second full suite.

New deterministic tests independently construct common/group matrices from original caller rows, including unequal group counts, equal labels across factors, three families, intercept and centering combinations, categorical interaction coding, normalized priors/likelihood/Jacobians, PyTensor/JAX gradients, fixed-scale Gaussian precision, synthetic prediction arithmetic, mixed unseen policies, label sharing and factor/row ordering, binomial zero-trial filtering, names, prior/value refusals, records, legacy constructors and the build-only example. Synthetic response draws only check count support; no new sampler is invoked by these tests.

Independent code review is in progress; its confirmed metadata regression is repaired as recorded above. Exact-head hosted checks and any separately authorized multigroup CPU scientific acceptance remain pending. No calibration or GPU claim is made. PR8, its historical failures, and prior milestone evidence were not changed.
