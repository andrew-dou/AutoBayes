# M3a-3: multiple numeric-by-categorical interactions

2026-09-20 UTC. Decision before implementation on merged main `f23beea20607cdd049b808c0c9140072fbcd6bdd`, branch `m3a3/multiple-interactions`. Implementation dispatch SHA256 `4c4c91df5de3e6cd8bba76994f5e46289526da1b36611532198738c9a7373c40`. The independently reviewed source design and 25 source snapshots are archived at `/Users/andrewdou/iqss-research/artifacts/2026-09-20-autobayes-multiple-interactions-design/` (proposal SHA256 `524539b691a887683e899a923c6580be53cfb06dd4a1c7f0b21e3f4a0154606f`). Design approval is not executed validation.

## Observed basis

The M3a2 repair deliberately imposed one interaction after broader forms built outside its advertised scope. Preserve that repair and all historical evidence. At the reviewed source, ModelSpec, design checks, priors, value points, prediction matrices and records already iterate over ordered interaction pairs. Formulae 0.6.2 groups categorical encoding by numeric components; an explicit numeric main effect supplies the lower-order span, leaving reduced interaction coding. Bambi 0.17.2 constructs each term's coefficient vector and coordinates, centers every expanded column when an intercept exists, and back-transforms using all column means. The concrete singleton assumptions are the parser count guard and `describe()`'s first-partner/pairwise slope explanations. These are source observations, not new builds.

Official [brms formula](https://paulbuerkner.com/brms/reference/brmsformula.html) and [prior](https://paulbuerkner.com/brms/reference/set_prior.html) documentation are semantic blueprints, not claims of complete parity. The locked Python stack and checked original-input coding define this implementation.

## Accepted contract

Permit any finite list of distinct numeric-by-declared-categorical pairs with both main effects present. Shared factor (`x+w+z+x:z+w:z`), shared numeric (`x+z+v+x:z+x:v`), disjoint pairs and three-pair combinations use the same API. Expand each `*` at its position; main effects and interaction pairs retain first-appearance order, pairs canonicalize to `numeric:categorical`, and group terms retain their order.

Preserve strict duplicate refusals, including overlapping shorthand (`x*z+w*z` repeats `z`), reversed pairs and duplicate explicit mains. Preserve refusals of missing mains, numeric-numeric, categorical-categorical, higher-order interactions, transformed terms, splines, distributional predictors, slopes and other unsupported grammar. No new formula or prior framework.

Each pair has raw columns `x_i*1[z_i=l]` for fitted non-reference levels in sorted order. Keep variable `x:z`, coordinate `x:z_dim`, column labels `x:z[l]`, flattened value-point order, and exact prior mappings. One term accepts a location prior or exact level mappings with one distribution family per term; different terms may use existing different families/parameters. Missing, extra, unused and ambiguous keys remain errors.

With an intercept and requested centering, center every raw expanded column at its actual fitted-row mean; product-column mean is `mean(x*indicator)`, not a product of means. Report raw intercept `a=a_c-mean(X_raw)@b`. Without an intercept, keep no Intercept/prior and disabled effective centering: first categorical main effect is full-level (reference first), other categorical mains and every interaction remain reduced. Ordering factors or changing references can change independent-prior meaning.

Describe composed semantics. A treatment-coded factor's contrast is `b_z[l]+sum_x x*b_x:z[l]` over all numeric partners; a full-level factor's contrast starts with `b_z[l]-b_z[reference]`. A numeric slope is `b_x+sum_z b_x:z[z_i]` over all categorical partners, with reference contributions zero. Qualify baseline/reference claims by other predictors/references and group effects; retain readable accurate single-interaction wording. Group effects remain additive and are not centered.

Predictions retain raw coefficient arithmetic, known-group and manual/new-group paths, existing family/link and binomial probability/expected-count semantics, mixed group policies and exact labels. Every unseen categorical predictor level fails regardless of group policy. Records retain all interaction pairs, terms, priors, means and coordinates. Reference maps preserve predictors/likelihoods with a transported joint prior; do not claim posterior invariance under unchanged independent priors or add a public prior-transport API.

## Feature-to-test matrix and stopping rules

All expected columns, labels and row filters come from original caller data/declarations, never model.data/design/raw_design.

| Feature | Independent deterministic evidence required |
| --- | --- |
| Shared factor, shared numeric, disjoint and three-pair forms | Unequal category counts/proportions, nonalphabetical references, distinct conditional numeric means; canonical/reversed/shorthand ordering, labels, vector dimensions. |
| Three families, intercept/centering and groups | Meaningful combinations of population-only and two unequal groups, both categorical-main orders without intercept; graph predictor and raw/centered/product columns. |
| Priors, density and gradients | Unequal term/level Normal parameters and a different supported location family; normalized likelihood/prior, scale Jacobians, PyTensor/JAX gradients and original-input finite differences at nonzero points. |
| Reference changes | Explicit coefficient maps for both shared-side forms, intercept states and simultaneous categorical reference changes; predictor/likelihood equality, unequal unchanged independent priors, and correct transported density/Jacobian. |
| Predictions, descriptions and records | Synthetic named draws, training/reordered/new rows, known/mixed new groups, binomial probability/count semantics, nonzero summed contrasts/slopes and complete JSON records. |
| Filtering and compatibility | Independent binomial zero-trial removal before fitted labels/means; removed-only prediction refusal, exact large-integer labels, name/prior/grammar refusals, legacy calls. Replace only obsolete count-cap expectations with positive coverage. |

Build-only/focused deterministic invocations: at most 300 seconds each, 900 seconds aggregate. Record every invocation/failure under exclusive process-group supervision. Then ONE inherited fast suite, at most 900 seconds; its existing CPU sampler fixtures are authorized and it is not sampler-free. Never repeat the full suite; focused repairs only within remaining budget. If runtime coding differs from the contract or an independent check fails, retain the failure and investigate; do not weaken thresholds.

No new feature MCMC, slow suite, SBC/calibration, prior-predictive campaign, GPU or benchmark. A later CPU backend comparison needs a separate frozen dispatch. Own-package version may change; third-party pins and sampler defaults may not. Root owns independent review, merge and cleanup. PR8 and the original dirty checkout remain separate.

## Execution status

Pending implementation and validation. PR11 merged at the base above after independent code/draw review and final hosted checks (530 tests, 7 deselected). Its one crossed-Gaussian CPU unit passed the prescribed diagnostics, five mean gates and retained-draw semantics; this establishes neither multiple-interaction sampling nor calibration/GPU coverage. Subsequent results must be recorded as observed.


## Implementation checkpoint (2026-09-20 UTC)

The parser now accepts several distinct pairs while preserving the original strict grammar and existing graph/prior/prediction paths. `describe()` composes all numeric partners for factor contrasts and all categorical partners for numeric slopes, including no-intercept full-level contrasts and qualified baseline statements. Version 0.8.0 changes only AutoBayes's own manifest/lock entry; third-party lock entries and sampler defaults are unchanged. The offline environment was installed before the version edit, so recorded editable dist-info remains 0.7.0 while source/manifest/lock are 0.8.0; no evidence reinstallation occurred.

The new module adds 33 deterministic cases; focused-1 passed 32 cases and focused-2 passed the build-only example plus five legacy cases. Total focused monotonic time was 164.89762258299743 seconds (165.111948 UTC seconds), within the aggregate 900-second budget and each 300-second limit. Original-input matrices, densities, Jacobians, gradients, explicit reference/prior-transport maps, descriptions, labels, records and synthetic predictions passed across the selected three-family compositions. The small example only builds a model.

The ONE inherited fast suite is **incomplete**. Root terminated its group after 1605.664287 UTC seconds exceeded the 900-second cap. The original supervisor measured 584.2125582499721 monotonic seconds and reported return code -15 with `timed_out=false`; both that unchanged receipt and the separate root intervention are retained. Partial dots are not a completion result or failure count. The cause of the clock discrepancy is unknown. No full-suite retry or later numerical command occurred. Future supervisors must enforce both UTC and monotonic deadlines. Required hosted checks remain necessary for a completed suite result.

See [new evidence](m3a3-evidence/README.md), including the preserved timing-policy failure and exact source hashes. No new-feature MCMC, calibration, slow suite, GPU or benchmark was run. Independent review, hosted checks and any later separately scoped CPU acceptance remain distinct from these passed deterministic checks.
