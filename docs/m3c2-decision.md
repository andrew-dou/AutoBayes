# M3c-2: explicit population intercept controls

Decision recorded September 19, 2026 before implementation. Base 516d8f4.

## Measured semantics and contract

Build-only evidence: `m3c2-evidence/coding_probe.py` and `.json`, Bambi 0.17.2 / formulae 0.6.2. Twenty Gaussian builds (ten formulas, both centering requests) show no Intercept value variable, identical raw-design linear predictors (maximum absolute discrepancy zero), full-level coding for the first categorical main effect in formula order, reduced coding for later categorical main effects, and reduced numeric-by-categorical interaction coding when both lower-order terms are present. A numeric main effect before the first factor does not consume its full-level coding; adding a varying intercept does not change common coding. Bambi backend/model_components.py explicitly centers only when has_intercept AND center_predictors.

Primary blueprint: [brmsformula](https://paulbuerkner.com/brms/reference/brmsformula.html) documents cell-mean coding on intercept removal and centering conditional on an intercept; [Bambi getting started](https://bambinos.github.io/bambi/notebooks/getting_started.html#coding-of-categorical-variables) documents full-rank factor coding. These support the contract; they do not establish whole-library parity.

Support top-level `0 +` or subtraction of the literal `1`, canonicalized to `0 + ...`, across the three current families and optional single varying intercept. Other subtraction and group-term changes remain refused. Explicit contradictory inclusion/removal tokens are refused rather than silently resolved by ordering. At least one population predictor is required when the intercept is absent; pure zero / group-only models remain outside this increment.

Keep reference-first then sorted-other-level order. The first categorical main effect has all these coefficients, including reference; remaining categorical main effects and the one interaction retain non-reference levels. Explicit level priors must cover the actual term coefficient set. Ordering multiple factors can change independent prior meaning; disclose it. No implicit prior transport or claim of prior invariance.

`Priors(intercept=None, b=...)` (or omit intercept) is required without a population intercept, and supplying an intercept prior is an error. Default-intercept models still require an explicit intercept prior. Preserve existing positional Priors arguments; omitted b fails validation, not an automatic prior. `value_point` follows the same required/forbidden intercept rule. No Intercept variable, posterior row, or intercept-prior entry appears when absent. Zero is a valid explicitly supplied intercept value only when the model has an intercept.

Without a free intercept the effective design is raw for either centering request: centering would change the target. Preserve the requested flag separately in fit records, record effective false, describe this policy, and perform no intercept back-transformation. Default-intercept semantics are unchanged.

## Acceptance fixed before code

Meaningful deterministic original-input design, normalized density/Jacobian, PyTensor/JAX gradients, graph names, explicit-prior coverage, fixed-draw predictions (in-sample/new rows/known and new groups), describe and synthetic fit-record checks. Cover numeric, categorical, interaction, multiple-factor ordering and both centering requests across Gaussian/Bernoulli/binomial; retain existing default/grouped regressions. Update obsolete refusal tests to supported forms with positive coverage, never relax numerical criteria.

Run targeted tests, then ONE inherited fast suite under a 600-second process-group wall deadline. Existing CPU sampling fixtures are authorized regression sampling; no new feature sampling, slow runs, GPU, dependency upgrades or retry campaign. Preserve failure/timeout. Update skill/example/support matrix and test mapping, distinguishing deterministic evidence from unperformed no-intercept sampling/GPU validation. Open a focused PR and leave it open for independent review and bounded sampling acceptance. PR8 and the original dirty checkout remain untouched.

## Implementation checkpoint

Version 0.6.0 candidate. Initial targeted run: 83 passed, one refusal-message mismatch (66.47s); numerical/prediction checks passed. Moved the empty-predictor refusal after missing interaction-main-effect diagnostics. Affected grammar rerun: 30 passed (0.48s). Historical first failure is retained under m3c2-evidence. No sampling in these new tests. One full inherited fast suite is next under the committed supervisor's 600-second wall cap. Results pending.

| Contract | Tests in tests/test_intercept_controls.py |
|---|---|
| Equivalent grammar and narrow refusals | test_spellings, test_scoped_refusals |
| Raw design, graph, both centering requests, 3 families, group/no group | test_design_graph_and_prior_contract |
| Normalized density, log-scale Jacobians, PyTensor/JAX gradients against original-input SciPy reference | test_density_jacobian_and_gradients |
| Required/forbidden priors, per-level Normal/StudentT and default-intercept compatibility | test_prior_coverage_and_default_intercept_compatibility |
| First categorical full-level, second treatment, interaction reduced | test_multiple_factors_only_first_spans_intercept |
| Fixed-draw in-sample/new-row, known/new-group predictions and synthetic fit records | test_fixed_draw_predictions_and_records |

Synthetic constant-scale draws can produce undefined ArviZ scale diagnostics; their records use a clearly labelled panel stub and are not diagnostic acceptance evidence. CPU feature sampling and GPU validation remain pending.

## Full regression outcome and assertion repair (2026-09-19)

The ONE local full fast-suite invocation at f85b4db finished (no timeout) in 473.82 wall seconds, pytest 466.82s: **474 passed, 3 failed, 7 slow checks deselected**. All 84 new intercept-control checks passed in this invocation. Preserve `m3c2-evidence/fast-suite-initial.*`; this historical full run is failed, not relabeled successful. The inherited CPU fixture fits ran; this was not a no-sampling invocation. No new feature recovery fit or GPU run occurred.

The three failures were test expectations left behind by the API extension: binomial and interaction refusal tests expected FormulaError for now-supported intercept removal, whereas their supplied intercept prior is correctly rejected with PriorError; the categorical fit-record test expected an exact dictionary without the newly explicit `coding: treatment` field. No density/gradient/prediction failure occurred. Only assertions were aligned, with no library-code or numerical-threshold change after that full run. An added default-intercept synthetic-record check verifies the new metadata and preserved semantics without sampling. The two affected refusal checks plus this synthetic check passed: **3 passed in 2.18s**. The previously sampled categorical record test was not rerun locally. No second local full suite was launched. Hosted exact-head checks and independent review remain pending.

Build-only example executed successfully. Skill metadata validation passed with the existing system Python/PyYAML; the first check in the locked library venv lacked PyYAML, so no dependency was added to that stack. Only the own-package version changed in uv.lock (0.5.0 → 0.6.0).
