# Namespace and messages (F09)

The syntax audit (#85, docs/brms-parity/reports/syntax-audit.pdf, recommendations 5 to 7) found the top-level namespace carrying
the agent scaffold's session API and the term model's internals, and the library's messages carrying the project's milestone
jargon. F09 removes both and records the call-surface differences that had no divergence row.

**Names moved out of `autobayes.__all__`.** Each is imported from its module; `ab.<name>` (and `from autobayes import <name>`)
still works and warns once per name with a `DeprecationWarning` that names the module. `from autobayes import *` no longer brings
them in.

| name | import it from |
|------|----------------|
| `environment`, `fit_record`, `compare_records`, `format_comparison`, `format_environment` | `autobayes.session` |
| `panel` | `autobayes.diagnostics` |
| `ModelSpec`, `PopTerm`, `GroupTerm`, `Categorical`, `parse_formula` | `autobayes.formula` |
| `NonlinearSpec` | `autobayes.nonlinear` |

`DeviceError` stays exported: `fit()` raises it for a device JAX cannot see, so a user catches it as `ab.DeviceError`, like
`ab.FormulaError` and `ab.PriorError`. The modules `ab.session` and `ab.diagnostics` stay reachable. The milestone examples
(`examples/m*.py`) and the CI workflows import from the modules; `docs/history/` keeps the milestone record as it was.

**Messages.** `describe()` names the family: "AutoBayes Gaussian model", "AutoBayes Bernoulli model", "AutoBayes binomial model"
(and "AutoBayes multivariate Gaussian model", "AutoBayes Gaussian model with a non-linear formula") in place of the milestone tags
M0, M1 and M2. A refusal says what is not supported ("is not supported", "is not offered") in place of "in this slice", "in this
subset" and "not part of the verified subset", and cites the current documentation and the divergence row in place of
`docs/history/m*-decision.md` and feature ids. No refusal changed what it refuses. Three say more than before: rows with missing
values (brms drops them; drop or impute them first, or model a missing predictor with `mi()`, D174), an undeclared factor (the
declaration, and R's polynomial contrasts for an ordered pandas Categorical declared with its lowest level, D172) and a Bernoulli
response outside {0, 1} (`ab.binomial` for successes out of several trials). `tests/test_namespace_and_messages.py` checks every
string literal under `autobayes/` outside a docstring for the jargon, and every refusal of the syntax corpus as AutoBayes raises it;
the corpus entries whose messages changed are re-recorded (`tests/brms_syntax/corpus.json`, message text only).

**The order of a model's free variables.** Bambi's `Family.auxiliary_parameters` is a `set` of names, and `bmb.Model` creates
the auxiliary parameters' PyMC variables in its iteration order, which depends on `PYTHONHASHSEED`. A model with two or more
auxiliary parameters (25 of the oracle cases: a Student-t model's `sigma` and `nu`, the hurdle and zero-inflated families' shape
and `hu` or `zi`, the Wiener model's `bs`, `ndt` and `bias`, the hurdle cumulative family's thresholds and `hu`, ...) had its free
variables, its posterior's variables and the layout of the sampler's unconstrained vector in another order in each process, and
`describe()` of `hurdle_cumulative_varint` listed them so. The model handed to Bambi is now `likelihoods.ordered_family`'s copy of
its family, whose auxiliary parameters come in the order its likelihood lists them; the model, its density and its target identity
are unchanged. Probed on every oracle case under six hash seeds: the free variables, `describe()`, `names`, `coefficients`, the
coordinates, the deterministics and the prior table are the same under each; before, 25 cases' free variables and one case's
`describe()` were not. Four earlier tests (`test_student_learned_nu.py`, `test_continuous_families.py`, `test_continuous_families_e02.py`,
`test_wiener.py`) had pinned the varying order, with a `describe()` that reordered it; they now check that it is the same
under two hash seeds that order Bambi's set differently.

**Divergence rows.** D172 (factor, character and logical predictors are declared), D173 (how priors cover a class), D174 (rows with
missing values are refused, not dropped) and D175 (`fit()`'s `draws`/`tune` and `target_accept=0.9` against brms's `iter`/`warmup`
and `adapt_delta = 0.8`; the posterior's Bambi names).
