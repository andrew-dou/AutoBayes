# Named family metadata contract

September 25, 2026. Starting from ba4cc02 (merged PR12).

Replace positional entries in the existing private family registry with immutable named metadata. Route existing Bambi parent, link and prediction-parameter names through it. Preserve all constructors, priors, likelihoods, fitting, prediction algorithms, support errors and record schemas. The binomial legacy parameter-output name stays p; expected count remains a separate expected=n*p output. No new response family, default or dependency.

Validate public Gaussian, Bernoulli and binomial model names/link and fixed-draw mean predictions on training and modified covariates, with unequal binomial trials. No MCMC is needed. Existing likelihood and sampling evidence is retained; deterministic tests do not claim new GPU or calibration evidence.

This is the first small slice of the family-contract plan. Auxiliary-parameter and prediction-kind refactors remain future work, as does Student-t response support. No change to the outstanding PR8 acceptance decision.
