# Column dictionary

**Status:** Active
**Last updated:** (set on first entry)

## Summary

Every column in the fitted dataset: what it means physically, its range, and what role
it plays. Grouped by use class, because "is this a predictor or a leftover?" is the
question that actually gets asked.

An agent proposing a model needs this to avoid two specific errors: putting an
identifier in the design matrix, and inventing a covariate that already exists under a
different name.

## Outcome

| Column | Meaning | Scale / range | Notes |
| --- | --- | --- | --- |
| | | | |

Record the scale the model works on and the transform used to get there. A likelihood
written on the logit scale and a column stored on the natural scale is a bug that
produces a plausible fit.

## Predictors

| Column | Meaning | Scale / range | Transform | Missing | In model? |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

**Missing** is load-bearing. State whether the column is complete, imputed before the
fit, or imputed jointly inside the model — the three have different consequences for
the posterior, and only the third propagates uncertainty honestly.

## Grouping variables

| Column | Levels | Construction | Notes |
| --- | --- | --- | --- |
| | | | |

Record how levels are constructed and the ordering rule. Non-deterministic factor
ordering is a leading cause of silent divergence between two implementations of the
same model, and it is invisible until the coefficients stop matching.

## Identifiers

Keys and labels. Not predictors — list them so nobody mistakes one for a predictor.

| Column | Meaning |
| --- | --- |
| | |

## Vestigial

Columns present in the file and used by nothing. Say so explicitly. An unlabelled
unused column is a standing invitation for someone to add it to the model on the theory
that it must be there for a reason.

| Column | Why it is still here |
| --- | --- |
| | |

## Related

[[data/provenance]]
