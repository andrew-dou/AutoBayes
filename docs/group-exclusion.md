# Predictions without group effects

`model.predict(fit, include_group_effects=False, kind="mean")` sets all group
contributions to zero, retaining the uncertainty in fixed coefficients. Works
with linear, mean, response, and binomial probability predictions. The default
True preserves the existing conditional group predictions.

This is not integration over the group-effect distribution. Under nonlinear
links, the inverse link at zero effects generally differs from the mean over
random effects. It is also different from new_groups="population", which draws
unseen-group effects from their population distribution.

When excluding effects, newdata needs fixed predictors and any required binomial
trials, but no grouping-only columns. Keep new_groups at its default "error";
other policies are refused rather than ignored. Fixed categorical validation and
fit-target binding remain in force. The existing formula restriction against
using a grouping factor as a fixed predictor is unchanged (the initial design
considered that case before this restriction was verified).

The output records `include_group_effects=0` (an integer flag, 1 when group effects are included, so that the
dataset stays netCDF-serialisable) and `new_groups="not_applicable"`.
Group arrays are not accessed and no group-effect random draws are consumed.
There is no subset selection of groups yet. Tests use synthetic posterior draws;
35 focused checks pass across five families and existing linear/binding behavior.
No fits, forward-prior generation, GPU or calibration campaign was performed.
