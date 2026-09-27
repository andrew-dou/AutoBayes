# Linear-predictor draws

`model.predict(fit, kind="linear", newdata=frame)` returns an xarray Dataset
containing `eta` on chain/draw/__obs__ dimensions. Omit newdata to use modeled
rows. The output carries prediction_scale=linear, the model link and fit binding
status. It includes the existing group effects and follows the same unseen-group
policies and seed handling as other predictions.

Eta is location for Gaussian/Student-t, log odds for Bernoulli/binomial, and log
rate for Poisson. It has neither inverse-link transformation nor observation
noise. Binomial newdata therefore needs no trials column for linear prediction;
mean/response still require trials. Finite large Poisson log-rates can be inspected
even when their exponentials cannot be represented. Nonfinite eta is refused.

This uses raw-scale posterior intercepts, including synthetic fixtures explicitly
prepared in that parameterization. Native prior-predictive output remains refused
as a posterior fit. There is no group-effect exclusion or full brms re_formula
interface yet. No new fits, forward sampling, GPU or calibration evidence is
introduced by this deterministic prediction feature.
