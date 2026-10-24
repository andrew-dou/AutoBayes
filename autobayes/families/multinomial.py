"""Count-matrix responses: brms's ``multinomial()`` and ``dirichlet_multinomial()`` families (brms-parity F01a,
docs/multinomial.md).

The response is brms's matrix ``cbind(y_1, ..., y_K) | trials(n)``: each row's counts of K categories, the columns' names
the categories in the order written, the first the reference, and ``n`` the row's total (brms requires ``trials()`` and
refuses a row whose counts do not sum to it). Each non-reference category ``k`` has its own linear predictor ``mu_k``
(brms's ``mu<k>``, with its own centred intercept, coefficients and group-level terms, ``autobayes.multinomial``); the
reference category's is 0 and the category probabilities are ``p = softmax(0, mu_2, ..., mu_K)``. With ``N = n`` and ``C(y) =
N! / prod_k y_k!`` the multinomial coefficient, the log densities are the published ones (checked against brms's
``log_prob`` on the frozen cases, constants included):

    multinomial            log C(y) + sum_k y_k log p_k                           (Stan's multinomial_logit_lpmf)
    dirichlet_multinomial  log C(y) + lgamma(phi) - lgamma(N + phi)
                                    + sum_k [lgamma(y_k + phi p_k) - lgamma(phi p_k)]  (the Dirichlet-multinomial, alpha = phi p)

The coefficient is a sum of stable log binomial coefficients, ``log C(y) = sum_k log C(r_k, y_k)`` with ``r_k = y_k + ... +
y_K`` (``likelihoods.log_binomial_coefficient_sym``, M2's form, never a difference of three lgamma values); ``log p`` is the
log-softmax with the largest entry factored out (B05's ``_log_sum_exp``); each Dirichlet-multinomial difference of lgamma
values is a log rising factorial written through Stirling's remainder (``likelihoods._log_rising``, D03's form for the
negative binomial), and below ``log alpha = -40`` it is ``lgamma(y) + log alpha``, its value to double precision there (the
next term, ``alpha (psi(y) + Euler's gamma)``, is below 1e-16 of the value), so that a category probability that underflows
leaves the row finite with a finite gradient on both paths (the other branch's gradient, clamped at the floor, stays finite).
``phi`` is the Dirichlet-multinomial's precision (brms's ``real<lower=0> phi``, sampled as ``log(phi)``); the multinomial is
its limit as ``phi`` grows.

``predict(kind="mean")`` returns ``p`` and ``expected = n p`` (brms's ``posterior_epred``), ``kind="response"`` count draws
(the multinomial's; for the Dirichlet-multinomial a probability vector from ``Dirichlet(phi p)`` first, drawn on the log scale
so that a tiny concentration never gives an all-zero vector). brms refuses ``bayes_R2`` for these families, and so does this
library (``FamilySpec.per_category``).
"""
from __future__ import annotations

import numpy as np

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec
from autobayes.priors import Gamma

LOG_ALPHA_FLOOR = -40.0      # below it lgamma(y + alpha) - lgamma(alpha) is lgamma(y) + log(alpha) to double precision


# -- PyTensor: the likelihoods -------------------------------------------------------------------------------------------
def log_softmax(full):
    """``log softmax(full)`` over the last axis, the largest entry factored out (``likelihoods._log_sum_exp``): finite with
    a finite gradient for every finite argument on the PyTensor path and under JAX."""
    from autobayes.likelihoods import _log_sum_exp

    return full - _log_sum_exp(full)[..., None]


def log_multinomial_coefficient(y):
    """``log(N! / prod_k y_k!)`` per row of the counts ``y`` (``(..., K)``, float64): ``sum_k log C(r_k, y_k)`` with ``r_k =
    sum_{j >= k} y_j``, each through M2's stable log binomial coefficient (``r_k`` at least 1 there, so that a zero row's
    terms are its exact zeros)."""
    from autobayes.likelihoods import log_binomial_coefficient_sym

    import pytensor.tensor as pt

    remaining = pt.cumsum(y[..., ::-1], axis=-1)[..., ::-1]
    return pt.sum(log_binomial_coefficient_sym(pt.maximum(remaining, np.float64(1.0)), y), axis=-1)


def _valid_counts(y, n):
    """Whether each row of ``y`` holds whole non-negative counts summing to its trials ``n``."""
    import pytensor.tensor as pt

    whole = pt.all(pt.and_(pt.ge(y, 0.0), pt.eq(y, pt.floor(y))), axis=-1)
    return pt.and_(whole, pt.eq(pt.sum(y, axis=-1), n))


def multinomial_logit_logp(value, full, n):
    """``log Multinomial(value | n, softmax(full))`` per row, brms's ``multinomial_logit2_lpmf`` (Stan's
    ``multinomial_logit_lpmf``) with ``full = (0, mu_2, ..., mu_K)``; ``-inf`` for a row that is not ``n`` whole non-negative
    counts."""
    import pytensor.tensor as pt

    y = pt.cast(value, "float64")
    nf = pt.cast(n, "float64")
    body = log_multinomial_coefficient(y) + pt.sum(y * log_softmax(full), axis=-1)
    return pt.switch(_valid_counts(y, nf), body, -np.inf)


def log_rising(log_alpha, y):
    """``lgamma(y + alpha) - lgamma(alpha)`` for counts ``y >= 0`` and ``alpha = exp(log_alpha)``: 0 at ``y = 0``,
    ``likelihoods._log_rising`` where ``log_alpha >= LOG_ALPHA_FLOOR`` and ``lgamma(y) + log_alpha`` below it; each branch
    is evaluated on arguments moved into its own range, so that the branch a switch does not take stays finite."""
    import pytensor.tensor as pt

    from autobayes.likelihoods import _log_rising

    floor = np.float64(LOG_ALPHA_FLOOR)
    positive = pt.gt(y, 0.0)
    y_pos = pt.switch(positive, y, np.float64(1.0))
    tiny = pt.lt(log_alpha, floor)
    low = pt.gammaln(y_pos) + pt.minimum(log_alpha, floor)
    high = _log_rising(pt.exp(pt.maximum(log_alpha, floor)), y_pos)
    return pt.switch(positive, pt.switch(tiny, low, high), 0.0)


def dirichlet_multinomial_logit_logp(value, full, phi, n):
    """``log DirichletMultinomial(value | n, phi softmax(full))`` per row (brms's ``dirichlet_multinomial_logit2_lpmf``):
    ``log C(y) + lgamma(phi) - lgamma(N + phi) + sum_k [lgamma(y_k + alpha_k) - lgamma(alpha_k)]`` with ``log alpha =
    log(phi) + log softmax(full)``, the differences as ``log_rising``; ``-inf`` for a row that is not ``n`` whole non-negative
    counts."""
    import pytensor.tensor as pt

    y = pt.cast(value, "float64")
    nf = pt.cast(n, "float64")
    log_phi = pt.log(phi)
    log_alpha = log_phi[..., None] + log_softmax(full)
    body = (log_multinomial_coefficient(y) + pt.sum(log_rising(log_alpha, y), axis=-1)
            - log_rising(log_phi, pt.sum(y, axis=-1)))
    return pt.switch(_valid_counts(y, nf), body, -np.inf)


# -- NumPy: probabilities, pointwise log densities, draws ------------------------------------------------------------------
def probabilities(eta):
    """``softmax(0, eta)`` over the last axis: ``eta`` ``(..., K - 1)`` holds the non-reference categories' linear
    predictors, the result ``(..., K)`` every category's probability (``likelihoods.categorical_probabilities``)."""
    from autobayes.likelihoods import categorical_probabilities

    return categorical_probabilities(eta)


def _log_softmax_full(full):
    """``log softmax(full)`` over the last axis (NumPy), the largest entry factored out."""
    full = np.asarray(full, dtype=float)
    lp = full - np.max(full, axis=-1, keepdims=True)
    return lp - np.log(np.exp(lp).sum(axis=-1, keepdims=True))


def _log_softmax_np(eta):
    """``log softmax(0, eta)`` over the last axis (NumPy): ``(..., K - 1)`` -> ``(..., K)``."""
    eta = np.asarray(eta, dtype=float)
    return _log_softmax_full(np.concatenate([np.zeros(eta.shape[:-1] + (1,)), eta], axis=-1))


def log_multinomial_coefficient_np(y):
    """``log(N! / prod_k y_k!)`` per row (NumPy): ``sum_k log C(r_k, y_k)`` through ``pointwise.log_binomial_coefficient``."""
    from autobayes.families.pointwise import log_binomial_coefficient

    y = np.asarray(y, dtype=float)
    remaining = np.cumsum(y[..., ::-1], axis=-1)[..., ::-1]
    return log_binomial_coefficient(remaining, y).sum(axis=-1)


def log_rising_np(log_alpha, y):
    """``lgamma(y + alpha) - lgamma(alpha)`` with ``alpha = exp(log_alpha)`` (NumPy): 0 at ``y = 0``, ``lgamma(y + 1) -
    log(y + alpha) - log B(alpha, y + 1)`` (SciPy's ``betaln``, exact where a difference of two lgamma values of size ``alpha
    log alpha`` is not), and ``lgamma(y) + log_alpha`` below ``LOG_ALPHA_FLOOR``."""
    from scipy.special import betaln, gammaln

    log_alpha = np.asarray(log_alpha, dtype=float)
    y = np.broadcast_to(np.asarray(y, dtype=float), np.broadcast_shapes(np.shape(y), log_alpha.shape))
    y_pos = np.where(y > 0, y, 1.0)
    alpha = np.exp(np.maximum(log_alpha, LOG_ALPHA_FLOOR))
    high = gammaln(y_pos + 1.0) - np.log(y_pos + alpha) - betaln(alpha, y_pos + 1.0)
    low = gammaln(y_pos) + np.minimum(log_alpha, LOG_ALPHA_FLOOR)
    return np.where(y > 0, np.where(log_alpha < LOG_ALPHA_FLOOR, low, high), 0.0)


def multinomial_logit_rows(y, eta):
    """brms's ``log_lik()`` of the multinomial (C07): ``log C(y_i) + sum_k y_ik log softmax(0, eta_i)_k`` per draw and row;
    ``y`` ``(rows, K)``, ``eta`` ``(..., rows, K - 1)``."""
    y = np.asarray(y, dtype=float)
    return log_multinomial_coefficient_np(y) + (y * _log_softmax_np(eta)).sum(axis=-1)


def dirichlet_multinomial_logit_rows(y, eta, phi):
    """brms's ``log_lik()`` of the Dirichlet-multinomial: the density above per draw and row, ``phi`` broadcastable against
    ``eta``'s leading axes (one per draw)."""
    y = np.asarray(y, dtype=float)
    log_phi = np.log(np.asarray(phi, dtype=float))
    log_alpha = log_phi[..., None] + _log_softmax_np(eta)
    total = y.sum(axis=-1)
    return log_multinomial_coefficient_np(y) + log_rising_np(log_alpha, y).sum(axis=-1) - log_rising_np(log_phi, total)


def draw_multinomial(p, trials, rng):
    """Count draws ``Multinomial(trials_i, p_i)``: ``p`` ``(..., rows, K)``, ``trials`` ``(rows,)``; the result has ``p``'s
    shape (NumPy's generator broadcasts the trials against the probabilities' leading axes)."""
    p = np.asarray(p, dtype=float)
    p = p / p.sum(axis=-1, keepdims=True)                           # exact simplices for the generator
    n = np.broadcast_to(np.asarray(trials, dtype=np.int64), p.shape[:-1])
    return rng.multinomial(n, p).astype(np.int64)


def draw_dirichlet_log(log_alpha, rng):
    """Draws of ``Dirichlet(exp(log_alpha))`` over the last axis, on the log scale: ``log G_k = log Gamma(alpha_k + 1) +
    log(U) / alpha_k`` (a ``Gamma(alpha_k)`` variable, the boost of Marsaglia and Tsang 2000), then the softmax of ``log G``,
    so that a tiny concentration gives a tiny probability rather than an all-zero vector."""
    log_alpha = np.asarray(log_alpha, dtype=float)
    alpha = np.exp(log_alpha)
    with np.errstate(over="ignore", divide="ignore"):
        log_g = np.log(rng.gamma(alpha + 1.0)) + np.log(rng.uniform(size=alpha.shape)) / alpha
    m = log_g.max(axis=-1, keepdims=True)
    e = np.exp(log_g - m)
    return e / e.sum(axis=-1, keepdims=True)


def multinomial_logit_random(full, n, rng=None, size=None):
    """Forward draws of the multinomial likelihood (``pm.CustomDist``'s ``random``; the prior predictive): ``full``
    ``(..., K)`` the linear predictors with the reference category's 0, ``n`` the trials; ``size`` the batch shape."""
    p = np.exp(_log_softmax_full(full))
    if size is not None:
        p = np.broadcast_to(p, tuple(size) + p.shape[-1:])
    return draw_multinomial(p, np.broadcast_to(n, p.shape[:-1]), rng)


def dirichlet_multinomial_logit_random(full, phi, n, rng=None, size=None):
    """Forward draws of the Dirichlet-multinomial likelihood: a probability vector from ``Dirichlet(phi softmax(full))``
    (``draw_dirichlet_log``), then the multinomial counts."""
    log_alpha = np.log(np.asarray(phi, dtype=float))[..., None] + _log_softmax_full(full)
    if size is not None:
        log_alpha = np.broadcast_to(log_alpha, tuple(size) + log_alpha.shape[-1:])
    p = draw_dirichlet_log(log_alpha, rng)
    return draw_multinomial(p, np.broadcast_to(n, p.shape[:-1]), rng)


# -- the family hooks ----------------------------------------------------------------------------------------------------------
def _no_bambi():
    """No Bambi family: ``autobayes.multinomial.MultinomialModel`` builds the graph itself."""
    raise RuntimeError("the multinomial families are built by autobayes.multinomial, not by Bambi")


def _log_likelihood_multinomial(model, post, eta):
    """brms's log_lik (C07): each fitted row's log multinomial mass with its coefficient; ``eta`` ``(chain, draw, row, K - 1)``."""
    return multinomial_logit_rows(model.design.y, eta)


def _log_likelihood_dirichlet_multinomial(model, post, eta):
    """brms's log_lik of the Dirichlet-multinomial: each fitted row's log mass, ``phi`` per draw."""
    return dirichlet_multinomial_logit_rows(model.design.y, eta, post[model.names["phi"]].to_numpy()[..., None])


def _trials(model, df) -> np.ndarray:
    return df[model.spec.trials].to_numpy(dtype=np.int64)


def _draw_multinomial_response(model, post, eta, mean, df, rng):
    """brms's posterior_predict: ``Multinomial(n_i, p_i)`` per draw and row; ``mean`` holds ``p`` ``(..., rows, K)``."""
    return draw_multinomial(mean, _trials(model, df), rng)


def _draw_dirichlet_multinomial_response(model, post, eta, mean, df, rng):
    """``Multinomial(n_i, q)`` with ``q ~ Dirichlet(phi p_i)`` per draw and row, ``log(phi p)`` as ``log(phi) + log
    softmax(0, eta)`` (never the log of an underflowed probability)."""
    log_alpha = np.log(post[model.names["phi"]].to_numpy())[..., None, None] + _log_softmax_np(eta)
    return draw_multinomial(draw_dirichlet_log(log_alpha, rng), _trials(model, df), rng)


def _validate_response(spec, data) -> None:
    """The count matrix and its trials are validated by ``autobayes.multinomial.response_counts`` (``MultinomialModel``
    reads the matrix itself; the generic ``formula.validate_data`` never sees a matrix response)."""


def _mean_lines(model) -> list[str]:
    from autobayes.multinomial import mean_lines

    return mean_lines(model)


def _likelihood_lines(model) -> list[str]:
    from autobayes.multinomial import likelihood_lines

    return likelihood_lines(model)


_COMMON = dict(
    parent="mu", link="logit", link_scale="logit", mean_name="p",
    response_kind="binomial", response_dtype=np.int64,              # counts with trials (has_trials: G4 reads predict()'s expected)
    fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=_no_bambi,
    validate_response=_validate_response,
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: model.design.trials[:, None],
    inverse_link=probabilities,
    expected_response=None,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean' (the category probabilities p and expected counts n p), 'response' (count draws) or 'linear' (mu)",
    scale_missing_error=None,
    linpred_symbol="mu", scale_phrase="log-odds (against the reference category)",
    intercept_meaning="log-odds of category k against the reference category at the predictor means",
    mean_lines=_mean_lines,
    likelihood_lines=_likelihood_lines,
    centred_intercept_note=(),
    deterministic_note="mu = the linear predictors of the non-reference categories; p = softmax(0, mu) and n p are computed by predict()",
    default_prior_link=None,                                        # brms never reads the counts for its defaults: student_t(3, 0, 2.5)
    per_category=True,                                              # one linear predictor per non-reference category (bayes_R2 refused)
    matrix_response=True,                                           # cbind(...) | trials(n): autobayes.multinomial builds the model
)

MULTINOMIAL = FamilySpec(
    name="multinomial", tag="multinomial",
    scale_parameter=None, bambi_scale_parameter=None,
    draw_response=_draw_multinomial_response,
    scale_rule="sigma, shape and phi do not apply to the multinomial family (no auxiliary parameter)",
    default_scale_prior=None,
    log_likelihood=_log_likelihood_multinomial,                     # C07: log_lik(), verified against multinomial_varint_post
    **_COMMON,
)

DIRICHLET_MULTINOMIAL = FamilySpec(
    name="dirichlet_multinomial", tag="Dirichlet-multinomial",
    scale_parameter="phi", bambi_scale_parameter="phi",
    draw_response=_draw_dirichlet_multinomial_response,
    scale_rule="phi (the precision) is required for the dirichlet_multinomial family; sigma and shape do not apply",
    default_scale_prior=lambda scale: Gamma(0.01, 0.01),            # brms dirichlet_multinomial phi: gamma(0.01, 0.01), lb 0 (frozen case)
    log_likelihood=_log_likelihood_dirichlet_multinomial,
    **{**_COMMON, "scale_missing_error": "the dirichlet_multinomial family needs a prior on the precision parameter phi: pass phi=..."},
)
