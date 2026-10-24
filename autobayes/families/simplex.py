"""Responses on the simplex: brms's ``dirichlet()`` and ``logistic_normal()`` (brms-parity F01b, docs/dirichlet-logistic-normal.md).

The response is written ``cbind(y1, ..., yK) ~ <terms>`` (brms's matrix response): K columns, each row K proportions strictly
inside (0, 1) that sum to one (within Stan's simplex tolerance, 1e-8). The categories are the columns, named by them; one is
the reference (the first column unless ``refcat=`` names another, brms's ``refcat``), and ``model.categories`` lists the
reference first and the others in the columns' order. As for the categorical family (B05) every non-reference category ``k``
has its own linear predictor ``mu_k`` (brms's ``mu<k>``: its own centred intercept, coefficients and, here, varying
intercepts with their own SD): Bambi's categorical family builds that layout (every coefficient, SD and standardised effect
with a trailing category axis), and the model keeps the K - 1 linear predictors as the parent ``mu`` under the identity link.

- ``dirichlet``: ``y_i ~ Dirichlet(phi * softmax(0, mu_i))`` (brms's ``dirichlet_logit_lpdf``; the reference category's
  predictor is 0), the precision ``phi`` (brms's ``real<lower=0> phi``, sampled as ``log(phi)``, ``Priors(phi=...)``).
  ``predict(kind="mean")`` is ``softmax(0, mu_i)``, brms's ``posterior_epred``. Written with the log of each mean proportion
  as ``mu_k - log_sum_exp(0, mu)`` and each ``lgamma(alpha_k)`` from ``log(alpha_k)`` where ``alpha_k`` is below 1e-10
  (``likelihoods._lgamma_of_shape``), so the density stays finite for every finite predictor.
- ``logistic_normal``: the log-ratios ``z_i = log(y_i[k] / y_i[ref])`` of the non-reference categories are
  ``MultiNormal(mu_i, diag(sigma) Lncor diag(sigma))``, and the density on the simplex carries the Jacobian of
  ``y -> z``, ``-sum_k log y_i[k]`` over all K parts (Aitchison 1986, ch. 6; brms's ``logistic_normal_cholesky_cor_lpdf(Y |
  mu, sigma, Llncor, refcat)``, checked against the frozen points). One SD per non-reference category (brms's ``sigma<k>``,
  ``sigmas2``, ...; ``real<lower=0>``, on the log scale, one prior ``Priors(sigma=...)`` for all) and the Cholesky factor of
  their correlation (brms's ``cholesky_factor_corr[K - 1] Llncor`` under ``lkj_corr_cholesky(eta)``, ``Priors(lncor=LKJ(eta))``,
  sampled on Stan's ``cholesky_corr`` coordinates, ``autobayes.correlation``). brms refuses ``posterior_epred`` for the
  family (the mean of a logistic-normal vector has no closed form): ``predict(kind="mean")`` is refused too.

``kind="linear"`` returns ``mu`` with a ``category`` dimension over the non-reference categories (brms's
``posterior_linpred``); ``kind="response"`` draws on the simplex (Dirichlet draws through Gamma variates taken on the log
scale; logistic-normal draws as ``softmax(0, mu + diag(sigma) Lncor e)``), with a ``category`` dimension over all K in the
order of ``model.categories``; ``log_lik()`` is each row's density as above. Refused, naming the construct and the divergence
row (D138 dirichlet, D139 logistic_normal): another reference category for the dirichlet family, two columns for the
logistic-normal family, group-level slopes, correlation IDs, ``mm()`` and ``gr(by = )``, offsets, addition terms, the other
special terms (by their own refusals), formulas of ``phi``, ``sigma<k>`` and ``lncor``, new levels in ``predict()`` under any
policy but ``"error"``, ``bayes_R2()`` and ``prior_predictive()``. Checked against brms in the oracle cases
``dirichlet_varint``, ``logistic_normal_varint`` and ``logistic_normal_refcat``, not assumed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from autobayes.families.spec import NO_FIXED_PARAMETERS, STANDARD_PREDICTION_KINDS, FamilySpec, scale_line
from autobayes.formula import FormulaError

DOC = "docs/dirichlet-logistic-normal.md"
SIMPLEX_TOLERANCE = 1e-8                 # Stan's check_simplex tolerance (CONSTRAINT_TOLERANCE) on each row's sum
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
_LOG_TINY = math.log(1e-10)             # below it lgamma(x) is written from log(x) (likelihoods._lgamma_of_shape's bound)
_EULER = 0.5772156649015329


@dataclass(frozen=True)
class Simplex:
    """What a simplex family (``FamilySpec.simplex``) takes: its kind, its divergence row, whether ``refcat`` may name another
    column than the first, the fewest columns it takes (and why fewer are refused), and the ``Priors`` fields it validates and
    builds itself (the logistic-normal family's ``sigma`` and ``lncor``; the Dirichlet ``phi`` is the family's
    ``scale_parameter``)."""

    kind: str                            # "dirichlet" or "logistic_normal"
    divergence: str                      # "D138", "D139"
    takes_refcat: bool
    min_categories: int
    few_categories: str = ""
    own_priors: tuple[str, ...] = ()

    @property
    def epred(self) -> bool:
        """Whether ``predict(kind="mean")`` is offered (brms's ``posterior_epred``: the Dirichlet's, not the logistic-normal's)."""
        return self.kind == "dirichlet"


DIRICHLET_PARTS = Simplex("dirichlet", "D138", takes_refcat=False, min_categories=2)
LOGISTIC_NORMAL_PARTS = Simplex(
    "logistic_normal", "D139", takes_refcat=True, min_categories=3,
    few_categories="with two, brms's correlation of the log-ratios is 1 x 1 and has no parameter, which no frozen case verifies",
    own_priors=("sigma", "lncor"))


# -- the response ---------------------------------------------------------------------------------------------------
def response_matrix(spec, frame, label: str) -> np.ndarray:
    """The ``(n, K)`` response of ``cbind(y1, ..., yK)`` from ``frame``, columns in the order written (brms's ``Y``): every
    entry a finite number strictly between 0 and 1 and every row summing to one within ``SIMPLEX_TOLERANCE``, or refused naming
    the column, the row and the value. ``label`` names the family in the messages."""
    from autobayes.formula import _is_numeric

    columns = list(spec.response_columns)
    out = np.empty((len(frame), len(columns)))
    for k, c in enumerate(columns):
        s = frame[c]
        if not _is_numeric(s):
            raise FormulaError(f"response column {c!r} has dtype {s.dtype}; the {label} family's response is {len(columns)} numeric "
                               "proportions per row")
        v = s.to_numpy(dtype=float)
        bad = np.flatnonzero(~np.isfinite(v) | (v <= 0.0) | (v >= 1.0))
        if bad.size:
            raise FormulaError(f"response column {c!r} has the value {float(v[bad[0]])!r} at row {int(bad[0])}; the {label} family's "
                               "proportions lie strictly between 0 and 1 (its density takes their logarithms)")
        out[:, k] = v
    off = np.abs(out.sum(axis=1) - 1.0)
    if off.size and off.max() > SIMPLEX_TOLERANCE:
        i = int(np.argmax(off))
        raise FormulaError(f"the response cbind({', '.join(columns)}) sums to {float(out[i].sum())!r} at row {i}; each row of a {label} "
                           f"response must sum to 1 (within {SIMPLEX_TOLERANCE:g}, Stan's simplex tolerance)")
    return out


def categories_of(spec, refcat, fam) -> tuple[tuple[str, ...], np.ndarray]:
    """The categories, the reference first (``refcat``, or the first column) and the others in the columns' order, and the
    position of each in the columns (``Y[:, order]`` puts brms's ``Y`` in category order). ``refcat`` must name a column; the
    Dirichlet family takes the first column only (D138); two categories whose brms names would clash are refused
    (``brms_dpar_names``)."""
    from autobayes.default_priors import brms_dpar_names

    columns = tuple(spec.response_columns)
    if refcat is None:
        ref = columns[0]
    else:
        if not isinstance(refcat, str) or refcat not in columns:
            raise ValueError(f"refcat must name one of the response columns {list(columns)}, got {refcat!r}")
        if refcat != columns[0] and not fam.simplex.takes_refcat:
            raise ValueError(f"refcat={refcat!r}: another reference category than the first column ({columns[0]!r}) is not offered for "
                             f"the {fam.name} family (brms's refcat; no frozen case verifies it; {DOC}, divergence "
                             f"{fam.simplex.divergence}); reorder the columns of cbind() to make {refcat!r} the first")
        ref = refcat
    order = (ref, *(c for c in columns if c != ref))
    brms_dpar_names(order[1:])                                       # refuses categories brms would name alike
    return order, np.array([columns.index(c) for c in order], dtype=np.int64)


def check_refcat(fam, refcat) -> None:
    """``refcat=`` applies to the simplex families only."""
    if refcat is not None and fam.simplex is None:
        from autobayes.families import FAMILIES

        taking = [name for name, other in FAMILIES.items() if other.simplex is not None and other.simplex.takes_refcat]
        raise ValueError(f"refcat= names the reference category of the {' and '.join(taking)} family's cbind() response; the "
                         f"{fam.name} family takes none")


def _validate_response(label: str):
    def validate(spec, data) -> None:
        if not getattr(spec, "response_columns", ()):           # a spec built without parse_formula, which refuses it first
            from autobayes.formula import SIMPLEX_RESPONSE

            raise FormulaError(f"{spec.formula!r}: {SIMPLEX_RESPONSE.format(family=label)}")
        response_matrix(spec, data, label)

    return validate


def check_priors(priors, fam) -> None:
    """The logistic-normal family's own priors (``Simplex.own_priors``, ``Priors.validate``): ``sigma``, one positive-support
    prior for the SD of every non-reference category (brms takes one per class ``sigma<k>``: divergence D139), and ``lncor``, an
    ``LKJ`` on the correlation of the log-ratios (brms's class ``lncor``)."""
    from autobayes.priors import _SCALE, Prior, PriorError

    sigma, lncor = priors.sigma, priors.lncor
    if sigma is None:
        raise PriorError(f"the {fam.name} family needs a prior on the SDs of its log-ratios: pass sigma=..., one positive-support prior "
                         "for every sigma<k>, such as ab.HalfStudentT(3, 2.5) (brms's default student_t(3, 0, 2.5))")
    if not isinstance(sigma, Prior):
        raise PriorError(f"prior for 'sigma' must be one Prior, shared by every non-reference category's sigma<k> (brms takes one per "
                         f"class; {DOC}, divergence {fam.simplex.divergence}); got {type(sigma).__name__}")
    if sigma.name not in _SCALE:
        raise PriorError(f"prior for 'sigma' must be a positive-support prior ({sorted(_SCALE)}), got {sigma}")
    if lncor is None:
        raise PriorError(f"the {fam.name} family needs lncor=ab.LKJ(eta), the prior of the correlation of its log-ratios (brms class "
                         "lncor; its default is lkj(1))")
    if not isinstance(lncor, Prior) or lncor.name != "LKJ":
        raise PriorError(f"prior for 'lncor' must be ab.LKJ(eta) (brms's lkj(eta), placed on Llncor as lkj_corr_cholesky), got {lncor!r}")


# -- names ----------------------------------------------------------------------------------------------------------
def sigma_names(categories) -> list[str]:
    """brms's names of the logistic-normal family's SDs, one per non-reference category: ``sigma`` and the category's letters
    and digits (``sigmas2`` for ``s2``), as ``mu<k>`` is named."""
    from autobayes.default_priors import brms_dpar_names

    return ["sigma" + name[2:] for name in brms_dpar_names(categories[1:])]


def names(model) -> dict:
    """PyMC names of a simplex model's own parameters and coordinates: ``"response_dim"`` the coordinate of the K categories of
    the observed response (``<response>_simplex_dim``); the logistic-normal family's ``"sigma"`` (brms's ``sigma<k>``, in
    category order) and ``"L"`` (brms's ``Llncor``, on ``"coordinate"`` x ``"coordinate_T"``, the non-reference categories)."""
    out = {"response_dim": f"{model.spec.response}_simplex_dim"}
    if model.family_spec.simplex.kind == "logistic_normal":
        out.update(sigma=sigma_names(model.categories), L="Llncor", coordinate="Llncor_dim", coordinate_T="Llncor_dim_T")
    return out


def own_point(model, *, sigma, Llncor) -> dict[str, np.ndarray]:
    """``Model.value_point``'s entries of the logistic-normal family's own parameters: ``sigma``, K - 1 positive SDs in category
    order (brms's ``sigma<k>``), each stored as its log, and ``Llncor``, the Cholesky factor of their correlation (brms's
    ``Llncor``), on Stan's ``cholesky_corr`` coordinates."""
    from autobayes.correlation import check_correlation_cholesky, cholesky_corr_free

    roles, M = names(model), len(model.categories) - 1
    if sigma is None or Llncor is None:
        raise ValueError(f"sigma ({M} SDs, one per non-reference category {list(model.categories[1:])}: brms's "
                         f"{', '.join(roles['sigma'])}) and Llncor (the {M} x {M} Cholesky factor of their correlation) are required for "
                         f"the {model.family} family")
    arr = np.atleast_1d(np.asarray(sigma, dtype=float))
    if arr.shape != (M,) or not np.isfinite(arr).all() or (arr <= 0).any():
        raise ValueError(f"sigma must be {M} finite positive SDs, one per non-reference category {list(model.categories[1:])}")
    L = check_correlation_cholesky(Llncor, M, "Llncor")
    out = {f"{s}_log__": np.asarray(np.log(v)) for s, v in zip(roles["sigma"], arr)}
    out[f"{roles['L']}_cholesky_corr__"] = cholesky_corr_free(L)
    return out


# -- the densities, PyTensor ------------------------------------------------------------------------------------------
def dirichlet_logit_logp(value, mu, phi, *_):
    """``log Dirichlet(value | phi softmax(0, mu))`` per row: ``value`` ``(..., K)`` in category order (the reference first),
    ``mu`` ``(..., K - 1)``, ``phi`` positive. ``lgamma(phi) - sum_k lgamma(alpha_k) + sum_k (alpha_k - 1) log y_k`` with
    ``log alpha_k = log phi + mu_k - log_sum_exp(0, mu)`` (brms's ``dirichlet_logit_lpdf``)."""
    import pytensor.tensor as pt

    from autobayes.likelihoods import _lgamma_of_shape, _log_sum_exp

    full = pt.concatenate([pt.zeros_like(mu[..., :1]), mu], axis=-1)
    log_p = full - _log_sum_exp(full)[..., None]
    log_phi = pt.log(phi)
    log_alpha = log_phi[..., None] + log_p
    alpha = pt.exp(log_alpha)
    return (_lgamma_of_shape(phi, log_phi) - pt.sum(_lgamma_of_shape(alpha, log_alpha), axis=-1)
            + pt.sum((alpha - np.float64(1.0)) * pt.log(value), axis=-1))


def _whitened(r, sigma, L):
    """``(diag(sigma) L)^{-1} r`` over the last axis by forward substitution (PyTensor; ``L`` lower triangular, its size fixed)."""
    import pytensor.tensor as pt

    u = r / sigma
    M = int(L.type.shape[-1])
    w = []
    for i in range(M):
        acc = u[..., i]
        for j in range(i):
            acc = acc - L[..., i, j] * w[j]
        w.append(acc / L[..., i, i])
    return pt.stack(w, axis=-1)


def logistic_normal_logp(value, mu, sigma, L, *_):
    """``log`` of the logistic-normal density on the simplex per row: ``MultiNormal(log(y_k / y_ref) | mu, diag(sigma) L L^T
    diag(sigma)) - sum_k log y_k`` over all K parts, ``value`` ``(..., K)`` in category order (the reference first), ``mu`` and
    ``sigma`` ``(..., K - 1)``, ``L`` the ``(K - 1, K - 1)`` Cholesky factor of the correlation matrix (brms's
    ``logistic_normal_cholesky_cor_lpdf``)."""
    import pytensor.tensor as pt

    log_y = pt.log(value)
    r = log_y[..., 1:] - log_y[..., :1] - mu
    w = _whitened(r, sigma, L)
    M = int(L.type.shape[-1])
    log_det = pt.sum(pt.log(sigma), axis=-1) + pt.sum(pt.log(pt.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
    return (np.float64(-M * _HALF_LOG_2PI) - log_det - np.float64(0.5) * pt.sum(w * w, axis=-1)
            - pt.sum(log_y, axis=-1))


# -- the densities and draws, NumPy -------------------------------------------------------------------------------------
def _log_softmax0(eta) -> np.ndarray:
    """``log softmax(0, eta)`` over the last axis: ``(..., K - 1)`` -> ``(..., K)``, the maximum subtracted inside."""
    eta = np.asarray(eta, dtype=float)
    full = np.concatenate([np.zeros(eta.shape[:-1] + (1,)), eta], axis=-1)
    m = full.max(axis=-1, keepdims=True)
    return full - (m + np.log(np.exp(full - m).sum(axis=-1, keepdims=True)))


def _lgamma(x, log_x) -> np.ndarray:
    """``lgamma(x)`` from ``x`` and ``log_x``: SciPy's ``gammaln`` from 1e-10 up, ``-log_x - gamma x`` below (as the graph's)."""
    from scipy.special import gammaln

    x, log_x = np.asarray(x, dtype=float), np.asarray(log_x, dtype=float)
    tiny = log_x < _LOG_TINY
    return np.where(tiny, -log_x - _EULER * np.exp(np.minimum(log_x, _LOG_TINY)), gammaln(np.where(tiny, 1.0, x)))


def dirichlet_log_density(y, eta, phi) -> np.ndarray:
    """NumPy ``dirichlet_logit_logp``: ``y`` ``(n, K)`` in category order, ``eta`` ``(..., n, K - 1)``, ``phi`` broadcastable
    against ``eta``'s leading shape (``(..., 1)``, one per draw); the result ``(..., n)``."""
    log_p = _log_softmax0(eta)
    log_phi = np.log(np.asarray(phi, dtype=float))
    log_alpha = log_phi[..., None] + log_p
    alpha = np.exp(log_alpha)
    return (_lgamma(phi, log_phi) - _lgamma(alpha, log_alpha).sum(axis=-1)
            + ((alpha - 1.0) * np.log(np.asarray(y, dtype=float))).sum(axis=-1))


def logistic_normal_log_density(y, eta, sigma, L) -> np.ndarray:
    """NumPy ``logistic_normal_logp``: ``y`` ``(n, K)`` in category order, ``eta`` ``(..., n, K - 1)``, ``sigma``
    ``(..., K - 1)`` and ``L`` ``(..., K - 1, K - 1)`` per draw (leading shape ``eta``'s without its last two axes); the result
    ``(..., n)``."""
    log_y = np.log(np.asarray(y, dtype=float))
    eta, sigma, L = (np.asarray(a, dtype=float) for a in (eta, sigma, L))
    r = log_y[..., 1:] - log_y[..., :1] - eta
    u = r / sigma[..., None, :]
    M = L.shape[-1]
    w = np.empty_like(u)
    for i in range(M):
        acc = u[..., i].copy()
        for j in range(i):
            acc -= L[..., None, i, j] * w[..., j]
        w[..., i] = acc / L[..., None, i, i]
    log_det = np.log(sigma).sum(axis=-1) + np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(axis=-1)
    return -M * _HALF_LOG_2PI - log_det[..., None] - 0.5 * (w * w).sum(axis=-1) - log_y.sum(axis=-1)


def _softmax_of_logs(log_g) -> np.ndarray:
    m = log_g.max(axis=-1, keepdims=True)
    e = np.exp(log_g - m)
    return e / e.sum(axis=-1, keepdims=True)


def dirichlet_draws(eta, phi, rng) -> np.ndarray:
    """Draws of ``Dirichlet(phi softmax(0, eta))``, ``eta`` ``(..., K - 1)``, ``phi`` broadcastable against ``eta[..., 0]``: the
    normalised Gamma variates, each taken on the log scale as ``log G(alpha + 1) + log(U) / alpha`` (``G(alpha) = G(alpha + 1)
    U^(1 / alpha)``), so that a share whose concentration is tiny is a tiny share rather than a 0 / 0; ``(..., K)``."""
    alpha = np.exp(np.log(np.asarray(phi, dtype=float))[..., None] + _log_softmax0(eta))
    with np.errstate(divide="ignore"):
        log_g = np.log(rng.gamma(alpha + 1.0)) + np.log(rng.uniform(size=alpha.shape)) / alpha
    return _softmax_of_logs(log_g)


def logistic_normal_draws(eta, sigma, L, rng) -> np.ndarray:
    """Draws of the logistic-normal response: ``z = eta + diag(sigma) L e`` with ``e`` standard normal over the non-reference
    categories, then ``softmax(0, z)``; ``eta`` ``(..., n, K - 1)``, ``sigma`` ``(..., K - 1)``, ``L`` ``(..., K - 1, K - 1)``
    (leading shape ``eta``'s without its last two axes); ``(..., n, K)``."""
    eta, sigma, L = (np.asarray(a, dtype=float) for a in (eta, sigma, L))
    e = rng.standard_normal(eta.shape)
    z = eta + sigma[..., None, :] * np.einsum("...ij,...nj->...ni", L, e)
    return np.exp(_log_softmax0(z))


# -- the Bambi families --------------------------------------------------------------------------------------------------
def _bambi_family(model, label: str, params: list[str], links: dict, dist):
    """Bambi's categorical family (B05: every coefficient, SD and standardised effect with a trailing axis over the
    non-reference categories, the design centred, the intercepts uncentred after sampling) with the K - 1 linear predictors as
    the parent ``mu`` under the identity link and the likelihood ``dist``; Bambi's response is the synthetic column, which the
    likelihood does not read (it reads the K columns)."""
    import bambi as bmb
    from bambi.families.univariate import Categorical as BambiCategorical

    others, dim, reference = list(model.categories[1:]), model.names["category_dim"], model.categories[0]

    class SimplexFamily(BambiCategorical):
        SUPPORTED_LINKS = {p: [links[p]] for p in params}

        def get_data(self, response):
            return np.asarray(response.term.data, dtype=float).squeeze()

        def get_levels(self, response):
            return list(others)

        def get_coords(self, response):
            return {dim: list(others)}

        def get_reference(self, response):
            return reference

        @staticmethod
        def transform_backend_eta(eta, kwargs):
            # one row per observation even for an intercept-only formula, (1, K - 1), as Bambi's categorical family makes them
            return np.ones((np.shape(kwargs["observed"])[0], 1)) * eta

    likelihood = bmb.Likelihood(label, params=params, parent="mu", dist=dist)
    return SimplexFamily(label.lower(), likelihood, link=links)


def _scale_rv(name: str, prior):
    """A positive PyMC variable named ``name`` under the positive-support ``prior`` (float64 parameters), sampled on the log
    scale. Must be called inside a PyMC model context."""
    import pymc as pm

    return getattr(pm, prior.name)(name, **{k: np.float64(v) for k, v in prior.params.items()})


def _bambi_setup(model, bambi_priors):
    import pymc as pm
    import pytensor.tensor as pt

    kind, n = model.family_spec.simplex.kind, model.names
    y = np.asarray(model.response_matrix, dtype=np.float64)[:, model.category_order]   # brms's Y in category order
    K = y.shape[1]
    template = np.zeros(K, dtype=np.float64)                       # the size of the response, for the signature
    response_dim, categories = n["response_dim"], list(model.categories)

    def observe(name, *params, logp, random, signature, kwargs):
        pm.modelcontext(None).add_coords({response_dim: categories})
        kwargs.pop("observed", None), kwargs.pop("dims", None)        # Bambi's are the synthetic column's; the response is y
        return pm.CustomDist(name, *params, template, logp=logp, random=random, signature=signature, observed=y,
                             dims=("__obs__", response_dim), **kwargs)

    if kind == "dirichlet":
        class DirichletLogit:
            """``y ~ Dirichlet(phi softmax(0, mu))``."""

            def __new__(cls, name, mu, phi, **kwargs):
                return observe(name, mu, phi, logp=dirichlet_logit_logp, random=_dirichlet_random, signature="(k),(),(m)->(m)",
                               kwargs=kwargs)

        family = _bambi_family(model, "DirichletLogit", ["mu", "phi"], {"mu": "identity", "phi": "log"}, DirichletLogit)
        return family, bambi_priors, model.effective_center_predictors

    from autobayes.correlation import CholeskyCorrTransform, lkj_cholesky_logp, lkj_cholesky_rng

    roles = names(model)
    sigma_prior, eta = model.priors.sigma, float(model.priors.lncor.params["eta"])
    M = K - 1

    class LogisticNormal:
        """``log(y[k] / y[ref]) ~ MultiNormal(mu, diag(sigma) Lncor diag(sigma))`` with the Jacobian on the simplex."""

        def __new__(cls, name, mu, **kwargs):
            model_ = pm.modelcontext(None)
            model_.add_coords({roles["coordinate"]: categories[1:], roles["coordinate_T"]: categories[1:]})
            sigmas = [_scale_rv(s, sigma_prior) for s in roles["sigma"]]
            L = pm.CustomDist(roles["L"], np.float64(eta), logp=lambda value, e: lkj_cholesky_logp(value, eta, M),
                              random=lambda e, rng=None, size=None: lkj_cholesky_rng(eta, M, rng, size),
                              signature=f"()->({M},{M})", default_transform=CholeskyCorrTransform(M),
                              dims=(roles["coordinate"], roles["coordinate_T"]))
            return observe(name, mu, pt.stack(sigmas), L, logp=logistic_normal_logp, random=_logistic_normal_random,
                           signature="(k),(k),(k,k),(m)->(m)", kwargs=kwargs)

    family = _bambi_family(model, "LogisticNormal", ["mu"], {"mu": "identity"}, LogisticNormal)
    return family, bambi_priors, model.effective_center_predictors


def _dirichlet_random(mu, phi, template, rng=None, size=None):
    rng = rng if rng is not None else np.random.default_rng()
    mu = np.asarray(mu, dtype=float)
    shape = tuple(size) if size is not None else mu.shape[:-1]
    return dirichlet_draws(np.broadcast_to(mu, shape + mu.shape[-1:]), np.broadcast_to(phi, shape), rng)


def _logistic_normal_random(mu, sigma, L, template, rng=None, size=None):
    rng = rng if rng is not None else np.random.default_rng()
    mu, sigma, L = (np.asarray(a, dtype=float) for a in (mu, sigma, L))
    shape = tuple(size) if size is not None else mu.shape[:-1]
    mu = np.broadcast_to(mu, shape + mu.shape[-1:])
    e = rng.standard_normal(mu.shape)
    z = mu + np.broadcast_to(sigma, mu.shape) * np.einsum("...ij,...j->...i", np.broadcast_to(L, shape + L.shape[-2:]), e)
    return np.exp(_log_softmax0(z))


# -- predictions, log_lik ----------------------------------------------------------------------------------------------
def _softmax(eta):
    """brms's ``posterior_epred`` of the dirichlet family: ``softmax(0, mu)`` over the categories, ``(..., K - 1)`` ->
    ``(..., K)`` (the categorical family's)."""
    from autobayes.likelihoods import categorical_probabilities

    return categorical_probabilities(eta)


def _refuse_mean(eta):
    raise ValueError("the expected response of the logistic_normal family is not offered: the mean of a logistic-normal vector has "
                     "no closed form, and brms refuses posterior_epred for the family too (\"Cannot compute expected values of the "
                     "posterior predictive distribution for family 'logistic_normal'.\")")


def logistic_normal_draws_of(model, post) -> tuple[np.ndarray, np.ndarray]:
    """The posterior's ``sigma<k>`` stacked in category order, ``(chain, draw, K - 1)``, and ``Llncor``, ``(chain, draw, K - 1,
    K - 1)``."""
    roles = names(model)
    sigma = np.stack([post[s].to_numpy() for s in roles["sigma"]], axis=-1)
    L = post[roles["L"]].transpose("chain", "draw", roles["coordinate"], roles["coordinate_T"]).to_numpy()
    return sigma, L


def _draw_dirichlet(model, post, eta, mean, df, rng):
    return dirichlet_draws(eta, post[model.names["phi"]].to_numpy()[..., None], rng)


def _draw_logistic_normal(model, post, eta, mean, df, rng):
    sigma, L = logistic_normal_draws_of(model, post)
    return logistic_normal_draws(eta, sigma, L, rng)


def _response_in_category_order(model) -> np.ndarray:
    return np.asarray(model.design.y, dtype=float)[:, model.category_order]


def _log_lik_dirichlet(model, post, eta):
    """brms's log_lik: ``log Dirichlet(y_i | phi softmax(0, mu_i))``; ``eta`` ``(chain, draw, row, K - 1)``."""
    return dirichlet_log_density(_response_in_category_order(model), eta, post[model.names["phi"]].to_numpy()[..., None])


def _log_lik_logistic_normal(model, post, eta):
    """brms's log_lik: each row's logistic-normal density on the simplex, its Jacobian included."""
    sigma, L = logistic_normal_draws_of(model, post)
    return logistic_normal_log_density(_response_in_category_order(model), eta, sigma, L)


# -- formulas ----------------------------------------------------------------------------------------------------------
def _check_formula(name: str, divergence: str):
    def check(spec) -> None:
        for t in spec.group_terms:
            what = None
            if t.slopes:
                what = f"a varying slope ('{t.canonical}')"
            elif t.id is not None:
                what = f"a correlation ID ('{t.canonical}')"
            elif t.custom:
                what = f"the group term '{t.canonical}' (mm() or gr(by = ))"
            if what is not None:
                raise FormulaError(f"{spec.formula!r}: {what} is not offered for the {name} family: each non-reference category would "
                                   f"get it in its own linear predictor, which no frozen case verifies; the family takes varying "
                                   f"intercepts (1 | g) ({DOC}, divergence {divergence})")
        if spec.offset is not None:
            raise FormulaError(f"{spec.formula!r}: an offset term is not offered for the {name} family (brms adds it to every "
                               f"non-reference category's linear predictor; no frozen case verifies it; {DOC}, divergence {divergence})")

    return check


# -- describe() ----------------------------------------------------------------------------------------------------------
def response_lines(model) -> list[str]:
    """``describe()``'s data line of the response: its columns, categories and reference."""
    cols = ", ".join(model.spec.response_columns)
    ref, others = model.categories[0], model.categories[1:]
    from autobayes.default_priors import brms_dpar_names

    return [f"  response  : cbind({cols}), {len(model.categories)} proportions per row summing to 1 (brms's matrix Y, in the columns' "
            f"order); categories {', '.join(model.categories)}, the reference {ref!r}"
            + (" (refcat)" if ref != model.spec.response_columns[0] else " (the first column)")
            + f"; the non-reference categories' linear predictors are brms's {', '.join(brms_dpar_names(others))}"]


def _mean_lines(kind: str):
    def lines(model) -> list[str]:
        ref, others = model.categories[0], model.categories[1:]
        out = [f"  One such linear predictor mu_i[k] per non-reference category k = {', '.join(others)}, each with its own intercept,",
               f"  coefficients and group effects (a_c[k], b[k], u[., k]); the reference category {ref!r} has none."]
        if kind == "dirichlet":
            out += [f"  E[y_i] = softmax(0, mu_i) (brms's posterior_epred): each mu_i[k] is the log of E[y_i[k]] / E[y_i[{ref}]]."]
        else:
            out += [f"  mu_i[k] is the mean of log(y_i[k] / y_i[{ref}]); E[y_i] has no closed form (brms refuses posterior_epred)."]
        return out

    return lines


def _likelihood_lines(kind: str):
    def lines(model) -> list[str]:
        from autobayes.priors import describe_prior

        K = len(model.categories)
        if kind == "dirichlet":
            return [scale_line(model), "", "Likelihood",
                    "  y_i ~ Dirichlet(phi * softmax(0, mu_i)): brms's dirichlet_logit_lpdf, lgamma(phi) - sum_k lgamma(alpha_ik) +",
                    "  sum_k (alpha_ik - 1) log y_ik with log alpha_ik = log phi + mu_i[k] - log_sum_exp(0, mu_i) (alpha in category",
                    f"  order, the reference first); phi is the precision, sampled as log(phi). The {K} columns are proportions strictly",
                    "  inside (0, 1) whose rows sum to 1."]
        roles = names(model)
        sig = ", ".join(roles["sigma"])
        return [f"  {sig} ~ {describe_prior(model.priors.sigma, lb=0)}, each   (brms classes {sig}; one prior for all)",
                f"  Lncor ~ {model.priors.lncor} (brms lkj({model.priors.lncor.params['eta']:g}) on the correlation matrix of the log-ratios, "
                "placed on Llncor as lkj_corr_cholesky)",
                "", "Likelihood",
                f"  z_i = log(y_i[k] / y_i[{model.categories[0]}]), k = {', '.join(model.categories[1:])}: z_i ~ MultiNormal(mu_i, "
                "diag(sigma) Lncor diag(sigma)), Lncor = Llncor Llncor^T;",
                "  the density of y_i on the simplex adds the Jacobian -sum_k log y_ik over all "
                f"{K} parts (brms's logistic_normal_cholesky_cor_lpdf). The",
                f"  {K} columns are proportions strictly inside (0, 1) whose rows sum to 1."]

    return lines


# -- the families ----------------------------------------------------------------------------------------------------
def _gamma_default(scale):
    from autobayes.priors import Gamma

    return Gamma(0.01, 0.01)                                         # brms's gamma(0.01, 0.01) on phi, as for the Beta family


DIRICHLET = FamilySpec(
    name="dirichlet", tag="Dirichlet", parent="mu", link="logit", link_scale="logit", mean_name="p",
    response_kind="continuous", response_dtype=None,
    scale_parameter="phi", bambi_scale_parameter="phi", fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=lambda: None,                                       # built per model by bambi_setup
    validate_response=_validate_response("dirichlet"),
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: None,
    inverse_link=_softmax,
    expected_response=None,
    draw_response=_draw_dirichlet,
    manual_prediction=True, prediction_kinds=STANDARD_PREDICTION_KINDS,
    kind_error="kind must be 'mean' (the expected proportions softmax(0, mu)), 'response' (draws on the simplex) or 'linear' (mu)",
    scale_rule="phi (the precision) is required for the dirichlet family; sigma and shape do not apply",
    scale_missing_error=("the dirichlet family needs a prior on its precision phi: pass phi=..., a positive-support prior such as "
                         "ab.Gamma(0.01, 0.01) (brms's default)"),
    linpred_symbol="mu", scale_phrase="log-ratio of the mean proportions (against the reference category)",
    intercept_meaning="log of the mean proportion of category k over the reference category's",
    mean_lines=_mean_lines("dirichlet"),
    likelihood_lines=_likelihood_lines("dirichlet"),
    centred_intercept_note=("  The centred intercepts a_c[k] are at the proportion-weighted predictor, not the average over levels.",),
    deterministic_note="mu = the linear predictors of the non-reference categories; softmax(0, mu) is computed by predict()",
    default_prior_link=None,                                         # brms's Intercept default is student_t(3, 0, 2.5) per category
    default_scale_prior=_gamma_default,
    per_category=True,
    bambi_setup=_bambi_setup,
    check_formula=_check_formula("dirichlet", "D138"),
    log_likelihood=_log_lik_dirichlet,
    simplex=DIRICHLET_PARTS,
)

LOGISTIC_NORMAL = FamilySpec(
    name="logistic_normal", tag="logistic-normal", parent="mu", link="identity", link_scale="log-ratio", mean_name="mean",
    response_kind="continuous", response_dtype=None,
    scale_parameter=None, bambi_scale_parameter=None, fixed_parameters=NO_FIXED_PARAMETERS,
    bambi_family=lambda: None,
    validate_response=_validate_response("logistic_normal"),
    response_warnings=lambda spec, df: [],
    response_upper_bound=lambda model: None,
    inverse_link=_refuse_mean,
    expected_response=None,
    draw_response=_draw_logistic_normal,
    manual_prediction=True, prediction_kinds=frozenset({"response", "linear"}),
    kind_error=("kind must be 'response' (draws on the simplex) or 'linear' (mu, brms's posterior_linpred); kind='mean' is not "
                "offered: the mean of a logistic-normal vector has no closed form, and brms refuses posterior_epred for the "
                "logistic_normal family too"),
    scale_rule=("the logistic_normal family's SDs are given as sigma (one per non-reference category) with its correlation's "
                "Cholesky factor Llncor; shape and phi do not apply"),
    scale_missing_error=None,
    linpred_symbol="mu", scale_phrase="log-ratio (against the reference category)",
    intercept_meaning="mean log-ratio of category k over the reference category",
    mean_lines=_mean_lines("logistic_normal"),
    likelihood_lines=_likelihood_lines("logistic_normal"),
    centred_intercept_note=("  The centred intercepts a_c[k] are at the proportion-weighted predictor, not the average over levels.",),
    deterministic_note="mu = the linear predictors (mean log-ratios) of the non-reference categories",
    default_prior_link=None,                                         # brms's Intercept default is student_t(3, 0, 2.5) per category
    default_scale_prior=None,
    per_category=True,
    bambi_setup=_bambi_setup,
    check_formula=_check_formula("logistic_normal", "D139"),
    log_likelihood=_log_lik_logistic_normal,
    simplex=LOGISTIC_NORMAL_PARTS,
)
