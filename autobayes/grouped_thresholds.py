"""Grouped thresholds of the cumulative family: brms's ``thres(gr = z)`` and ``thres(K, gr = z)`` (brms-parity F06b,
docs/grouped-thresholds.md).

``ord | thres(gr = z) ~ x + (1 | g)`` gives every level of ``z`` its own ordered thresholds: as many as the level's largest
category less one (brms's ``nthres``, one entry per level), or ``K`` for every level under ``thres(K, gr = z)``. brms declares
one ``ordered[nthres[k]] Intercept_k`` per level ``k`` (its ``merged_Intercept`` is their concatenation, and each row reads the
block its ``Jthres`` points to), puts the intercept prior on every threshold and, unlike the ungrouped family, does not centre
the design: ``Intercept_k`` is its ``b_Intercept_k`` and the linear predictor is the raw design times ``b`` plus the group
effects. A row of level ``l`` has ``P(y <= k) = expit(tau_l[k] - mu)`` for ``k = 1..nthres_l`` and ``P(y <= nthres_l + 1) = 1``,
so a category above ``nthres_l + 1`` has probability exactly 0 there (brms's ``posterior_epred``), and its log density is
brms's ``ordered_logistic_lpmf`` with the level's thresholds.

This library samples the levels' thresholds as one vector ``Intercept`` (brms's ``merged_Intercept``, the levels in sorted
order, each level's block in threshold order; dims ``Intercept_dim``, coordinates ``"<level>[<k>]"``) on Stan's ordered
coordinates within each block (``BlockOrdered``: ``(tau_l[1], log(tau_l[2] - tau_l[1]), ...)`` per level, the log Jacobian
the sum of the log gaps), under the intercept prior on every entry (``grouped_ordered_thresholds``); ``b_Intercept`` is the
same vector, the deterministic the likelihood reads (``grouped_cumulative_logit_family``: each row's thresholds are its
level's block, padded to the widest block with the block's last threshold, which no row selects, and its own number of
thresholds bounds the categories its stable row ``grouped_ordered_logistic_logp`` takes).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pymc.logprob.transforms import Transform

from autobayes.formula import _IDENT, AdditionTerm, FormulaError

DIVERGENCE = "D158"                     # what this slice does not offer (docs/brms-parity/divergences.md)
PREDICTION_DIVERGENCE = "D159"          # what its predictions do not offer
ROW_LEVEL = "__thres_level__"           # the posterior variable predict() hands the families: each row's level (0-based)
_FORMALS = ("x", "gr")                  # brms's thres(x, gr): the number of thresholds, then the grouping column
_DOC = "docs/grouped-thresholds.md"


# -- the term ------------------------------------------------------------------------------------------------------------
def parse_thres(args: list[str], piece: str, formula: str) -> AdditionTerm:
    """brms's ``thres(x, gr)`` on the left-hand side, its arguments matched as R matches them (by name, then by position):
    ``thres(gr = z)``, ``thres(3, gr = z)``, ``thres(3, z)``, ``thres(x = 3, gr = z)``. ``gr`` is a column name and ``x`` a
    positive whole number, the same for every level; ``thres(K)`` without ``gr`` and a number per level are refused."""
    named, positional = {}, []
    for arg in args:
        m = re.fullmatch(rf"({_IDENT})\s*=\s*(.+)", arg, re.S)
        if m is None:
            positional.append(arg)
            continue
        if m.group(1) not in _FORMALS or m.group(1) in named:
            raise FormulaError(f"{formula!r}: {piece!r}: thres() takes the number of thresholds x and the grouping column gr, each "
                               f"at most once (brms's thres(x, gr)); got {m.group(1)!r}")
        named[m.group(1)] = m.group(2).strip()
    for name in _FORMALS:
        if name not in named and positional:
            named[name] = positional.pop(0)
    if positional:
        raise FormulaError(f"{formula!r}: {piece!r}: thres() takes at most two arguments, the number of thresholds x and the "
                           "grouping column gr (brms's thres(x, gr))")
    if "gr" not in named:
        raise FormulaError(f"{formula!r}: {piece!r}: thres() without gr = , one number of thresholds for every row, is not offered; "
                           f"grouped thresholds thres(gr = z) and thres(K, gr = z) are ({_DOC}, divergence {DIVERGENCE})")
    gr, x = named["gr"], named.get("x")
    if not re.fullmatch(_IDENT, gr):
        raise FormulaError(f"{formula!r}: {piece!r}: thres()'s gr must be a column name, got {gr!r}")
    count = None
    if x is not None:
        if not re.fullmatch(r"[0-9]+", x) or int(x) < 1:
            raise FormulaError(f"{formula!r}: {piece!r}: thres()'s number of thresholds must be one positive whole number for every "
                               f"level of {gr!r} here, got {x!r}; a number per level or per row is not offered (brms refuses a "
                               "named vector: 'Number of thresholds should be unique for each group.'; "
                               f"{_DOC}, divergence {DIVERGENCE})")
        count = int(x)
    return AdditionTerm("thres", column=gr, count=count)


def check_thres(spec, formula: str) -> None:
    """The terms that combine with ``thres(gr = z)`` (F06b): population-level terms with design columns, the offset and group
    terms; ``z`` is read by ``thres()`` alone. Everything else is refused naming the construct (divergence D158)."""
    a = spec.addition("thres")
    others = ([f"the addition term {b.term!r}" for b in spec.additions if b.kind != "thres"]
              + [f"the term {t.name!r}" for t in spec.pop_terms if t.is_special]          # mo(), me(), mi(), cs()
              + [f"the Gaussian process {g.term!r}" for g in spec.gp_terms]
              + [f"the term '{t.canonical}'" for t in spec.group_terms if t.custom or t.id is not None]
              + ([f"the autocorrelation term {spec.autocor.term!r}"] if spec.autocor is not None else [])
              + ([f"the SAR term {spec.sar.term!r}"] if spec.sar is not None else [])
              + [f"the distributional formula '{d.formula}'" for d in spec.dpars])
    if others:
        raise FormulaError(f"{formula!r}: {a.term!r} next to {others[0]} is not offered: no frozen brms case verifies the pair "
                           f"({_DOC}, divergence {DIVERGENCE})")
    read = ([spec.response] + list(spec.pop_columns) + [c for t in spec.group_terms for c in t.data_columns]
            + ([spec.trials] if spec.trials else []) + ([spec.offset.column] if spec.offset is not None else []))
    if a.column in read:
        raise FormulaError(f"{formula!r}: {a.column!r} is the grouping column of {a.term!r} and cannot also be the response, a "
                           f"predictor, a grouping factor or the offset ({_DOC}, divergence {DIVERGENCE})")


def level_labels(column: str, s: pd.Series, where: str = "column") -> np.ndarray:
    """The text labels of ``thres(gr = column)`` on a column: text or a pandas Categorical, as ``str`` (brms's factor-like
    variable); a numeric column is refused as brms refuses it, a boolean one because no frozen case verifies brms's
    FALSE / TRUE levels."""
    from autobayes.formula import reject_complex

    reject_complex(f"{where} {column!r}", s)
    if pd.api.types.is_bool_dtype(s):
        raise FormulaError(f"{where} {column!r} of thres(gr = {column}) is boolean; brms reads it as a factor with the levels FALSE "
                           f"and TRUE, which no frozen case verifies here: recode it as text ({_DOC}, divergence {DIVERGENCE})")
    if pd.api.types.is_numeric_dtype(s):
        raise FormulaError(f"{where} {column!r} of thres(gr = {column}) has dtype {s.dtype}; brms refuses a numeric grouping column "
                           "too (\"Variable 'gr' in 'thres' needs to be factor-like.\"): recode it as text")
    return s.astype(str).to_numpy()


def threshold_coding(a: AdditionTerm, spec, data: pd.DataFrame) -> AdditionTerm:
    """The ``thres()`` term with its levels (the column's labels on the fitted rows, sorted as the grouping factors' are) and
    each level's number of thresholds: its largest category less one (brms's rule for ``thres(gr = z)``), or ``K`` for
    ``thres(K, gr = z)``. Refused as brms refuses them: a level whose rows all take the first category, and a ``K`` below the
    response's largest category less one; refused here (D158): a response category above every level's largest
    (``thres(gr = z)``, an ordered Categorical's unused top category) or above ``K + 1``, whose expected values brms would
    give over another number of categories than the response's."""
    from autobayes.formula import ordinal_categories

    labels = level_labels(a.column, data[a.column])
    categories, codes, _ = ordinal_categories(spec.response, data[spec.response])
    levels = tuple(sorted(set(labels.tolist())))
    largest = {level: int(codes[labels == level].max()) for level in levels}
    if a.count is None:
        flat = [level for level in levels if largest[level] < 2]
        if flat:
            raise FormulaError(f"{spec.formula!r}: {a.term!r}: the rows of level(s) {flat} of {a.column!r} all take the first category "
                               f"{categories[0]!r}, which leaves them no threshold (brms: 'Could not extract the number of thresholds.')")
        nthres = tuple(largest[level] - 1 for level in levels)
        if max(nthres) + 1 != len(categories):
            raise FormulaError(f"{spec.formula!r}: {a.term!r}: no level of {a.column!r} reaches the response's categories "
                               f"{list(categories[max(nthres) + 1:])}; brms gives each level its largest category less one threshold, "
                               f"which is not verified here with categories above every level's ({_DOC}, divergence {DIVERGENCE})")
    else:
        if a.count + 1 < int(codes.max()):
            raise FormulaError(f"{spec.formula!r}: {a.term!r}: {a.count} thresholds leave no place for the response's category "
                               f"{categories[int(codes.max()) - 1]!r} (brms: 'Number of thresholds is smaller than required by the "
                               "response.')")
        if a.count + 1 != len(categories):
            raise FormulaError(f"{spec.formula!r}: {a.term!r}: {a.count} thresholds make {a.count + 1} categories, more than the "
                               f"response's {len(categories)}; brms's expected values would then cover categories the response does not "
                               f"have, which is not verified here ({_DOC}, divergence {DIVERGENCE})")
        nthres = (a.count,) * len(levels)
    return replace(a, levels=levels, nthres=nthres)


@dataclass(frozen=True, eq=False)
class ThresholdGroups:
    """The levels of ``thres(gr = z)`` on the fitted rows (``Design.thresholds``): ``term`` as the canonical formula writes it,
    ``column`` (z), ``levels`` (sorted, brms's group order for a text column), ``nthres`` (each level's number of thresholds)
    and ``index``, each fitted row's level (0-based). The merged vector of thresholds (brms's ``merged_Intercept``) holds the
    levels' blocks in this order."""

    term: str
    column: str
    levels: tuple[str, ...]
    nthres: tuple[int, ...]
    index: np.ndarray

    @classmethod
    def of(cls, a: AdditionTerm, frame: pd.DataFrame) -> "ThresholdGroups":
        groups = cls(a.term, a.column, tuple(a.levels), tuple(int(m) for m in a.nthres), np.zeros(0, dtype=np.int64))
        return replace(groups, index=groups.level_index(frame[a.column]))

    @property
    def starts(self) -> tuple[int, ...]:
        """Where each level's block starts in the merged vector (0-based)."""
        return tuple(int(s) for s in np.cumsum((0,) + self.nthres[:-1]))

    @property
    def total(self) -> int:
        return int(sum(self.nthres))

    @property
    def width(self) -> int:
        """The widest block: the number of thresholds of the level with the most."""
        return int(max(self.nthres))

    @property
    def labels(self) -> tuple[str, ...]:
        """The coordinates of the merged vector: ``"<level>[<k>]"``, ``k = 1..nthres`` of each level in turn."""
        return tuple(f"{level}[{k}]" for level, m in zip(self.levels, self.nthres) for k in range(1, m + 1))

    def jthres(self, index: np.ndarray | None = None) -> np.ndarray:
        """brms's ``Jthres``: each row's first and last threshold in the merged vector, 1-based, ``(rows, 2)``."""
        index = self.index if index is None else index
        starts, nthres = np.asarray(self.starts), np.asarray(self.nthres)
        return np.column_stack([starts[index] + 1, starts[index] + nthres[index]])

    def level_index(self, s: pd.Series, where: str = "column") -> np.ndarray:
        """Each row's level (0-based) on a column of labels; a level the fitted rows do not have is refused (divergence D159)."""
        labels = level_labels(self.column, s, where)
        unseen = sorted(set(labels.tolist()) - set(self.levels))
        if unseen:
            raise FormulaError(f"{where} {self.column!r} has the level(s) {unseen}, which the fitted rows of {self.term!r} do not have; "
                               f"they have no thresholds (fitted levels {list(self.levels)}; {_DOC}, divergence {PREDICTION_DIVERGENCE})")
        position = {level: k for k, level in enumerate(self.levels)}
        return np.array([position[label] for label in labels], dtype=np.int64)

    def blocks(self, tau) -> dict[str, np.ndarray]:
        """Each level's thresholds from merged draws ``(..., total)`` (NumPy), by level."""
        tau = np.asarray(tau, dtype=float)
        return {level: tau[..., s:s + m] for level, s, m in zip(self.levels, self.starts, self.nthres)}

    def rows(self, tau, index: np.ndarray) -> np.ndarray:
        """Each row's thresholds from merged draws ``(..., total)`` (NumPy): ``(..., rows, width)``, its level's block padded with
        the block's last threshold."""
        tau = np.asarray(tau, dtype=float)
        table = np.stack([np.concatenate([tau[..., s:s + m], np.repeat(tau[..., s + m - 1:s + m], self.width - m, axis=-1)], axis=-1)
                          for s, m in zip(self.starts, self.nthres)], axis=-2)
        return table[..., index, :]


# -- the graph -----------------------------------------------------------------------------------------------------------
class BlockOrdered(Transform):
    """Stan's ``ordered`` transform within each block of a vector (brms's ``ordered[nthres[k]] Intercept_k``, one block per
    level): ``backward`` gives each block ``(u_1, u_1 + exp(u_2), ...)``, ``forward`` its ``(x_1, log(x_2 - x_1), ...)``, and the
    log Jacobian is the sum of the unconstrained entries that are not a block's first. Written with constant masks and
    matrices (every entry's exponential taken where it is used, never in an untaken branch), so that it is finite with a
    finite gradient on the PyTensor path and under JAX."""

    name = "ordered"

    def __init__(self, nthres):
        nthres = [int(m) for m in nthres]
        total = sum(nthres)
        first = np.zeros(total)
        cumulative = np.zeros((total, total))        # x @ cumulative: each block's running sum
        previous = np.zeros((total, total))          # x @ previous: each entry's predecessor in its block (0 for a first)
        start = 0
        for m in nthres:
            first[start] = 1.0
            for i in range(start, start + m):
                cumulative[start:i + 1, i] = 1.0
                if i > start:
                    previous[i - 1, i] = 1.0
            start += m
        self.first, self.cumulative, self.previous = first, cumulative, previous

    def backward(self, value, *inputs):
        rest = 1.0 - self.first
        steps = self.first * value + rest * pt.exp(rest * value)
        return pt.tensordot(steps, self.cumulative, axes=[[steps.ndim - 1], [0]])

    def forward(self, value, *inputs):
        gaps = value - pt.tensordot(value, self.previous, axes=[[value.ndim - 1], [0]])
        return self.first * value + (1.0 - self.first) * pt.log(pt.switch(pt.eq(self.first, 1.0), 1.0, gaps))

    def log_jac_det(self, value, *inputs):
        return pt.sum((1.0 - self.first) * value, axis=-1)


def grouped_ordered_thresholds(name, prior_name: str, params: dict, groups: ThresholdGroups, dim: str):
    """brms's ``ordered[nthres[k]] Intercept_k`` of every level as one PyMC variable ``name`` (brms's ``merged_Intercept``; dims
    ``dim``, coordinates ``groups.labels``) on Stan's ordered coordinates within each level's block (``BlockOrdered``), with the
    intercept prior on every threshold (``Priors(intercept=...)``, brms's class ``Intercept``): its density is the prior's summed
    over the thresholds, as brms adds it, and its forward draws the prior's independent draws sorted within each block (that
    density's distribution on the ordered region). Must be called inside a PyMC model context."""
    from autobayes.likelihoods import _LOCATION_DRAWS

    total, names = groups.total, list(params)
    values = [np.zeros(total)] + [np.broadcast_to(np.asarray(params[k], dtype=np.float64), (total,)).copy() for k in names]
    base = getattr(pm, prior_name)

    def logp(value, anchor, *ps):
        return pt.sum(pm.logp(base.dist(**dict(zip(names, ps))), value), axis=-1)

    def random(anchor, *ps, rng=None, size=None):
        if prior_name not in _LOCATION_DRAWS:
            raise NotImplementedError(f"a {prior_name} prior on the thresholds has no forward draws")
        core = np.broadcast_shapes(np.shape(anchor), *(np.shape(p) for p in ps))
        shape = (tuple(size) + core[-1:]) if size is not None else core
        out = _LOCATION_DRAWS[prior_name](rng, shape, *ps)
        for s, m in zip(groups.starts, groups.nthres):
            out[..., s:s + m] = np.sort(out[..., s:s + m], axis=-1)
        return out

    location = float(params.get("mu", 0.0))
    start = np.concatenate([location + np.arange(m) - (m - 1) / 2.0 for m in groups.nthres])   # ordered within each block

    def support_point(rv, size, anchor, *ps):
        return pt.broadcast_to(pt.as_tensor(start), rv.shape)

    model = pm.modelcontext(None)
    if dim not in model.coords:
        model.add_coord(dim, list(groups.labels))
    return pm.CustomDist(name, *values, logp=logp, random=random, support_point=support_point,
                         signature=",".join(["(k)"] * len(values)) + "->(k)", transform=BlockOrdered(groups.nthres), dims=dim)


def threshold_table(tau, groups: ThresholdGroups):
    """The levels' thresholds as a ``(levels, width)`` PyTensor matrix from the merged vector ``tau``: each level's block, padded
    with its last threshold (an entry no row of the level selects)."""
    parts = []
    for s, m in zip(groups.starts, groups.nthres):
        block = tau[s:s + m]
        parts.append(block if m == groups.width else pt.concatenate([block, pt.repeat(tau[s + m - 1:s + m], groups.width - m)]))
    return pt.stack(parts)


def grouped_ordered_logistic_logp(value, eta, thresholds, nthres):
    """``log P(y = value | eta, thresholds)`` of brms's ``cumulative("logit")`` with a number of thresholds per row
    (``nthres``; the row's thresholds are the first ``nthres`` entries of ``thresholds``): the stable row of
    ``likelihoods.ordered_logistic_logp``, ``-softplus(eta - tau_y) - softplus(tau_{y-1} - eta) + log(1 - exp(tau_{y-1} -
    tau_y))``, without the first term in the row's last category ``nthres + 1`` and the second in category 1; ``-inf`` outside
    ``1..nthres + 1``. Each row's thresholds are selected by the one-hot switch of ``likelihoods._select`` at an index clipped
    into the row's range, so an untaken branch stays finite."""
    from autobayes.likelihoods import _select, stable_softplus

    y, k = pt.cast(value, "int64"), pt.cast(nthres, "int64")
    thr = thresholds + pt.zeros_like(eta)[..., None]
    upper = _select(thr, pt.clip(y - 1, 0, k - 1))
    lower = _select(thr, pt.clip(y - 2, 0, k - 1))
    has_upper, has_lower = pt.le(y, k), pt.ge(y, 2)
    both = pt.and_(has_upper, has_lower)
    out = (pt.switch(has_upper, -stable_softplus(eta - upper), 0.0)
           + pt.switch(has_lower, -stable_softplus(lower - eta), 0.0)
           + pt.switch(both, pt.log1mexp(pt.switch(both, lower - upper, -1.0)), 0.0))
    return pt.switch(pt.and_(pt.ge(y, 1), pt.le(y, k + 1)), out, -np.inf)


def grouped_cumulative_probabilities(eta, thresholds, nthres, categories: int | None = None) -> np.ndarray:
    """``P(y = k)``, ``k = 1..categories``, with a number of thresholds per row (NumPy): ``eta`` ``(...)``, ``thresholds`` ``(...,
    width)`` (the row's first ``nthres`` entries are its thresholds), ``nthres`` broadcasting against ``eta``. A row's categories
    ``1..nthres + 1`` take ``likelihoods.cumulative_probabilities`` of its thresholds and every category above them exactly 0
    (brms's ``posterior_epred``); ``categories`` defaults to ``width + 1``."""
    from autobayes.likelihoods import cumulative_probabilities

    eta = np.asarray(eta, dtype=float)
    thr = np.broadcast_to(np.asarray(thresholds, dtype=float), eta.shape + np.shape(thresholds)[-1:])
    k = np.broadcast_to(np.asarray(nthres, dtype=np.int64), eta.shape)
    out = np.zeros(eta.shape + ((thr.shape[-1] + 1) if categories is None else categories,))
    for m in np.unique(k):
        sel = k == m
        out[sel, :m + 1] = cumulative_probabilities(eta[sel], thr[sel][..., :m])
    return out


def grouped_ordered_logistic_random(eta, thresholds, nthres, rng=None, size=None):
    from autobayes.likelihoods import draw_categories

    return draw_categories(grouped_cumulative_probabilities(eta, thresholds, nthres), rng, size)


def grouped_cumulative_logit_family(groups: ThresholdGroups, dim: str):
    """Bambi family for brms's ``cumulative("logit")`` with ``thres(gr = z)``: the linear predictor ``mu`` (no thresholds, the
    design uncentred) as the parent under the identity link and the merged thresholds ``Intercept``
    (``grouped_ordered_thresholds``) as an auxiliary parameter; ``b_Intercept`` is that vector as a deterministic (brms does
    not centre the design with grouped thresholds, so ``b_Intercept_k`` is ``Intercept_k``), and each fitted row reads its
    level's block (``threshold_table``) and its own number of thresholds in ``grouped_ordered_logistic_logp``."""
    import bambi as bmb

    index = np.asarray(groups.index, dtype=np.int64)
    nthres = np.asarray(groups.nthres, dtype=np.int64)[index]

    class GroupedCumulativeLogit:
        """``y ~ OrderedLogistic(mu, b_Intercept[level])``, ``b_Intercept = Intercept``."""

        def __new__(cls, name, mu, Intercept, **kwargs):
            thresholds = pm.Deterministic("b_Intercept", Intercept, dims=dim)
            rows = threshold_table(thresholds, groups)[index]
            return pm.CustomDist(name, mu, rows, nthres, logp=grouped_ordered_logistic_logp, random=grouped_ordered_logistic_random,
                                 signature="(),(k),()->()", dtype="int64", **kwargs)

    class GroupedCumulativeLogitFamily(bmb.Family):
        SUPPORTED_LINKS = {"mu": ["identity"], "Intercept": ["identity"]}

    likelihood = bmb.Likelihood("GroupedCumulativeLogit", params=["mu", "Intercept"], parent="mu", dist=GroupedCumulativeLogit)
    return GroupedCumulativeLogitFamily("grouped_cumulative_logit", likelihood, link={"mu": "identity", "Intercept": "identity"})


# -- the family's functions (``families.cumulative._grouped``) -----------------------------------------------------------
def bambi_setup(model, bambi_priors):
    """Bambi is handed the formula with formulae's intercept made to add nothing (``likelihoods.no_intercept``), as for the
    cumulative family, the design uncentred (brms does not centre it with grouped thresholds), and the merged thresholds
    ``Intercept`` as the family's auxiliary parameter under the intercept prior."""
    import bambi as bmb

    from autobayes.likelihoods import no_intercept

    groups, prior, dim = model._threshold_groups, model.priors.intercept, model.names["threshold_dim"]

    def thresholds(name, **kwargs):
        return grouped_ordered_thresholds(name, prior.name, dict(prior.params), groups, dim)

    terms = {k: v for k, v in bambi_priors.items() if k != "Intercept"}
    priors = {"mu": {**terms, "Intercept": bmb.Prior("NoIntercept", dist=no_intercept)},
              "Intercept": bmb.Prior("GroupedOrderedThresholds", dist=thresholds)}
    return grouped_cumulative_logit_family(groups, dim), priors, False


def with_row_levels(model, post, df: pd.DataFrame):
    """The posterior with each row's level of ``df`` (the fitted frame, or checked new rows) under ``ROW_LEVEL``, which the
    family's expected response reads (``predict()`` hands it the rows it predicts)."""
    groups = model.design.thresholds
    return post.assign({ROW_LEVEL: (("__row__",), groups.level_index(df[groups.column], where="newdata column"))})


def _row_thresholds(model, post, n: int):
    """``(chain, draw, rows, width)`` thresholds and ``(rows,)`` numbers of thresholds of the rows ``post`` names (``ROW_LEVEL``),
    or of the ``n`` fitted rows."""
    groups, names = model.design.thresholds, model.names
    if ROW_LEVEL in post:
        index = np.asarray(post[ROW_LEVEL].values, dtype=np.int64)
    elif n == model.design.n:
        index = groups.index
    else:
        raise ValueError(f"{n} rows but no level of {groups.column!r} for them: predict() hands each row's level ({ROW_LEVEL})")
    tau = post[names["thresholds"]].transpose("chain", "draw", names["threshold_dim"]).to_numpy()
    return groups.rows(tau, index), np.asarray(groups.nthres, dtype=np.int64)[index]


def expected_response(model, post, eta):
    """brms's ``posterior_epred``: ``P(y = k)`` per draw, row and category, each row from its level's thresholds and exactly 0
    above its level's ``nthres + 1`` categories."""
    rows, nthres = _row_thresholds(model, post, np.shape(eta)[-1])
    return grouped_cumulative_probabilities(eta, rows, nthres, len(model.categories))


def log_likelihood(model, post, eta):
    """brms's log_lik: ``log P(y_i | mu_i, b_Intercept[level_i])`` of each fitted row (``pointwise.ordered_logistic`` on the
    rows of each number of thresholds)."""
    from autobayes.families.pointwise import ordered_logistic

    eta = np.asarray(eta, dtype=float)
    rows, nthres = _row_thresholds(model, post, eta.shape[-1])
    y, out = np.asarray(model.design.y), np.empty(eta.shape)
    for m in np.unique(nthres):
        sel = nthres == m
        out[..., sel] = ordered_logistic(y[sel], eta[..., sel], rows[..., sel, :m])
    return out


def mean_lines(model) -> list[str]:
    from autobayes.families.cumulative import _categories

    g = model.design.thresholds
    per = ", ".join(f"{level} {m}" for level, m in zip(g.levels, g.nthres))
    return [f"  P(y_i <= k) = expit(b_Intercept[{g.column}_i, k] - mu_i), k = 1..nthres[{g.column}_i], for the {len(model.categories)} "
            f"ordered categories {_categories(model)}:",
            f"  each level of {g.column} has its own ordered thresholds ({g.term}; thresholds per level: {per}), and a row's",
            "  categories above nthres[level] + 1 have probability exactly 0 (brms's posterior_epred). P(y_i = k) is the difference",
            "  of consecutive ones. mu has no intercept: the thresholds take its place, and a larger mu moves probability towards",
            "  the higher categories."]


def intercept_lines(model) -> list[str]:
    g = model.design.thresholds
    return [f"  Predictors are not centred: brms does not centre the design when the thresholds are grouped ({g.term}), so the",
            f"  sampled thresholds Intercept[l, k] are those of the raw design and b_Intercept equals them; one ordered vector per level",
            f"  l of {g.column} (brms's Intercept_1, ..., Intercept_{len(g.levels)}, in the levels' order {', '.join(g.levels)}), held",
            "  together as one vector (brms's merged_Intercept)."]


def likelihood_lines(model) -> list[str]:
    g = model.design.thresholds
    return ["", "Likelihood",
            f"  y_i ~ OrderedLogistic(mu_i, b_Intercept[{g.column}_i]): brms's ordered_logistic_lpmf (cumulative(\"logit\"), disc = 1)",
            f"  with the thresholds of row i's level of {g.column}, written as -softplus(mu_i - tau[y]) - softplus(tau[y - 1] - mu_i) +",
            "  log(1 - exp(-(tau[y] - tau[y - 1]))) with the end terms dropped for the first category and the level's last one:",
            "  finite with a finite gradient for every finite mu (autobayes/grouped_thresholds.py). The response is fitted as its",
            "  1-based category (brms's Y)."]


def data_lines(model) -> list[str]:
    """``describe()``'s data line of the term."""
    g = model.design.thresholds
    counts = ", ".join(f"{level} {int((g.index == k).sum())} row(s), {m} threshold(s)" for k, (level, m) in enumerate(zip(g.levels, g.nthres)))
    return [f"  thresholds: {g.term}: one ordered set of thresholds per level of {g.column} ({counts}); predict() needs {g.column} on "
            "every row and refuses a level the fitted rows do not have"]


def prior_line(model) -> str:
    from autobayes.priors import describe_prior

    g = model.design.thresholds
    return (f"  Intercept[l, k] ~ {describe_prior(model.priors.intercept)}, for each of the {g.total} thresholds of the {len(g.levels)} "
            f"levels of {g.column} (brms's class Intercept)")


def value_line(model) -> tuple[str, str]:
    """``describe()``'s line of the sampled thresholds."""
    g, n = model.design.thresholds, model.names
    return (f"{n['intercept']}_ordered__",
            f"Intercept[{', '.join(g.levels)}], the thresholds of each level of {g.column} ({g.total} entries; dims {n['threshold_dim']}, "
            "coordinates level[k]), on Stan's ordered coordinates within each level (Intercept[l, 1], log(Intercept[l, k] - "
            "Intercept[l, k - 1]))")


def value_point(model, intercept) -> np.ndarray:
    """``Model.value_point``'s thresholds: ``intercept`` a mapping over exactly the levels, each its ``nthres`` thresholds,
    finite and strictly increasing (brms's ``Intercept_k``), stored on the block-ordered coordinates."""
    from collections.abc import Mapping

    g = model.design.thresholds
    if not isinstance(intercept, Mapping) or set(intercept) != set(g.levels):
        raise ValueError(f"intercept must map each level of {g.column!r} {list(g.levels)} to its thresholds ({g.term})")
    out = []
    for level, m in zip(g.levels, g.nthres):
        arr = np.atleast_1d(np.asarray(intercept[level], dtype=float))
        if arr.shape != (m,) or not np.isfinite(arr).all() or not (np.diff(arr) > 0).all():
            raise ValueError(f"intercept[{level!r}] must be the {m} thresholds of level {level!r}, finite and strictly increasing")
        out.append(np.r_[arr[0], np.log(np.diff(arr))])
    return np.concatenate(out)
