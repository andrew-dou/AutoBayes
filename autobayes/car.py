"""Spatial conditional autoregressive terms: brms's ``car(M, gr, type)`` (F02a, docs/spatial-car.md).

The rows belong to locations, the labels of the column ``gr``; ``M`` is a 0/1 adjacency matrix of the locations, labelled by
location on both axes, handed to the model in ``data2`` under its name (brms's ``data2 = list(W = W)``). The term adds one
spatial effect per location to the linear predictor, ``rcar[gr_i]``, under one of brms's four structures, with ``D`` the
diagonal matrix of the numbers of neighbours (brms's ``Nneigh``), ``W`` the adjacency over the fitted locations, ``N`` their
number (brms's ``Nloc``) and ``tau = 1 / sdcar^2``:

- ``escar``, the exact sparse proper CAR: ``rcar ~ MVN(0, (tau (D - car W))^-1)`` with ``car`` in (0, 1), its log density
  ``(N log tau + sum_k log(1 - car lambda_k) - tau (rcar' D rcar - car rcar' W rcar)) / 2`` (Jin, Carlin and Banerjee 2005;
  the sparse evaluation of Joseph's Stan case study "Exact sparse CAR models in Stan", 2016), ``lambda`` the eigenvalues of
  ``D^-1/2 W D^-1/2`` (brms's ``eigenMcar``), without the constants ``log det D`` and ``N log(2 pi) / 2``, as brms writes it;
- ``esicar``, the exact sparse intrinsic CAR (``car = 1``): ``rcar`` the free ``zcar`` (``N - 1`` entries) with
  ``rcar[N] = -sum(zcar)``, log density ``((N - 1) log tau - tau (rcar' D rcar - rcar' W rcar)) / 2``;
- ``icar``, the intrinsic CAR in the pairwise-difference form of Morris et al. (2019, "Bayesian hierarchical spatial models:
  implementing the Besag York Mollie model in stan", Spatial and Spatio-temporal Epidemiology 31): ``rcar = sdcar * zcar``,
  ``-sum_{i~j} (zcar_i - zcar_j)^2 / 2`` over the edges plus the soft sum-to-zero constraint
  ``normal_lpdf(sum(zcar) | 0, 0.001 N)``;
- ``bym2``, Riebler et al.'s (2016, "An intuitive Bayesian spatial model for disease mapping that accounts for scaling",
  Statistical Methods in Medical Research 25) convolution: ``rcar = sdcar (sqrt(1 - rhocar) nszcar + sqrt(rhocar / s)
  zcar)``, ``zcar`` under the ICAR density above, ``nszcar ~ Normal(0, 1)``, ``rhocar`` in (0, 1), and ``s`` the scaling
  factor (brms's ``car_scale``, ``bym2_scale``).

The log densities and the transforms were checked against brms 2.23's ``rstan::log_prob`` on the frozen cases (constants
included); nothing here is taken from brms's code.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from autobayes.formula import _IDENT, FormulaError, _split_top_level, categorical_labels

TYPES = ("escar", "esicar", "icar", "bym2")
_CALL_RE = re.compile(r"^car\s*\((.*)\)$", re.S)
_FORMALS = ("M", "gr", "type")
SUM_TO_ZERO_SD = 0.001                  # brms's soft sum-to-zero constraint: sum(zcar) ~ normal(0, 0.001 * Nloc)
_CLASSES = {"escar": ("sdcar", "car"), "esicar": ("sdcar",), "icar": ("sdcar",), "bym2": ("sdcar", "rhocar")}
_LATENT = {"escar": ("rcar",), "esicar": ("zcar",), "icar": ("zcar",), "bym2": ("zcar", "nszcar")}
_MEANING = {"escar": "exact sparse CAR", "esicar": "exact sparse intrinsic CAR", "icar": "intrinsic CAR (pairwise differences)",
            "bym2": "BYM2 convolution (Riebler et al. 2016)"}


@dataclass(frozen=True)
class Car:
    """A formula's spatial CAR term, brms's ``car(M, gr = gr, type = type)``: ``matrix`` names the adjacency matrix in
    ``data2``, ``gr`` the column of locations, ``type`` one of ``TYPES``. It is not a population-level term: it has no design
    column, but its own parameters (``classes``, with priors of brms's classes ``sdcar``, ``car`` and ``rhocar``) and latent
    vectors (``latent``)."""

    matrix: str
    gr: str
    type: str

    @property
    def term(self) -> str:
        """The term as the canonical formula writes it: ``car(W, gr = site, type = "escar")``."""
        return f'car({self.matrix}, gr = {self.gr}, type = "{self.type}")'

    @property
    def columns(self) -> tuple[str, ...]:
        return (self.gr,)

    @property
    def classes(self) -> tuple[str, ...]:
        """brms's prior classes of the term's parameters: ``sdcar``, and ``car`` (escar) or ``rhocar`` (bym2)."""
        return _CLASSES[self.type]

    @property
    def latent(self) -> tuple[str, ...]:
        """brms's latent vectors: ``rcar`` (escar), ``zcar`` (esicar, icar), ``zcar`` and ``nszcar`` (bym2)."""
        return _LATENT[self.type]

    @property
    def meaning(self) -> str:
        return _MEANING[self.type]


def parse_car(piece: str, formula: str) -> Car:
    """brms's ``car(M, gr = NA, type = "escar")`` as written in ``formula``, with R's argument rules (named arguments first,
    then the positional ones in order). Refused, naming the construct: an argument the function does not have, ``M`` or ``gr``
    that is not one name, ``gr = NA`` (brms deprecates it: "Please use argument 'gr' even if each observation represents its own
    location"; divergence D141) and a ``type`` brms does not have."""
    m = _CALL_RE.match(piece)
    if m is None:
        raise FormulaError(f"{piece!r}: a CAR term is written car(M, gr = <column>, type = \"...\") (docs/spatial-car.md)")
    named, positional = {}, []
    for arg in (_split_top_level(m.group(1).strip(), ",") if m.group(1).strip() else []):
        km = re.fullmatch(rf"({_IDENT})\s*=(?!=)\s*(.+)", arg.strip(), re.S)
        if km is None:
            positional.append(arg.strip())
        elif km.group(1) not in _FORMALS:
            raise FormulaError(f"{piece!r}: car() has no argument {km.group(1)!r}; brms's car() takes M, gr, type")
        elif km.group(1) in named:
            raise FormulaError(f"{piece!r}: the argument {km.group(1)!r} is given twice")
        else:
            named[km.group(1)] = km.group(2).strip()
    rest = [f for f in _FORMALS if f not in named]
    if len(positional) > len(rest):
        raise FormulaError(f"{piece!r}: too many arguments; brms's car() takes M, gr, type")
    named.update(zip(rest, positional))
    if not re.fullmatch(_IDENT, named.get("M", "")):
        raise FormulaError(f"{piece!r}: car() needs M, the name of the adjacency matrix the model receives in data2 "
                           "(car(W, gr = site) with data2={'W': ...})")
    gr = named.get("gr", "NA")
    if gr == "NA":
        raise FormulaError(f"{piece!r}: car() without gr (each row its own location) is not offered; brms deprecates it too "
                           "('Please use argument 'gr' even if each observation represents its own location'): name the column of "
                           "locations, car(M, gr = site) (docs/spatial-car.md, divergence D141)")
    if not re.fullmatch(_IDENT, gr):
        raise FormulaError(f"{piece!r}: gr = {gr}: gr must be one column name, the column of locations (docs/spatial-car.md, "
                           "divergence D141)")
    kind = named.get("type", '"escar"')
    tm = re.fullmatch(r"""(["'])(\w+)\1""", kind)
    if tm is None or tm.group(2) not in TYPES:
        raise FormulaError(f"{piece!r}: type = {kind}: type must be one of {', '.join(repr(t) for t in TYPES)} (brms's rule too)")
    return Car(matrix=named["M"], gr=gr, type=tm.group(2))


def check_car(term: Car, piece: str, *, response: str, trials: str | None, fam, others: list[str], formula: str) -> None:
    """The refusals of a CAR term that depend on the rest of the formula (divergence D141): its column of locations may not be
    the response or the trials column; the family must take the term (``FamilySpec.car``: the Gaussian and Poisson families,
    verified against frozen brms cases); ``others`` are the formula's constructs no frozen case verifies next to it (the
    addition terms, the special terms ``mo()``, ``gp()``, ``me()``, ``mi()``, ``cs()``, the group terms ``mm()`` and ``gr(by =
    )``, an autocorrelation term), each named as it reads."""
    from autobayes.families import FAMILIES

    if term.gr in (response, trials):
        raise FormulaError(f"{piece!r}: the {'response' if term.gr == response else 'trials column'} {term.gr!r} cannot be the "
                           "column of locations of a car() term")
    if not fam.car:
        offering = [name for name, other in FAMILIES.items() if other.car]
        raise FormulaError(f"{formula!r}: the CAR term {term.term!r} is not available for the {fam.name} family; brms's car() is "
                           f"verified here for the {' and '.join(offering)} families (docs/spatial-car.md, divergence D141)")
    if others:
        raise FormulaError(f"{formula!r}: the CAR term {term.term!r} next to {others[0]} is not supported in this slice: no frozen "
                           "brms case verifies the pair (docs/spatial-car.md, divergence D141)")


# -- the locations of the fitted rows -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class CarDesign:
    """A CAR term on the fitted rows (F02a). ``locations`` are the sorted labels of the locations the fitted rows have
    (brms's ``Nloc`` of them, in brms's level order of a column of text), ``index`` (n,) each fitted row's location (brms's
    ``Jloc - 1``), ``W`` the 0/1 adjacency of those locations, ``edges`` (E, 2) its edges as brms lists them (``edges1 - 1``,
    ``edges2 - 1``: the lower triangle, column by column), ``neighbours`` (Nloc,) the numbers of neighbours (brms's
    ``Nneigh``), ``eigenvalues`` those of ``D^-1/2 W D^-1/2`` in decreasing order (brms's ``eigenMcar``; escar and esicar) and
    ``scale`` the BYM2 scaling factor (brms's ``car_scale``; bym2)."""

    term: str
    type: str
    locations: tuple[str, ...]
    index: np.ndarray
    W: np.ndarray
    edges: np.ndarray
    neighbours: np.ndarray
    eigenvalues: np.ndarray | None = None
    scale: float | None = None

    @property
    def Nloc(self) -> int:
        return len(self.locations)


def adjacency(term: Car, data2) -> tuple[pd.DataFrame, list[str]]:
    """The adjacency matrix ``term.matrix`` of ``data2``, labelled by location (text) on both axes, with a warning when entries
    other than 0 and 1 were read as 1 (brms: "Converting all non-zero values in 'M' to 1."). Refused: a ``data2`` without it, a
    matrix that is not a labelled pandas DataFrame (brms: "Row names are required for 'M' in CAR terms."), whose columns are
    neither its row labels nor positions, with repeated labels, non-finite entries, not symmetric (brms's rule too) or with a
    non-zero diagonal (brms counts it as an edge; divergence D141)."""
    name = term.matrix
    if not isinstance(data2, Mapping) or name not in data2:
        raise FormulaError(f"{term.term!r} reads the adjacency matrix {name!r} from data2: pass data2={{{name!r}: <a pandas DataFrame "
                           "labelled by location on both axes>} (brms's data2 = list(...))")
    M = data2[name]
    if not isinstance(M, pd.DataFrame) or M.index.equals(pd.RangeIndex(len(M))) or M.shape[0] != M.shape[1]:
        raise FormulaError(f"data2[{name!r}] must be a square pandas DataFrame whose index labels the locations of {term.term!r} "
                           "(brms: 'Row names are required for 'M' in CAR terms.')")
    rows, _ = categorical_labels(f"the row labels of data2[{name!r}]", pd.Series(list(M.index)))
    if M.columns.equals(pd.RangeIndex(M.shape[1])):         # the columns by position
        cols = rows
    else:
        cols, _ = categorical_labels(f"the column labels of data2[{name!r}]", pd.Series(list(M.columns)))
    if len(set(rows.tolist())) != len(rows) or cols.tolist() != rows.tolist():
        raise FormulaError(f"data2[{name!r}]: the locations must label the rows once each, and the columns in the same order")
    values = M.to_numpy()
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values) or not np.isfinite(values.astype(float)).all():
        raise FormulaError(f"data2[{name!r}] must hold finite real numbers")
    values = values.astype(float)
    if not np.array_equal(values, values.T):
        raise FormulaError(f"data2[{name!r}] must be symmetric (brms: ''M' for CAR terms must be symmetric.')")
    if np.any(np.diag(values) != 0):
        raise FormulaError(f"data2[{name!r}] has a non-zero diagonal: a location is not its own neighbour here (brms counts the "
                           "diagonal entry as an edge; docs/spatial-car.md, divergence D141)")
    notes = []
    if not np.isin(values, (0.0, 1.0)).all():
        notes.append(f"data2[{name!r}]: entries other than 0 and 1 are read as 1, an edge (brms: 'Converting all non-zero values in "
                     "'M' to 1.')")
    return pd.DataFrame((values != 0).astype(float), index=rows, columns=rows), notes


def car_layout(term: Car, df: pd.DataFrame, data2) -> tuple[CarDesign, list[str]]:
    """The CAR term ``term`` on the rows of ``df`` (the fitted frame) with the adjacency matrix of ``data2``, as brms's Stan data
    have it, and the warnings to record. Each row's location must be one of the matrix's (brms: "Row names of 'M' for CAR
    terms do not match the names of the grouping levels."); locations the rows lack are dropped from the graph, as brms drops
    them, with a warning. Refused (divergence D141): fewer than two locations, a location without a neighbour among the fitted
    ones (brms refuses it for escar and esicar, and accepts it for icar and bym2, whose densities then leave it unconstrained),
    and a graph of several connected components (one sum-to-zero constraint for all of them, and brms's BYM2 scaling of such a
    graph is meaningless; divergence D140)."""
    M, notes = adjacency(term, data2)
    labels, _ = categorical_labels(f"location column {term.gr!r} of {term.term!r}", df[term.gr])
    unknown = sorted(set(labels.tolist()) - set(M.index))
    if unknown:
        raise FormulaError(f"{term.term!r}: the locations {unknown[:5]} of {term.gr!r} are not labels of data2[{term.matrix!r}] (brms: "
                           "'Row names of 'M' for CAR terms do not match the names of the grouping levels.')")
    locations = tuple(sorted(set(labels.tolist())))
    dropped = [loc for loc in M.index if loc not in set(locations)]
    if dropped:
        notes.append(f"{term.term!r}: the locations {dropped[:5]}{' ...' if len(dropped) > 5 else ''} of data2[{term.matrix!r}] have no "
                     "fitted row and are dropped from the graph, with their edges (brms drops them too)")
    W = M.loc[list(locations), list(locations)].to_numpy()
    neighbours = W.sum(1)
    if len(locations) < 2:
        raise FormulaError(f"{term.term!r}: the fitted rows have {len(locations)} location; a CAR term needs at least two")
    lonely = [loc for loc, k in zip(locations, neighbours) if k == 0]
    if lonely:
        raise FormulaError(f"{term.term!r}: the locations {lonely[:5]} have no neighbour among the fitted locations (brms refuses "
                           "them for escar and esicar and accepts them for icar and bym2, leaving them unconstrained; "
                           "docs/spatial-car.md, divergence D141)")
    parts = components(W)
    if parts > 1:
        raise FormulaError(f"{term.term!r}: the graph of data2[{term.matrix!r}] on the fitted locations has {parts} connected "
                           "components; one connected graph is offered (brms puts one sum-to-zero constraint on all of them and "
                           "scales a BYM2 term over the whole graph; docs/spatial-car.md, divergences D140, D141)")
    col, row = np.nonzero(np.tril(W, -1).T)                  # brms's edges: the lower triangle, column by column
    index = np.array([locations.index(v) for v in labels], dtype=np.int64)
    eigenvalues = scale = None
    if term.type in ("escar", "esicar"):
        d = 1.0 / np.sqrt(neighbours)
        eigenvalues = np.linalg.eigvalsh(d[:, None] * W * d[None, :])[::-1]
    if term.type == "bym2":
        scale = bym2_scale(W)
    design = CarDesign(term=term.term, type=term.type, locations=locations, index=index, W=W,
                       edges=np.stack([row, col], axis=1).astype(np.int64), neighbours=neighbours, eigenvalues=eigenvalues,
                       scale=scale)
    return design, notes


def components(W: np.ndarray) -> int:
    """The number of connected components of the graph of the adjacency matrix ``W``."""
    seen, count = np.zeros(len(W), dtype=bool), 0
    for start in range(len(W)):
        if seen[start]:
            continue
        count += 1
        stack = [start]
        seen[start] = True
        while stack:
            for j in np.flatnonzero(W[stack.pop()]):
                if not seen[j]:
                    seen[j] = True
                    stack.append(j)
    return count


def bym2_scale(W: np.ndarray) -> float:
    """The BYM2 scaling factor of a connected graph, brms's ``car_scale``: the geometric mean of the marginal variances of the
    ICAR precision ``Q = D - W`` under the sum-to-zero constraint (Riebler et al. 2016), computed as INLA's ``inla.scale.model``
    and the Stan case study of Morris et al. (2019) compute it and as brms 2.23's value on the frozen graph shows (divergence
    D140): the inverse of ``Q + eps I``, ``eps = max(diag Q) sqrt(machine epsilon)``, conditioned on ``sum = 0``. In exact
    arithmetic that is ``sum_k v_k v_k' / (lambda_k + eps)`` over the eigenpairs of ``Q`` but the null vector, evaluated here
    from the eigendecomposition, without the cancellation the inverse of the nearly singular matrix suffers (brms's value is
    within 1.7e-10 of it on the frozen 5 by 5 grid, and 7.7e-8 relative below the exact geometric mean of ``Q``'s generalized
    inverse)."""
    Q = np.diag(W.sum(1)) - W
    eps = float(np.max(np.diag(Q))) * math.sqrt(np.finfo(float).eps)
    lam, V = np.linalg.eigh(Q)
    marginal = (V[:, 1:] ** 2) @ (1.0 / (lam[1:] + eps))
    return float(np.exp(np.mean(np.log(marginal))))


def car_values(design: CarDesign, values: dict) -> np.ndarray:
    """``rcar`` (..., Nloc) from the term's parameters (NumPy; brms's ``rcar``): ``values`` holds the type's latent vectors
    (..., length) and parameters (...), by brms's names."""
    if design.type == "escar":
        return np.asarray(values["rcar"], dtype=float)
    z = np.asarray(values["zcar"], dtype=float)
    if design.type == "esicar":
        return np.concatenate([z, -z.sum(-1, keepdims=True)], axis=-1)
    sd = np.asarray(values["sdcar"], dtype=float)[..., None]
    if design.type == "icar":
        return sd * z
    rho = np.asarray(values["rhocar"], dtype=float)[..., None]
    return sd * (np.sqrt(1.0 - rho) * np.asarray(values["nszcar"], dtype=float) + np.sqrt(rho / design.scale) * z)


# -- the model ---------------------------------------------------------------------------------------------------------
def _parameter(name: str, prior, kind: str):
    """``sdcar`` under its positive-support prior, or ``car`` / ``rhocar`` under ``Beta`` or ``Flat()`` (the uniform density on
    (0, 1), brms's flat prior on a ``real<lower=0,upper=1>``), parameters as float64. Inside a PyMC model context."""
    import pymc as pm

    if kind == "unit" and prior.name == "Flat":
        return pm.Uniform(name, lower=np.float64(0.0), upper=np.float64(1.0))
    return getattr(pm, prior.name)(name, **{k: np.float64(v) for k, v in prior.params.items()})


def car_effect(eta, term: Car, design: CarDesign, priors: dict, names: dict):
    """The linear predictor ``eta`` plus ``rcar`` at each row's location, building the term inside the PyMC model Bambi is
    building (``likelihoods.monotonic_family``'s hook): ``sdcar`` (and ``car`` or ``rhocar``) under their priors, the latent
    vectors on the locations (``names["coordinate"]``; esicar's ``zcar`` on all but the last, ``names["zcar_coordinate"]``),
    flat but for ``nszcar``'s Normal(0, 1), the CAR log density as a potential (``names["lpdf"]``) and ``rcar`` (a
    deterministic but for escar, whose ``rcar`` is the free vector)."""
    import pymc as pm
    import pytensor.tensor as pt

    model = pm.modelcontext(None)
    model.add_coord(names["coordinate"], design.locations)
    if term.type == "esicar":
        model.add_coord(names["zcar_coordinate"], design.locations[:-1])
    sdcar = _parameter(names["sdcar"], priors["sdcar"], "positive")
    i, j = design.edges[:, 0], design.edges[:, 1]
    N, half = np.float64(design.Nloc), np.float64(0.5)
    if term.type in ("escar", "esicar"):
        if term.type == "escar":
            alpha = _parameter(names["car"], priors["car"], "unit")
            rcar = pm.Flat(names["rcar"], dims=names["coordinate"])
        else:
            alpha = np.float64(1.0)
            zcar = pm.Flat(names["zcar"], dims=names["zcar_coordinate"])
            rcar = pm.Deterministic(names["rcar"], pt.concatenate([zcar, -pt.sum(zcar, keepdims=True)]), dims=names["coordinate"])
        log_tau = np.float64(-2.0) * pt.log(sdcar)
        quad = pt.sum(design.neighbours * pt.sqr(rcar)) - np.float64(2.0) * alpha * pt.sum(rcar[i] * rcar[j])
        if term.type == "escar":
            lpdf = half * (N * log_tau + pt.sum(pt.log1p(-alpha * design.eigenvalues)) - pt.exp(log_tau) * quad)
        else:
            lpdf = half * ((N - np.float64(1.0)) * log_tau - pt.exp(log_tau) * quad)
    else:
        zcar = pm.Flat(names["zcar"], dims=names["coordinate"])
        sd_sum = np.float64(SUM_TO_ZERO_SD) * N
        lpdf = (-half * pt.sum(pt.sqr(zcar[i] - zcar[j])) - half * pt.sqr(pt.sum(zcar) / sd_sum)
                - np.float64(math.log(SUM_TO_ZERO_SD * design.Nloc) + 0.5 * math.log(2.0 * math.pi)))
        if term.type == "icar":
            effect = sdcar * zcar
        else:
            rho = _parameter(names["rhocar"], priors["rhocar"], "unit")
            nszcar = pm.Normal(names["nszcar"], mu=np.float64(0.0), sigma=np.float64(1.0), dims=names["coordinate"])
            effect = sdcar * (pt.sqrt(np.float64(1.0) - rho) * nszcar + pt.sqrt(rho / np.float64(design.scale)) * zcar)
        rcar = pm.Deterministic(names["rcar"], effect, dims=names["coordinate"])
    pm.Potential(names["lpdf"], lpdf)
    return eta + rcar[design.index]


def roles(term: Car | None) -> dict[str, str]:
    """PyMC names of a CAR term (``Model.car_names``): its parameters and latent vectors under brms's names (``sdcar``,
    ``car``, ``rhocar``, ``rcar``, ``zcar``, ``nszcar``, those the type has), ``"coordinate"`` the locations' dimension
    (``rcar_dim``), esicar's ``"zcar_coordinate"`` (``zcar_dim``, all locations but the last) and ``"lpdf"`` the potential that
    carries the CAR log density (``rcar_lpdf`` or ``zcar_lpdf``). Empty without one."""
    if term is None:
        return {}
    out = {name: name for name in (*term.classes, *term.latent, "rcar")}
    out.update(coordinate="rcar_dim", lpdf=f"{term.latent[0]}_lpdf")
    if term.type == "esicar":
        out["zcar_coordinate"] = "zcar_dim"
    return out


def value_entries(model, car) -> dict[str, np.ndarray]:
    """``Model.value_point``'s entries of a CAR term (F02a): ``car`` maps exactly the term's parameters and latent vectors, by
    brms's names, to their values: ``sdcar`` positive (stored as its log), ``car`` and ``rhocar`` in (0, 1) (stored as their
    logits, the PyMC coordinate of their prior: interval or log-odds), the latent vectors finite, one entry per location (esicar's
    ``zcar`` one fewer), stored as given. ``{}`` for a model without the term, which refuses a ``car`` given."""
    term = getattr(model.spec, "car", None)
    if term is None:
        if car is not None:
            raise ValueError("car holds a CAR term's values; this model has no car() term")
        return {}
    wanted = (*term.classes, *term.latent)
    if not isinstance(car, Mapping) or set(car) != set(wanted):
        raise ValueError(f"car must map exactly {list(wanted)} to their values (the term {term.term!r}), got "
                         f"{sorted(car) if isinstance(car, Mapping) else type(car).__name__}")
    pmm, names, N = model.pymc_model, model.car_names, model.design.car.Nloc
    point = {}
    for name in wanted:
        value = np.asarray(car[name], dtype=float)
        size = N - 1 if (term.type, name) == ("esicar", "zcar") else N
        if name in term.classes:                              # sdcar positive, car and rhocar in (0, 1)
            ok = value.shape == () and np.isfinite(value) and value > 0 and (name == "sdcar" or value < 1)
        else:
            ok = value.shape == (size,) and np.isfinite(value).all()
        if not ok:
            raise ValueError(f"car[{name!r}] must be " + ({"sdcar": "a positive finite number", "car": "a number in (0, 1)",
                                                           "rhocar": "a number in (0, 1)"}.get(name) or f"{size} finite numbers"))
        if name == "sdcar":
            point[f"{names[name]}_log__"] = np.log(value)
        elif name in term.classes:
            point[pmm.rvs_to_values[pmm[names[name]]].name] = np.log(value) - np.log1p(-value)
        else:
            point[names[name]] = value
    return point


def location_index(design: CarDesign, column: pd.Series, term: Car) -> np.ndarray:
    """Each row's 0-based location (brms's ``Jloc - 1``) from its label in ``column``; a location the model was not fitted
    on is refused, as brms refuses it ("Cannot handle new locations in CAR models.")."""
    labels, _ = categorical_labels(f"newdata column {term.gr!r}", column)
    lookup = {loc: k for k, loc in enumerate(design.locations)}
    unknown = sorted(set(labels.tolist()) - set(lookup))
    if unknown:
        raise FormulaError(f"newdata has locations of {term.gr!r} the model was not fitted on: {unknown[:5]}; {term.term!r} predicts at "
                           "the fitted locations only (brms: 'Cannot handle new locations in CAR models.')")
    return np.array([lookup[v] for v in labels], dtype=np.int64)


def check_graph(model) -> None:
    """The CAR term read off the graph (F02a): at the graph's initial point, where every latent vector is 0 and so is ``rcar``,
    and at that point with the latent vectors moved, the graph's ``rcar`` is ``car_values`` of the moved point's parameters, and
    the parent's linear predictor (its log under a log link) moves by ``rcar`` at each row's location."""
    term, pmm, names, design = model.spec.car, model.pymc_model, model.car_names, model.design.car
    parent = model.family_spec.parent
    outputs = [pmm[parent], pmm[names["rcar"]], *(pmm[names[c]] for c in term.classes)]
    fn = pmm.compile_fn(pmm.replace_rvs_by_values(outputs), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore",
                        mode="FAST_COMPILE")
    start = {k: np.asarray(v, dtype=float) for k, v in pmm.initial_point().items()}
    moved = {**start, **{names[v]: np.cos(np.arange(1.0, start[names[v]].size + 1.0)) for v in term.latent}}
    low, (high, rcar, *parameters) = fn(start)[0], fn(moved)
    expected = car_values(design, {**{v: moved[names[v]] for v in term.latent}, **dict(zip(term.classes, parameters))})
    shift = np.asarray(high, dtype=float) - low
    if model.bambi_model.family.link[parent].name == "log":
        shift = np.log(np.asarray(high, dtype=float)) - np.log(low)
    if not (np.allclose(rcar, expected, rtol=1e-10, atol=1e-12) and np.allclose(shift, expected[design.index], rtol=1e-10, atol=1e-10)):
        raise RuntimeError(f"the graph's {parent!r} does not add {term.term!r} as car.car_values computes it")


def describe(model, part: str, lp: str = "eta"):
    """``describe()``'s lines of a CAR term (F02a): ``part`` is ``"data"`` (its locations and graph), ``"term"`` (its effect and
    density), ``"priors"`` or ``"roles"`` (the value variables' descriptions, a dict); empty without the term."""
    term = getattr(model.spec, "car", None)
    if term is None:
        return {} if part == "roles" else []
    d, names = model.design.car, model.car_names
    if part == "data":
        return [f"  car       : {term.term}: {d.Nloc} locations (the labels of {term.gr}, sorted) and {len(d.edges)} edges of "
                f"data2[{term.matrix!r}] on them, {int(d.neighbours.min())} to {int(d.neighbours.max())} neighbours each (brms's Nloc, "
                "Jloc, edges1, edges2, Nneigh)" + (f"; BYM2 scaling factor car_scale = {d.scale:.10g}" if d.scale is not None else "")]
    if part == "term":
        icar = (f"-sum over the edges (zcar_i - zcar_j)^2 / 2 plus log Normal(sum(zcar) | 0, {SUM_TO_ZERO_SD:g} Nloc), brms's soft "
                "sum-to-zero constraint")
        if term.type == "escar":
            density = ("(Nloc log tau + sum_k log(1 - car lambda_k) - tau (rcar' D rcar - car rcar' W rcar)) / 2, lambda the eigenvalues "
                       "of D^-1/2 W D^-1/2 (brms's eigenMcar); car in (0, 1) is the spatial dependence")
        elif term.type == "esicar":
            density = "((Nloc - 1) log tau - tau rcar' (D - W) rcar) / 2, rcar = (zcar, -sum(zcar)) summing to zero"
        elif term.type == "icar":
            density = f"rcar = sdcar * zcar, and zcar's {icar}"
        else:
            density = (f"rcar = sdcar (sqrt(1 - rhocar) nszcar + sqrt(rhocar / {d.scale:.10g}) zcar), zcar's {icar}, nszcar ~ Normal(0, "
                       "1); rhocar in (0, 1) is the spatial share of the variance")
        return [f"  rcar[l]: brms's {term.term}, the {term.meaning}, one effect per location added to {lp}_i at the row's location;",
                f"  D holds the numbers of neighbours, W the 0/1 adjacency, tau = 1 / sdcar^2; log density: {density}.",
                f"  The CAR log density is the potential {names['lpdf']} (in brms's model block, not its lprior); new locations are refused."]
    if part == "priors":
        what = {"sdcar": ("the SD of the CAR effects", 0, None), "car": ("the spatial dependence", 0, 1),
                "rhocar": ("the spatial share", 0, 1)}
        from autobayes.priors import describe_prior

        return [f"  {names[c]} ~ {describe_prior(p, lb=what[c][1], ub=what[c][2])}   (brms class {c}: {what[c][0]}"
                + ("; Flat() is the uniform on (0, 1)" if p.name == "Flat" else "") + ")"
                for c, p in model.priors.car_priors(model.spec).items()]
    pmm = model.pymc_model
    out = {f"{names['sdcar']}_log__": "log(sdcar), the SD of the CAR effects (brms's sdcar)"}
    out.update({pmm.rvs_to_values[pmm[names[c]]].name: f"logit({c}) (brms's {c}, in (0, 1))" for c in term.classes if c != "sdcar"})
    dims = names.get("zcar_coordinate", names["coordinate"])
    out.update({names[v]: f"{v}[1..{d.Nloc - 1 if dims != names['coordinate'] else d.Nloc}], brms's {v} (vector; dims "
                          f"{dims if v == 'zcar' else names['coordinate']}" + ("; Normal(0, 1)" if v == "nszcar" else "; flat, its density the "
                          f"potential {names['lpdf']}") + ")" for v in term.latent})
    return out
