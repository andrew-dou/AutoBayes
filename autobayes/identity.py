"""Versioned identity of the built target specification, without exporting rows."""
from __future__ import annotations
import hashlib
import json
from collections.abc import Mapping
from numbers import Real
import numpy as np

from autobayes.families import family_of

SCHEMA = 'autobayes-target-v1'


def _canonical(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.ndarray):
        return {'shape': list(value.shape), 'values': _canonical(value.ravel().tolist())}
    if isinstance(value, Real):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError('target identity requires finite numeric values')
        return {'float64': (0.0 if number == 0 else number).hex()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError('target identity mappings require string keys')
        return {k: _canonical(v) for k, v in sorted(value.items())}
    raise TypeError(f'unsupported target identity type: {type(value).__name__}')


def _prior(p):
    # an auxiliary parameter with its own formula (B01) has a mapping of priors, which enters with its formula instead
    return None if p is None or isinstance(p, Mapping) else {'name': p.name, 'parameters': dict(p.params)}


def _dpar_payload(model, ds):
    """The target identity's entry of one distributional parameter's own formula (B01)."""
    dd = model.design.dpars[ds.response]
    sub = model.priors.dpar_priors(model.spec, ds.response)
    return {'parameter': ds.response, 'formula': ds.canonical_formula, 'link': dd.link, 'intercept': ds.intercept,
            'centered': dd.centered, 'columns': dd.columns, 'X_raw': dd.X_raw, 'X_model': dd.X_model, 'x_means': dd.x_means,
            'categorical': [{'name': c.name, 'reference': c.reference, 'levels': c.levels, 'coefficients': c.coefficients,
                             'full_rank': c.full_rank} for c in ds.categoricals],
            'priors': {'intercept': _prior(sub.intercept) if ds.intercept else None,
                       'coefficients': {k: _prior(v) for k, v in sub.coefficient_priors(ds).items()}},
            # D08: a shared correlation ID's terms of the formula, only when it has one (every other identity is unchanged)
            **({'groups': [{'factor': g, 'coef': c, 'id': ds.group_term_of(g).id, 'levels': block.levels, 'index': block.group_index,
                            'Z': block.Z, 'prior': _prior(sub.sd_priors(ds)[(g, c)])} for (g, c), block in dd.group_blocks.items()]}
               if ds.group_terms else {})}


def _monotonic_payload(model):
    """The target identity's entries of the monotonic terms (B03): each term's coding, codes and number of steps, and its
    simplex prior with the concentrations it gives (its coefficient's prior is among the coefficients)."""
    from autobayes.priors import dirichlet_concentration

    s, simplexes = model.spec, model.priors.simo_priors(model.spec)
    out = []
    for column, block in model.design.monotonic.items():
        m = s.monotonic(column)
        out.append({'term': block.term, 'column': column, 'minimum': m.minimum, 'maximum': m.maximum, 'categories': list(m.categories),
                    'codes': block.codes, 'D': block.D, 'prior': _prior(simplexes[column]),
                    'concentration': dirichlet_concentration(simplexes[column], block.D)})
    return out


def _measurement_payload(model):
    """The target identity's entries of the me() terms (C06): each term's noisy values and known noise and the priors of its
    latent values (its coefficient's prior is among the coefficients)."""
    s, pr = model.spec, model.priors
    means, sds = pr.me_priors(s, 'meanme'), pr.me_priors(s, 'sdme')
    return [{'term': block.term, 'column': x, 'noise_column': block.noise_column, 'Xn': block.Xn, 'noise': block.noise,
             'meanme': _prior(means[x]), 'sdme': _prior(sds[x])} for x, block in model.design.measurement_error.items()]


def _mixture_payload(model):
    """The target identity's entry of a mixture (C05): its component families and identification, each component's
    priors (intercept, coefficients and sigma) and theta's prior with the concentrations it gives."""
    from autobayes.families.mixture import ORDER
    from autobayes.priors import dirichlet_concentration

    s, pr = model.spec, model.priors
    return {'families': list(model.components), 'order': ORDER,
            'components': [{'intercept': _prior(c.intercept), 'sigma': _prior(c.sigma),
                            'coefficients': {k: _prior(v) for k, v in c.coefficient_priors(s).items()}} for c in pr.components],
            'theta': _prior(pr.theta), 'concentration': dirichlet_concentration(pr.theta, len(model.components))}


def _gp_payload(model):
    """The target identity's entries of the Gaussian processes (C03): each process's term, distinct inputs, scaling and row
    index, an approximate process's basis and frequencies, and its sdgp and lscale priors."""
    s, pr = model.spec, model.priors
    sdgp, lscale = pr.sdgp_priors(s), pr.lscale_priors(s)
    out = []
    for column, block in model.design.gp.items():
        g = s.gp(column)
        out.append({'term': g.term, 'column': column, 'k': g.k, 'c': g.c, 'inputs': list(g.inputs), 'dmax': g.dmax,
                    'cmean': g.cmean, 'index': block.index,
                    **({'basis': block.basis, 'slambda': block.slambda} if g.approximate else {}),
                    'priors': {'sdgp': _prior(sdgp[column]), 'lscale': _prior(lscale[column])}})
    return out


def _autocor_payload(model):
    """The target identity's entry of an autocorrelation term (C04): the term, its series and each fitted row's series and
    position in time order (what the recursion reads), and the priors of its coefficients."""
    term, design = model.spec.autocor, model.design.autocor
    return {'term': term.term, 'time': term.time, 'gr': term.gr, 'p': term.p, 'q': term.q, 'series': list(design.series),
            'series_index': design.series_index, 'position': design.position,
            'priors': {part: _prior(prior) for part, prior in model.priors.autocor_priors(model.spec).items()}}


def freeze_target(model):
    """Call once immediately after construction; only the digest is retained."""
    s, d, pr = model.spec, model.design, model.priors
    mixture = bool(getattr(model, 'components', ()))      # C05: a mixture's priors are its components' (its own entry below)
    prior = _prior
    sd_priors = pr.sd_priors(s)
    cor_priors = pr.cor_priors(s)
    # Varying slopes (A01) enter as their own entry, present only when the formula has one, so that every model
    # without slopes keeps the identity it had under this schema version; the LKJ prior of a correlated term (A02)
    # likewise enters only when the formula has one (the 'correlated' flag of the slope entries already separates
    # '(1 + x | g)' from '(1 + x || g)').
    slopes = [(g, c, block) for (g, c), block in d.group_blocks.items() if c != 'Intercept']
    payload = {
        'schema': SCHEMA, 'family': model.family,
        # a learned Student-t nu (A13) is not a value: it enters through its prior below, so fixed-nu identities are unchanged
        'family_parameters': {name: getattr(model, name) for name in family_of(model).fixed_parameters
                              if getattr(model, name) is not None},
        'formula': s.canonical_formula, 'link': model.link,
        'intercept': s.intercept, 'centered': d.centered,
        # a response whose missing values are parameters (C06) enters with 0 at its missing rows, which 'missing' lists
        'columns': d.columns, 'y': np.where(np.isnan(d.y), 0.0, d.y) if s.response in getattr(d, 'missing', {}) else d.y,
        'trials': d.trials,
        'X_raw': d.X_raw, 'X_model': d.X_model, 'x_means': d.x_means,
        'categorical': [{'name': c.name, 'reference': c.reference, 'levels': c.levels,
                         'coefficients': c.coefficients, 'full_rank': c.full_rank}
                        for c in s.categoricals],
        'groups': [{'factor': g, 'levels': block.levels, 'index': block.group_index,
                    'Z': block.Z, 'noncentered': True} for g, block in d.group_designs.items()],
        **({'group_slopes': [{'factor': g, 'coef': c, 'Z': block.Z, 'correlated': s.group_term_of(g).correlated,
                              'prior': prior(sd_priors[(g, c)])} for g, c, block in slopes]} if slopes else {}),
        **({'correlations': [{'factor': g, 'coefs': list(s.group_term_of(g).coefs), 'prior': prior(p)}
                             for g, p in cor_priors.items()]} if cor_priors else {}),
        # D08: a multi-membership term's memberships and weights, a gr(by = ) term's by levels, and a correlation ID shared with
        # a distributional formula likewise enter only when the formula has one.
        **({'multi_membership': [{'factor': t.factor, 'members': list(t.members), 'weights': list(t.weights),
                                  'index': list(d.group_blocks[(t.factor, 'Intercept')].memberships),
                                  'W': list(d.group_blocks[(t.factor, 'Intercept')].weights)} for t in s.group_terms if t.is_mm]}
           if any(t.is_mm for t in s.group_terms) else {}),
        **({'group_by': [{'factor': t.factor, 'by': t.by, 'by_levels': list(d.group_blocks[(t.factor, 'Intercept')].by_levels),
                          'by_index': d.group_blocks[(t.factor, 'Intercept')].by_index} for t in s.group_terms if t.by is not None]}
           if any(t.by is not None for t in s.group_terms) else {}),
        **({'shared_ids': [{'factor': sh.factor, 'id': sh.id, 'coefs': list(sh.labels)} for sh in s.shared_ids]}
           if getattr(s, 'shared_ids', ()) else {}),
        # The offset term (A06) likewise enters only when the formula has one.
        **({'offset': {'term': s.offset.term, 'values': d.offset}} if d.offset is not None else {}),
        # So do the response addition terms (B02): the terms as the canonical formula writes them (trunc()'s bounds are
        # in them) and the per-row weights and censoring codes they read.
        **({'additions': {'terms': [a.term for a in s.additions],
                          **({'weights': d.weights} if d.weights is not None else {}),
                          **({'cens': d.cens} if d.cens is not None else {})}} if s.additions else {}),
        # The response's categories (B05: cumulative, categorical), whose 1-based codes are y, likewise
        **({'categories': list(model.categories)} if getattr(model, 'categories', ()) else {}),
        # and hurdle_cumulative's hurdle category (E01), brms's Y = 0
        **({'hurdle_category': model.hurdle_category} if getattr(model, 'hurdle_category', None) is not None else {}),
        # So do distributional formulas (B01): each parameter's canonical formula, link, design and priors; its own
        # entry under 'priors' is then None.
        **({'distributional': [_dpar_payload(model, ds) for ds in s.dpars]} if s.dpars else {}),
        # So do monotonic terms (B03): their coding, codes and simplex priors (each coefficient's prior is a coefficient's).
        **({'monotonic': _monotonic_payload(model)} if s.mo_terms else {}),
        # So does a mixture (C05): its components, their priors and theta's; its own entries under 'priors' are then empty.
        **({'mixture': _mixture_payload(model)} if mixture else {}),
        # So do Gaussian processes (C03): their inputs, scaling, row index, basis and priors.
        **({'gaussian_processes': _gp_payload(model)} if getattr(s, 'gp_terms', ()) else {}),
        # So does an autocorrelation term (C04): its series, positions and coefficient priors.
        **({'autocorrelation': _autocor_payload(model)} if getattr(s, 'autocor', None) is not None else {}),
        # So do an me() term (C06): its noisy values, known noise and the priors of its latent values; and each column whose
        # missing values are parameters (C06): its values with 0 at the missing rows, and those rows.
        **({'measurement_error': _measurement_payload(model)} if getattr(d, 'measurement_error', {}) else {}),
        **({'missing': [{'column': c, 'values': block.filled, 'index': block.index} for c, block in d.missing.items()]}
           if getattr(d, 'missing', {}) else {}),
        # So does a shrinkage prior on b (D07, horseshoe() or R2D2()): the prior and its arguments, its scaling by sigma (brms's
        # autoscale = TRUE) and the design columns its vectors (zb, the local scales, R2D2's simplex) follow.
        **({'shrinkage': {'class': 'b', 'prior': prior(pr.b), 'autoscale': 'sigma', 'columns': list(d.columns)}}
           if getattr(model, 'shrinkage_names', {}) else {}),
        # So do category-specific terms (D04): each cs() term's column on the fitted rows (its coefficients' prior is among the
        # coefficients'); unordered thresholds are the family's (sratio, cratio, acat), which 'family' names.
        **({'category_specific': [{'term': f'cs({c})', 'column': c, 'values': values} for c, values in d.category_specific.items()]}
           if getattr(d, 'category_specific', {}) else {}),
        'priors': {'intercept': prior(pr.intercept), 'sigma': prior(pr.sigma),
                   'coefficients': {} if mixture else {k: prior(v) for k,v in pr.coefficient_priors(s).items()},
                   'groups': {k: prior(v) for k,v in pr.group_priors(s).items()},
                   # the shape prior (A08/A09) enters only for the families that have one, so every other model
                   # keeps the identity it had under this schema version
                   **({'shape': prior(pr.shape)} if pr.shape is not None else {}),
                   # the Beta precision prior (A11) likewise
                   **({'phi': prior(pr.phi)} if pr.phi is not None else {}),
                   # the prior of a learned Student-t nu (A13) likewise
                   **({'nu': prior(pr.nu)} if pr.nu is not None else {}),
                   # the zero-inflation / hurdle probability's prior (B06) likewise
                   **({'zi': prior(pr.zi)} if pr.zi is not None else {}),
                   **({'hu': prior(pr.hu)} if pr.hu is not None else {}),
                   # a family's extra parameter's prior (D01: alpha, beta, ndt) likewise
                   **{name: prior(getattr(pr, name)) for name in ('alpha', 'beta', 'ndt') if getattr(pr, name, None) is not None},
                   # and E01's zoi, coi (zero_one_inflated_beta) and kappa (xbeta)
                   **{name: prior(getattr(pr, name)) for name in ('zoi', 'coi', 'kappa') if getattr(pr, name, None) is not None}},
    }
    encoded = json.dumps(_canonical(payload), sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    return {'schema': SCHEMA, 'sha256': hashlib.sha256(encoded).hexdigest()}


def comparison_identity(a, b, *, allow_legacy=False):
    identities = [record['model'].get('target_identity') for record in (a,b)]
    for identity in identities:
        if identity is None:
            continue
        if (not isinstance(identity, dict) or set(identity) != {'schema','sha256'}
                or identity['schema'] != SCHEMA or not isinstance(identity['sha256'], str)
                or len(identity['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in identity['sha256'])):
            raise ValueError('invalid or unsupported target identity; do not infer identity from summary metadata')
    if any(i is None for i in identities):
        if not allow_legacy:
            raise ValueError('target identity is missing; re-record from the original built model, or explicitly use allow_legacy=True for an unverified comparison')
        return 'unverified_legacy'
    if identities[0] != identities[1]:
        raise ValueError('records fit different models or data: target identities differ')
    return 'matched'


FIT_IDENTITY_ATTR = 'autobayes_target_identity_json'


def check_fit_binding(model, result, *, allow_unverified=False):
    """Check available result and serialized-draw identities against the model."""
    expected = {'model': {'target_identity': model.target_identity}}
    identities = []
    identity = getattr(result, 'target_identity', None)
    if identity is not None:
        identities.append(identity)
    idata = getattr(result, 'idata', result)
    attrs = getattr(idata, 'attrs', {})
    if attrs.get('autobayes_generation_kind') == 'prior_predictive':
        raise ValueError('prior predictive draws are not a posterior fit')
    if FIT_IDENTITY_ATTR in attrs:
        try:
            serialized = attrs[FIT_IDENTITY_ATTR]
            if not isinstance(serialized, str):
                raise ValueError('expected JSON text')
            identity = json.loads(serialized)
            if identity is None:
                raise ValueError('null identity')
        except (TypeError, ValueError) as exc:
            raise ValueError('invalid serialized fit target identity') from exc
        identities.append(identity)
    if not identities:
        if not allow_unverified:
            raise ValueError('fit target binding is missing; use allow_unverified=True only for an explicitly unverified legacy record')
        return 'unverified'
    for identity in identities:
        comparison_identity(expected, {'model': {'target_identity': identity}})
    return 'matched'
