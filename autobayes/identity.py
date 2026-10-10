"""Versioned identity of the built target specification, without exporting rows."""
from __future__ import annotations
import hashlib
import json
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


def freeze_target(model):
    """Call once immediately after construction; only the digest is retained."""
    s, d, pr = model.spec, model.design, model.priors
    def prior(p):
        return None if p is None else {'name': p.name, 'parameters': dict(p.params)}
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
        'columns': d.columns, 'y': d.y, 'trials': d.trials,
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
        # The offset term (A06) likewise enters only when the formula has one.
        **({'offset': {'term': s.offset.term, 'values': d.offset}} if d.offset is not None else {}),
        'priors': {'intercept': prior(pr.intercept), 'sigma': prior(pr.sigma),
                   'coefficients': {k: prior(v) for k,v in pr.coefficient_priors(s).items()},
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
                   **({'hu': prior(pr.hu)} if pr.hu is not None else {})},
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
