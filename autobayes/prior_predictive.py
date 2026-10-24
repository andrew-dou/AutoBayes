"""Forward draws from the built graph, kept distinct from posterior results."""
from numbers import Integral
import json
import numpy as np
import pymc as pm

from autobayes.families import family_of


def prior_predictive(model, *, draws=500, seed=None):
    """Return native ArviZ prior groups for modeled rows using CPU forward draws.

    Parameters retain graph parameterization (including a centered intercept).
    This result is not a posterior fit. No new-data or sampler-backend option.
    """
    if getattr(model, 'responses', None):                   # C01: not part of the multivariate slice
        return model.prior_predictive(draws=draws, seed=seed)
    if isinstance(draws, (bool, np.bool_)) or not isinstance(draws, Integral) or draws <= 0:
        raise ValueError('draws must be a positive integer')
    if seed is not None and (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0):
        raise ValueError('seed must be a nonnegative integer or None')
    if 'response' not in family_of(model).prediction_kinds:   # F04: a family whose response is not drawn (wiener, divergence D149)
        raise ValueError(f'prior predictive draws are not offered for the {model.family} family: {family_of(model).kind_error}')
    priors = getattr(model, 'priors', None)               # a flat prior (ab.Flat, brms's default for b) has nothing to draw from
    improper = priors.improper(model.spec) if priors is not None else []
    if improper:
        raise ValueError(f'the prior of {improper} is flat (Flat(), brms\'s default for b): an improper prior has no prior '
                         'predictive distribution; give proper priors for these parameters to draw from the prior')
    identity = json.dumps(model.target_identity, sort_keys=True)
    try:
        result = pm.sample_prior_predictive(draws=int(draws), random_seed=None if seed is None else int(seed),
                                            model=model.pymc_model, return_inferencedata=True)
    except (ValueError, OverflowError, FloatingPointError) as exc:
        raise ValueError('prior predictive generation failed; inspect priors and numerical support; no retry performed') from exc
    for group in ('prior', 'prior_predictive', 'observed_data'):
        if group not in result.groups():
            raise ValueError(f'prior predictive output is missing {group}')
    for group in ('prior', 'prior_predictive'):
        ds = getattr(result, group)
        if ds.sizes.get('chain') != 1 or ds.sizes.get('draw') != draws:
            raise ValueError('prior predictive output has unexpected draw dimensions')
        for arr in ds.data_vars.values():
            if not np.isfinite(arr.values).all():
                raise ValueError('prior predictive output contains nonfinite draws; inspect priors')
    y = result.prior_predictive[model.names['y']].values
    if y.shape != (1, draws, model.design.n):
        raise ValueError('prior predictive response dimensions do not match modeled rows')
    fam = family_of(model)
    if fam.discrete:
        if (y < 0).any() or (y != np.floor(y)).any():
            raise ValueError('prior predictive counts violate response support')
        bound = fam.response_upper_bound(model)
        if bound is not None and (y > bound).any():
            raise ValueError('prior predictive counts exceed trials')
    if not np.array_equal(result.observed_data[model.names['y']].values, model.design.y):
        raise ValueError('prior predictive observed rows differ from built data')
    from autobayes import __version__
    result.attrs.update(autobayes_generation_kind='prior_predictive',
                        autobayes_built_target_identity_json=identity,
                        autobayes_seed_json=json.dumps(None if seed is None else int(seed)),
                        autobayes_requested_draws=int(draws), autobayes_version=__version__,
                        autobayes_parameterization='native_graph')
    return result
