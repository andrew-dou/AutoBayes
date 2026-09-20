"""One frozen, two-backend CPU acceptance unit. No retries or extra fits."""
from pathlib import Path
import datetime
import hashlib
import inspect
import json
import os
import runpy
import subprocess
import sys
import time
import traceback
from importlib import metadata

ROOT = Path('/private/tmp/autobayes-intercept-cpu-validation')
REPO = Path('/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/m3c2-intercept-controls')
HEAD = '6a58eb3c9b8ab510aa8bd5d3d04936e64990dace'
ATTEMPT = ROOT / 'attempt-1'

def jsonable(obj):
    # Retain nonfinite failures explicitly without writing invalid JSON.
    import numpy as np
    if isinstance(obj, dict): return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray): return jsonable(obj.tolist())
    if isinstance(obj, np.generic): return jsonable(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj): return str(obj)
    return obj

def write(name, value):
    path = ATTEMPT / name
    with path.open('x') as stream:
        json.dump(jsonable(value), stream, indent=2, allow_nan=False)
        stream.write('\n')

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def event(kind, **details):
    row = {'time': stamp(), 'event': kind, **details}
    with (ATTEMPT / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(jsonable(row), allow_nan=False) + '\n')
    print(json.dumps(jsonable(row), allow_nan=False), flush=True)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_frozen(frozen):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
    assert head == HEAD and not status, (head, status)
    for entry in frozen['inputs']:
        assert sha(entry['path']) == entry['sha256'], entry['path']
    return {'head': head, 'status_porcelain': status, 'input_hashes_verified': len(frozen['inputs'])}

def named_arrays(idata):
    import numpy as np
    post = idata.posterior
    assert list(post.coords['arm_dim'].values) == ['control', 'treated']
    assert list(post.coords['dose:arm_dim'].values) == ['treated']
    named = {
        'dose': post['dose'],
        'arm[control]': post['arm'].sel(arm_dim='control'),
        'arm[treated]': post['arm'].sel(arm_dim='treated'),
        'dose:arm[treated]': post['dose:arm'].sel({'dose:arm_dim': 'treated'}),
        'sigma': post['sigma'],
    }
    return {name: np.asarray(value.transpose('chain', 'draw').values) for name, value in named.items()}

def analyze_draws(idata, backend, config):
    import arviz as az
    import numpy as np
    post, stats = idata.posterior, idata.sample_stats
    arrays = named_arrays(idata)
    posterior_checks = {}
    for name, value in post.data_vars.items():
        values = np.asarray(value.values)
        posterior_checks[name] = {'shape': list(values.shape), 'dims': list(value.dims),
            'complete': value.sizes.get('chain') == 4 and value.sizes.get('draw') == 1000,
            'finite': bool(np.isfinite(values).all()), 'dtype': str(values.dtype)}
    parameter_checks = {}
    for name, value in arrays.items():
        parameter_checks[name] = {
            'mean': float(value.mean()), 'sd': float(value.std(ddof=1)),
            'rhat': float(az.rhat(value, method='rank')),
            'ess_bulk': float(az.ess(value, method='bulk')),
            'ess_tail': float(az.ess(value, method='tail')),
            'mcse_mean': float(az.mcse(value, method='mean')),
            'complete': value.shape == (4, 1000), 'finite': bool(np.isfinite(value).all()),
        }
    statistic_checks, exclusions = {}, []
    for name, value in stats.data_vars.items():
        values = np.asarray(value.values)
        excluded = (backend == 'pymc' and name in config['unused_eigenvalue_placeholders'] and np.isnan(values).all())
        if excluded: exclusions.append(name)
        statistic_checks[name] = {'shape': list(values.shape), 'dims': list(value.dims), 'dtype': str(values.dtype),
            'complete': value.sizes.get('chain') == 4 and value.sizes.get('draw') == 1000,
            'finite': bool(np.isfinite(values).all()), 'excluded_unused_all_nan': bool(excluded),
            'nonfinite_count': int((~np.isfinite(values)).sum())}
    core = config['core_sampler_statistics'][backend]
    core_present = all(name in stats for name in core)
    divergences = int(stats['diverging'].sum()) if 'diverging' in stats else None
    saturation = int((stats['tree_depth'] >= 10).sum()) if 'tree_depth' in stats else None
    bfmi = np.asarray(az.bfmi(idata), dtype=float) if 'energy' in stats else np.full(4, np.nan)
    core_diag_finite = all(np.isfinite(row[key]) for row in parameter_checks.values()
        for key in ['mean', 'sd', 'rhat', 'ess_bulk', 'ess_tail', 'mcse_mean']) and bool(np.isfinite(bfmi).all())
    gates = {
        'four_complete_finite_posterior_chains': all(row['complete'] and row['finite'] for row in posterior_checks.values())
             and all(row['complete'] and row['finite'] for row in parameter_checks.values())
             and post.sizes.get('chain') == 4 and post.sizes.get('draw') == 1000,
        'no_intercept_posterior': 'Intercept' not in post,
        'core_sampler_statistics_present': core_present,
        'complete_finite_active_sampler_statistics': all(row['complete'] and (row['finite'] or row['excluded_unused_all_nan'])
             for row in statistic_checks.values()),
        'core_diagnostics_finite': bool(core_diag_finite),
        'zero_divergences': divergences == 0,
        'rhat_at_most_1_01': all(np.isfinite(row['rhat']) and row['rhat'] <= 1.01 for row in parameter_checks.values()),
        'bulk_ess_at_least_400': all(np.isfinite(row['ess_bulk']) and row['ess_bulk'] >= 400 for row in parameter_checks.values()),
        'tail_ess_at_least_400': all(np.isfinite(row['ess_tail']) and row['ess_tail'] >= 400 for row in parameter_checks.values()),
        'bfmi_at_least_0_3': bool(np.isfinite(bfmi).all() and (bfmi >= .3).all()),
        'no_depth_saturation': saturation == 0,
        'no_backend_max_depth_flags': bool(not stats['reached_max_treedepth'].any()) if 'reached_max_treedepth' in stats else True,
    }
    return {'backend': backend, 'parameters': parameter_checks, 'posterior': posterior_checks,
        'sampler_statistics': statistic_checks, 'excluded_unused_placeholders': exclusions,
        'divergences': divergences, 'depth_saturated_draws': saturation,
        'max_tree_depth_observed': int(stats['tree_depth'].max()) if 'tree_depth' in stats else None,
        'bfmi_by_chain': bfmi.tolist(), 'max_rhat': max(row['rhat'] for row in parameter_checks.values()),
        'min_ess_bulk': min(row['ess_bulk'] for row in parameter_checks.values()),
        'min_ess_tail': min(row['ess_tail'] for row in parameter_checks.values()),
        'min_bfmi': float(bfmi.min()), 'gates': gates, 'passed': all(gates.values())}

def semantic_checks(model, fit, frozen):
    import numpy as np
    import pandas as pd
    import autobayes as ab
    data = pd.DataFrame(frozen['config']['data'])
    new = pd.DataFrame(frozen['config']['newdata'])
    arrays = named_arrays(fit.idata)
    def design(frame):
        dose = frame['dose'].to_numpy(dtype=float)
        control = (frame['arm'].to_numpy() == 'control').astype(float)
        treated = (frame['arm'].to_numpy() == 'treated').astype(float)
        return np.column_stack([dose, control, treated, dose * treated])
    beta = np.stack([arrays[name] for name in frozen['config']['coefficient_labels']], axis=-1)
    predictions = {}
    for name, frame, data_arg in [('fitted', data, None), ('new', new, new)]:
        expected = np.einsum('cdp,np->cdn', beta, design(frame))
        predicted = model.predict(fit.idata, newdata=data_arg, kind='mean')['mu'].transpose('chain', 'draw', '__obs__').values
        difference = np.abs(predicted - expected)
        predictions[name] = {'shape': list(predicted.shape), 'max_abs_error': float(difference.max()),
            'finite': bool(np.isfinite(predicted).all()),
            'within_tolerance': bool(np.allclose(predicted, expected, rtol=1e-10, atol=1e-10))}
    record = ab.fit_record(model, fit, max_treedepth=10)
    write(f'{fit.backend}-fit-record.json', record)
    expected_priors = frozen['config']['record_priors']
    gates = {
        'no_intercept_graph_or_names': 'Intercept' not in model.pymc_model.named_vars and 'intercept' not in model.names,
        'no_intercept_posterior_or_prior': 'Intercept' not in record['posterior'] and 'intercept' not in record['model']['priors'],
        'record_intercept_absent': record['model']['intercept'] is False and record['model']['intercept_prior_applies_to'] is None,
        'record_centering_correct': record['model']['center_predictors_requested'] is True and record['model']['center_predictors'] is False,
        'actual_centering_correct': model.center_predictors is True and model.effective_center_predictors is False and not model.design.centered,
        'record_priors_exact': record['model']['priors'] == expected_priors,
        'record_coefficient_order': record['model']['design_columns'] == frozen['config']['coefficient_labels'],
        'raw_design_exact': bool(np.array_equal(model.design.X_raw, design(data)) and np.array_equal(model.design.X_model, design(data))
             and np.array_equal(model.design.x_means, np.zeros(4))),
        'record_five_named_parameters': set(record['posterior']) == set(frozen['config']['parameter_labels']),
        'predictions_correct': all(row['finite'] and row['within_tolerance'] for row in predictions.values()),
    }
    return {'predictions': predictions, 'gates': gates, 'passed': all(gates.values())}

def main():
    frozen = json.loads((ROOT / 'freeze.json').read_text())
    initial_verification = verify_frozen(frozen)
    # All numerical imports, compilation, fitting, and serialization occur inside the supervisor deadline.
    import autobayes as ab
    import arviz as az
    import numpy as np
    import pandas as pd
    import pymc as pm
    from numpyro.infer import NUTS
    assert Path(ab.__file__).resolve().parent.parent == REPO
    versions = {package: metadata.version(package) for package in frozen['config']['expected_versions']}
    assert versions == frozen['config']['expected_versions'], versions
    assert inspect.signature(pm.NUTS.__init__).parameters['max_treedepth'].default == 10
    assert inspect.signature(NUTS.__init__).parameters['max_tree_depth'].default == 10
    write('environment.json', {'versions': versions, 'autobayes_module': str(Path(ab.__file__).resolve()),
        'environment': ab.environment(), 'pymc_nuts_signature': str(inspect.signature(pm.NUTS.__init__)),
        'numpyro_nuts_signature': str(inspect.signature(NUTS.__init__)), 'verification': initial_verification})
    model = runpy.run_path(str(REPO / 'examples/m3c2_no_intercept.py'), run_name='frozen_example')['build_example']()
    config = frozen['config']
    assert model.data.to_dict(orient='records') == config['data']
    assert model.spec.formula == config['formula']
    assert list(model.spec.columns) == config['coefficient_labels']
    prior_strings = {f'b:{key}': str(value) for key, value in model.priors.coefficient_priors(model.spec).items()}
    prior_strings['sigma'] = str(model.priors.sigma)
    assert prior_strings == config['record_priors'], prior_strings
    assert model.priors.intercept is None and not model.spec.intercept
    write('model-snapshot.json', {'formula': model.spec.formula, 'data': model.data.to_dict(orient='records'),
        'columns': list(model.spec.columns), 'X_raw': model.design.X_raw.tolist(), 'X_model': model.design.X_model.tolist(),
        'x_means': model.design.x_means.tolist(), 'requested_centering': model.center_predictors,
        'effective_centering': model.effective_center_predictors, 'priors': prior_strings,
        'graph_names': list(model.pymc_model.named_vars), 'value_names': [v.name for v in model.pymc_model.value_vars],
        'describe': model.describe()})
    # Execute the two prespecified fits once each. Acceptance is evaluated only after their immediate serialization.
    fits, outputs = {}, {}
    for run in config['fits']:
        backend = run['backend']
        event('fit_started', backend=backend, kwargs=run)
        start = time.monotonic()
        fit = model.fit(**run)
        fits[backend] = fit
        netcdf = ATTEMPT / f'{backend}.nc'
        assert not netcdf.exists()
        fit.idata.to_netcdf(str(netcdf))
        outputs[backend] = {'path': str(netcdf), 'sha256': sha(netcdf), 'bytes': netcdf.stat().st_size,
            'fit_wall_seconds': fit.wall_seconds, 'fit_and_serialization_wall_seconds': time.monotonic() - start,
            'fit_metadata': {k: getattr(fit, k) for k in ['backend', 'device_requested', 'device_actual', 'platform', 'dtype',
                  'chain_method', 'chains', 'draws', 'tune', 'target_accept', 'seed', 'notes']}}
        write(f'{backend}-netcdf-receipt.json', outputs[backend])
        event('fit_saved', backend=backend, **outputs[backend])
    # Reopen actual retained files. Do not use transient in-memory posterior objects for acceptance.
    results = {}
    for backend, fit in fits.items():
        fit.idata = az.from_netcdf(outputs[backend]['path'])
        fit.idata.posterior.load()
        fit.idata.sample_stats.load()
        diagnostics = analyze_draws(fit.idata, backend, config)
        semantics = semantic_checks(model, fit, frozen)
        execution = {'cpu': fit.platform == 'cpu', 'float64': fit.dtype == 'float64',
            'exact_sampling_settings': fit.chains == 4 and fit.draws == 1000 and fit.tune == 1000 and fit.target_accept == .9,
            'numpyro_parallel_four_devices': (fit.chain_method == 'parallel') if backend == 'numpyro' else True}
        results[backend] = {'diagnostics': diagnostics, 'semantics': semantics, 'execution_gates': execution,
            'passed': diagnostics['passed'] and semantics['passed'] and all(execution.values())}
        write(f'{backend}-acceptance.json', results[backend])
        event('backend_evaluated', backend=backend, passed=results[backend]['passed'])
    comparison = {}
    for name in config['parameter_labels']:
        left = results['pymc']['diagnostics']['parameters'][name]
        right = results['numpyro']['diagnostics']['parameters'][name]
        difference = abs(left['mean'] - right['mean'])
        tolerance = 4 * float(np.hypot(left['mcse_mean'], right['mcse_mean']))
        comparison[name] = {'pymc_mean': left['mean'], 'numpyro_mean': right['mean'],
            'pymc_mcse_mean': left['mcse_mean'], 'numpyro_mcse_mean': right['mcse_mean'],
            'absolute_difference': difference, 'four_combined_mcse': tolerance,
            'passed': bool(np.isfinite(difference) and np.isfinite(tolerance) and difference <= tolerance)}
    final_verification = verify_frozen(frozen)
    passed = all(row['passed'] for row in results.values()) and all(row['passed'] for row in comparison.values())
    final = {'head': HEAD, 'finished': stamp(), 'passed': passed, 'posterior_fit_count': len(fits),
        'backends': results, 'comparison': comparison, 'outputs': outputs, 'final_verification': final_verification,
        'limitations': 'One small Gaussian backend/API agreement example; not calibration, all-family sampling evidence, or GPU validation.'}
    write('acceptance-result.json', final)
    event('acceptance_finished', passed=passed, posterior_fit_count=len(fits))
    return 0 if passed else 4

if __name__ == '__main__':
    try:
        exit_code = main()
    except Exception as exc:
        error = {'time': stamp(), 'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        with (ATTEMPT / 'execution-error.json').open('x') as stream:
            json.dump(error, stream, indent=2)
            stream.write('\n')
        traceback.print_exc()
        exit_code = 2
    # SystemExit is deliberately outside the exception handler.
    raise SystemExit(exit_code)
