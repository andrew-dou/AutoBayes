"""One frozen crossed-Gaussian CPU unit; no retry or response simulation."""
from pathlib import Path
from importlib import metadata
import datetime
import hashlib
import inspect
import json
import subprocess
import time
import traceback

ROOT = Path('/private/tmp/autobayes-multiple-intercepts-cpu-validation-1')
REPO = Path('/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/m3c3-multiple-intercepts')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return str(value)
    return value

def write(name, value):
    with (ROOT / name).open('x') as out:
        json.dump(clean(value), out, indent=2, allow_nan=False)
        out.write('\n')

def event(kind, **fields):
    row = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'event': kind, **fields}
    line = json.dumps(clean(row), allow_nan=False)
    with (ROOT / 'events.jsonl').open('a') as out: out.write(line + '\n')
    print(line, flush=True)

def verify(frozen):
    for row in frozen['inputs']:
        assert sha(row['path']) == row['sha256'], row['path']
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
    assert head == frozen['head'] and not status, (head, status)
    return {'head': head, 'status': status, 'hashes_verified': len(frozen['inputs'])}

def scalar(value):
    a = np.asarray(value)
    if a.size != 1: raise ValueError(f'Expected scalar, got {a.shape}')
    return float(a.item())

def blocks(frame, config):
    return {g: np.column_stack([(frame[g].to_numpy() == label).astype(float)
                               for label in labels]) for g, labels in config['levels'].items()}

def oracle(model, frame, config):
    Z = blocks(frame, config)
    x = frame.x.to_numpy(float)
    X = (x - x.mean())[:, None]
    A = np.column_stack([np.ones(len(frame)), X, .45 * Z['county'], .25 * Z['year']])
    Q = np.diag(np.r_[1 / 1.5**2, 1., np.ones(11)]) + A.T @ A / .7**2
    mean = np.linalg.solve(Q, A.T @ frame.y.to_numpy(float) / .7**2)
    pmm = model.pymc_model
    fixed = pm.do(pmm, {pmm['sigma']: .7, pmm['1|county_sigma']: .45, pmm['1|year_sigma']: .25})
    indices = {'Intercept': [0], 'x': [1], '1|county_offset': list(range(2, 8)),
               '1|year_offset': list(range(8, 13))}
    names = [v.name for v in fixed.value_vars]
    assert set(names) == set(indices), names
    order = np.concatenate([indices[n] for n in names])
    point = {n: np.asarray(mean[indices[n]][0] if len(indices[n]) == 1 else mean[indices[n]]) for n in names}
    hessian = fixed.compile_d2logp()(point)
    gradient = fixed.compile_dlogp()(point)
    gates = {'precision': bool(np.allclose(hessian, Q[np.ix_(order, order)], rtol=1e-9, atol=1e-9)),
             'gradient_at_mean': bool(np.allclose(gradient, 0, rtol=0, atol=1e-8)),
             'finite': bool(np.isfinite(hessian).all() and np.isfinite(gradient).all() and np.isfinite(mean).all()),
             'raw_X': bool(np.array_equal(model.design.X_raw, x[:, None])),
             'centered_X': bool(np.array_equal(model.design.X_model, X)),
             'concatenated_Z': bool(np.array_equal(model.design.Z, np.column_stack(list(Z.values())))),
             'centering': model.center_predictors and model.effective_center_predictors and model.design.centered,
             'zero_x_mean': float(x.mean()) == 0.}
    for g, labels in config['levels'].items():
        d = model.design.group_designs[g]
        gates[f'{g}_design'] = bool(np.array_equal(d.Z, Z[g]) and list(d.levels) == labels
            and d.group_index.tolist() == [labels.index(label) for label in frame[g]])
        gates[f'{g}_graph_labels'] = list(pmm.coords[f'{g}__factor_dim']) == labels
    natural = model.value_point(intercept=.2, b=[.3], sd={'county': .45, 'year': .25},
        z={'county': np.linspace(-.4, .6, 6), 'year': np.linspace(.5, -.3, 5)}, sigma=.7)
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[model.names['linpred']]]),
                       inputs=pmm.value_vars, point_fn=True, on_unused_input='ignore')
    expected = .2 + X[:, 0] * .3 + .45 * Z['county'] @ np.linspace(-.4, .6, 6) + .25 * Z['year'] @ np.linspace(.5, -.3, 5)
    actual = fn(natural)[0]
    gates['original_input_graph_arithmetic'] = bool(np.allclose(actual, expected, atol=1e-10, rtol=1e-10))
    result = {'gates': gates, 'passed': all(gates.values()), 'free_variable_names': names,
              'column_indices': indices, 'mapped_order': order, 'A': A, 'precision': Q, 'analytic_mean': mean,
              'negative_hessian': hessian, 'gradient': gradient,
              'max_precision_error': float(np.max(abs(hessian - Q[np.ix_(order, order)]))),
              'max_gradient_error': float(np.max(abs(gradient))), 'graph_max_error': float(np.max(abs(actual - expected)))}
    write('oracle.json', result)
    return result

def named(idata, config):
    p = idata.posterior
    arrays = {name: p[name].transpose('chain', 'draw').values for name in config['scalar_parameters']}
    for g, labels in config['levels'].items():
        for label in labels:
            arrays[f'1|{g}[{label}]'] = p[f'1|{g}'].sel({f'{g}__factor_dim': label}).transpose('chain', 'draw').values
            offset = f'1|{g}_offset'
            if offset in p:
                arrays[f'{offset}[{label}]'] = p[offset].sel({f'{g}__factor_dim': label}).transpose('chain', 'draw').values
    return arrays

def diagnostics(idata, backend, config):
    arrays = named(idata, config)
    stats = idata.sample_stats
    errors = {}
    def metric(label, fn):
        try: return scalar(fn())
        except Exception as exc:
            errors[label] = f'{type(exc).__name__}: {exc}'
            return float('nan')
    parameters = {}
    for name, a in arrays.items():
        parameters[name] = {'mean': float(a.mean()), 'sd': float(a.std(ddof=1)),
            'rhat': metric(name + ':rhat', lambda: az.rhat(a, method='rank')),
            'ess_bulk': metric(name + ':ess_bulk', lambda: az.ess(a, method='bulk')),
            'ess_tail': metric(name + ':ess_tail', lambda: az.ess(a, method='tail')),
            'mcse_mean': metric(name + ':mcse_mean', lambda: az.mcse(a, method='mean'))}
    posterior_checks = {name: {'dims': list(a.dims), 'shape': list(a.shape),
        'complete': a.sizes.get('chain') == 4 and a.sizes.get('draw') == 1000,
        'finite': bool(np.isfinite(a.values).all())} for name, a in idata.posterior.data_vars.items()}
    stat_checks = {}
    for name, a in stats.data_vars.items():
        excluded = backend == 'pymc' and name in config['unused_eigenvalue_placeholders'] and bool(np.isnan(a.values).all())
        stat_checks[name] = {'shape': list(a.shape), 'complete': a.sizes.get('chain') == 4 and a.sizes.get('draw') == 1000,
            'finite': bool(np.isfinite(a.values).all()), 'excluded_unused_all_nan': excluded,
            'nonfinite_count': int((~np.isfinite(a.values)).sum())}
    try: bfmi = np.asarray(az.bfmi(idata))
    except Exception as exc:
        bfmi = np.full(4, np.nan); errors['bfmi'] = f'{type(exc).__name__}: {exc}'
    div = metric('divergences', lambda: stats.diverging.sum())
    depth = metric('depth_saturation', lambda: (stats.tree_depth >= 10).sum())
    flags = metric('max_depth_flags', lambda: stats.reached_max_treedepth.sum()) if 'reached_max_treedepth' in stats else 0.
    gates = {'complete_finite_posterior': all(r['complete'] and r['finite'] for r in posterior_checks.values())
                and all(a.shape == (4, 1000) and np.isfinite(a).all() for a in arrays.values()),
        'core_stats_present': all(name in stats for name in config['core_sampler_statistics'][backend]),
        'complete_finite_active_stats': all(r['complete'] and (r['finite'] or r['excluded_unused_all_nan']) for r in stat_checks.values()),
        'finite_diagnostics': all(np.isfinite(v) for r in parameters.values() for v in r.values()) and np.isfinite(bfmi).all() and not errors,
        'zero_divergences': div == 0, 'no_depth_saturation': depth == 0, 'no_max_depth_flags': flags == 0,
        'rhat_le_1_01': all(np.isfinite(r['rhat']) and r['rhat'] <= 1.01 for r in parameters.values()),
        'ess_bulk_ge_400': all(np.isfinite(r['ess_bulk']) and r['ess_bulk'] >= 400 for r in parameters.values()),
        'ess_tail_ge_400': all(np.isfinite(r['ess_tail']) and r['ess_tail'] >= 400 for r in parameters.values()),
        'every_chain_bfmi_ge_0_3': bfmi.shape == (4,) and np.isfinite(bfmi).all() and (bfmi >= .3).all()}
    gates = {k: bool(v) for k, v in gates.items()}
    return {'parameters': parameters, 'posterior_checks': posterior_checks, 'sample_stats_checks': stat_checks,
        'metric_errors': errors, 'bfmi': bfmi, 'divergences': div,
        'divergences_per_chain': stats.diverging.sum('draw').values.tolist() if 'diverging' in stats else None,
        'max_depth': metric('max_depth', lambda: stats.tree_depth.max()), 'depth_saturated_draws': depth,
        'max_depth_flags': flags, 'gates': gates, 'passed': all(gates.values())}

def semantics(model, fit, frame, config):
    post = fit.idata.posterior
    arrays = named(fit.idata, config)
    predictions = {}
    for label, rows in [('training', None), ('six_reordered_rows', config['new_row_positions'])]:
        f = frame if rows is None else frame.iloc[rows].reset_index(drop=True)
        expected = arrays['Intercept'][..., None] + arrays['x'][..., None] * f.x.to_numpy()
        for g, labels in config['levels'].items():
            expected += np.stack([arrays[f'1|{g}[{level}]'] for level in f[g]], axis=-1)
        actual = model.predict(fit.idata, newdata=None if rows is None else f, kind='mean')['mu']
        values = actual.transpose('chain', 'draw', '__obs__').values
        predictions[label] = {'dims': list(actual.dims), 'shape': list(values.shape),
            'max_abs_error': float(np.max(abs(values - expected))), 'finite': bool(np.isfinite(values).all()),
            'passed': bool(np.allclose(values, expected, atol=1e-10, rtol=1e-10))}
    record = ab.fit_record(model, fit, max_treedepth=10)
    write(f'{fit.backend}-fit-record.json', record)
    groups = record['model']['groups']
    gates = {'formula': model.spec.formula == config['formula'] == record['model']['formula'],
        'groups': model.spec.groups == ('county', 'year') and model.spec.has_group
            and [g['factor'] for g in groups] == ['county', 'year'] and 'group' not in record['model'] and 'J' not in record['model'],
        'group_metadata': all(g['levels'] == config['levels'][g['factor']] and g['J'] == len(g['levels'])
            and g['coordinate'] == f"{g['factor']}__factor_dim" and g['noncentered']
            and g['sd_prior'] == config['record_priors'][f"sd:{g['factor']}"]
            and g['names'] == model.group_names[g['factor']] for g in groups),
        'posterior_labels': all(post[f'1|{g}'].dims == ('chain', 'draw', f'{g}__factor_dim')
            and list(post.coords[f'{g}__factor_dim'].values) == labels for g, labels in config['levels'].items()),
        'intercept_and_centering': 'Intercept' in post and 'Intercept' in model.pymc_model.named_vars
            and record['model']['intercept'] and record['model']['center_predictors_requested']
            and record['model']['center_predictors'] and model.design.centered and float(frame.x.mean()) == 0,
        'coefficient_order': list(model.spec.columns) == ['x'] == record['model']['design_columns'],
        'exact_priors': record['model']['priors'] == config['record_priors'],
        'five_record_parameters': set(record['posterior']) == set(config['scalar_parameters']),
        'record_means': all(np.isclose(record['posterior'][n]['mean'], arrays[n].mean(), rtol=1e-12, atol=1e-12) for n in config['scalar_parameters']),
        'record_diagnostics': bool(record['diagnostics']['values']) and set(record['diagnostics']) == {'values', 'status', 'overall'},
        'mean_predictions': all(r['passed'] and r['finite'] for r in predictions.values()),
        'record_json_serializes': bool(json.dumps(record, allow_nan=False))}
    settings = next(f for f in config['fits'] if f['backend'] == fit.backend)
    gates['execution_settings'] = all(record['execution'][k] == settings[k] for k in ['backend', 'chains', 'draws', 'tune', 'seed', 'target_accept'])
    gates['cpu_float64'] = record['execution']['device_requested'] == 'cpu' and fit.platform == 'cpu' and fit.dtype == 'float64'
    gates['chain_method'] = fit.chain_method == ('parallel' if fit.backend == 'numpyro' else None)
    gates = {k: bool(v) for k, v in gates.items()}
    return {'gates': gates, 'predictions': predictions, 'passed': all(gates.values())}

def main():
    global np, az, ab, pm
    frozen = json.loads((ROOT / 'freeze.json').read_text())
    before = verify(frozen)
    config = json.loads((ROOT / 'config.json').read_text())
    import autobayes as ab
    import numpy as np
    import arviz as az
    import pymc as pm
    import pandas as pd
    import pytensor
    from numpyro.infer import NUTS
    assert Path(ab.__file__).resolve().parent.parent == REPO
    versions = {name: metadata.version(name) for name in config['expected_versions']}
    assert versions == config['expected_versions']
    assert inspect.signature(pm.NUTS.__init__).parameters['max_treedepth'].default == 10
    assert inspect.signature(NUTS.__init__).parameters['max_tree_depth'].default == 10
    environment = ab.environment()
    assert pytensor.config.floatX == 'float64' and environment['jax_x64']
    write('environment.json', {'versions': versions, 'autobayes_source_version': ab.__version__, 'environment': environment,
        'source_verification': before, 'pymc_nuts_signature': str(inspect.signature(pm.NUTS.__init__)),
        'numpyro_nuts_signature': str(inspect.signature(NUTS.__init__)), 'depth_source': config['depth_source']})
    frame = pd.DataFrame(json.loads((ROOT / 'data.json').read_text()))
    model = ab.gaussian(config['formula'], frame, ab.Priors(intercept=ab.Normal(0, 1.5), b=ab.Normal(0, 1),
        sd={'county': ab.HalfNormal(.8), 'year': ab.Exponential(2)}, sigma=ab.HalfNormal(1)), center_predictors=True)
    write('model-snapshot.json', {'describe': model.describe(), 'names': model.names, 'group_names': model.group_names,
        'formula': model.spec.formula, 'levels': dict(model.spec.levels_by_group), 'data': model.data.to_dict(orient='records')})
    oracle_result = oracle(model, frame, config)
    event('oracle_completed', passed=oracle_result['passed'])
    if not oracle_result['passed']:
        write('result.json', {'passed': False, 'stage': 'oracle', 'fit_attempts': 0, 'oracle': oracle_result})
        return 4
    fits, outputs, fit_errors = {}, {}, {}
    for settings in config['fits']:
        backend = settings['backend']
        event('fit_started', backend=backend, settings=settings)
        try:
            started = time.monotonic()
            fit = model.fit(**settings)
            path = ROOT / f'{backend}.nc'
            assert not path.exists()
            fit.idata.to_netcdf(str(path))
            outputs[backend] = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
                'fit_wall_seconds': fit.wall_seconds, 'fit_and_serialization_seconds': time.monotonic() - started,
                'metadata': {k: getattr(fit, k) for k in ['backend', 'device_requested', 'device_actual', 'platform', 'dtype', 'chain_method', 'chains', 'draws', 'tune', 'target_accept', 'seed', 'notes']}}
            write(f'{backend}-netcdf-receipt.json', outputs[backend])
            fits[backend] = fit
            event('fit_saved', backend=backend, **outputs[backend])
        except Exception as exc:
            fit_errors[backend] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
            write(f'{backend}-fit-error.json', fit_errors[backend])
            event('fit_failed', backend=backend, error=fit_errors[backend])
    results, records = {}, {}
    for backend, fit in fits.items():
        fit.idata = az.from_netcdf(outputs[backend]['path'])
        fit.idata.posterior.load(); fit.idata.sample_stats.load()
        results[backend] = {}
        for name, fn in [('diagnostics', lambda: diagnostics(fit.idata, backend, config)),
                         ('semantics', lambda: semantics(model, fit, frame, config))]:
            try: results[backend][name] = fn()
            except Exception as exc:
                results[backend][name] = {'passed': False, 'error': type(exc).__name__ + ': ' + str(exc), 'traceback': traceback.format_exc()}
        record_path = ROOT / f'{backend}-fit-record.json'
        if record_path.exists(): records[backend] = json.loads(record_path.read_text())
        results[backend]['retained_hash_matches'] = sha(outputs[backend]['path']) == outputs[backend]['sha256']
        results[backend]['passed'] = all(results[backend][k]['passed'] for k in ['diagnostics', 'semantics']) and results[backend]['retained_hash_matches']
        write(f'{backend}-analysis.json', results[backend])
        event('backend_analyzed', backend=backend, passed=results[backend]['passed'])
    comparison, record_check = {}, {'passed': False}
    if all('parameters' in results.get(b, {}).get('diagnostics', {}) for b in ['pymc', 'numpyro']):
        for label in results['pymc']['diagnostics']['parameters']:
            a = results['pymc']['diagnostics']['parameters'][label]
            b = results['numpyro']['diagnostics']['parameters'][label]
            difference = abs(a['mean'] - b['mean'])
            tolerance = 4 * float(np.hypot(a['mcse_mean'], b['mcse_mean']))
            comparison[label] = {'pymc_mean': a['mean'], 'numpyro_mean': b['mean'], 'pymc_mcse': a['mcse_mean'],
                'numpyro_mcse': b['mcse_mean'], 'absolute_difference': difference, 'four_combined_mcse': tolerance,
                'passed': bool(np.isfinite(difference) and np.isfinite(tolerance) and difference <= tolerance),
                'acceptance_gate': label in config['scalar_parameters']}
    if set(records) == {'pymc', 'numpyro'}:
        try:
            record_check = ab.compare_records(records['pymc'], records['numpyro'])
            mismatch = json.loads(json.dumps(records['numpyro']))
            mismatch['posterior'].pop('sigma')
            try: ab.compare_records(records['pymc'], mismatch)
            except ValueError: refused = True
            else: refused = False
            record_check['unequal_parameter_sets_refused'] = refused
            record_check['passed'] = record_check['agree'] and refused and set(record_check['rows']) == set(config['scalar_parameters'])
        except Exception as exc: record_check = {'passed': False, 'error': str(exc), 'traceback': traceback.format_exc()}
    scalar_gate = all(n in comparison and comparison[n]['passed'] for n in config['scalar_parameters'])
    after = verify(frozen)
    passed = len(fits) == 2 and not fit_errors and all(r['passed'] for r in results.values()) and scalar_gate and record_check['passed']
    result = {'passed': passed, 'fit_attempts': 2, 'fits_saved': len(fits), 'fit_errors': fit_errors,
        'oracle_passed': oracle_result['passed'], 'backends': results, 'backend_mean_comparison': comparison,
        'five_scalar_mean_gate': scalar_gate, 'record_comparison': record_check, 'outputs': outputs,
        'initial_integrity': before, 'final_integrity': after, 'new_response_draws': 0,
        'limitations': 'One crossed Gaussian CPU backend/API check; no statistical calibration, all-family sampling, novel-group accuracy or GPU claim.'}
    write('result.json', result)
    event('finished', passed=passed, fit_attempts=2, fits_saved=len(fits))
    return 0 if passed else 4

if __name__ == '__main__':
    try: code = main()
    except Exception as exc:
        with (ROOT / 'execution-error.json').open('x') as out:
            json.dump({'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}, out, indent=2)
        traceback.print_exc()
        code = 2
    raise SystemExit(code)
