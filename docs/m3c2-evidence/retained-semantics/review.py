"""Read-only retained-draw semantic review. Sampling is forbidden."""
from pathlib import Path
from types import SimpleNamespace
import datetime
import hashlib
import json
import runpy
import subprocess
import traceback

ROOT = Path('/private/tmp/autobayes-intercept-retained-semantics')
SOURCE = Path('/private/tmp/autobayes-intercept-cpu-validation')
REPO = Path('/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/m3c2-intercept-controls')
HEAD = '6a58eb3c9b8ab510aa8bd5d3d04936e64990dace'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name, value):
    with (ROOT / name).open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write('\n')

def verify():
    frozen = json.loads((SOURCE / 'freeze.json').read_text())
    evidence = []
    for line in (SOURCE / 'CHECKSUMS.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = SOURCE / name
        assert sha(path) == digest, str(path)
        evidence.append({'path': str(path), 'sha256': digest})
    for entry in frozen['inputs']:
        assert sha(entry['path']) == entry['sha256'], entry['path']
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
    assert head == HEAD and not status, (head, status)
    return frozen, {'head': head, 'status_porcelain': status, 'original_frozen_inputs_verified': len(frozen['inputs']),
                    'original_evidence_verified': evidence}

def main():
    frozen, initial = verify()
    import autobayes as ab
    import autobayes.inference as inference
    import arviz as az
    import bambi as bmb
    import numpy as np
    import pandas as pd
    import pymc as pm
    import pymc.sampling.mcmc as mcmc
    from importlib import metadata
    assert Path(ab.__file__).resolve().parent.parent == REPO
    assert {pkg: metadata.version(pkg) for pkg in frozen['config']['expected_versions']} == frozen['config']['expected_versions']
    def forbidden(*args, **kwargs):
        raise AssertionError('No new fits, sampling, or response draws are permitted in this review')
    ab.Model.fit = ab.fit = inference.fit = bmb.Model.fit = forbidden
    pm.sample = mcmc.sample = pm.sample_posterior_predictive = pm.draw = forbidden
    model = runpy.run_path(str(REPO / 'examples/m3c2_no_intercept.py'), run_name='retained_review')['build_example']()
    config = frozen['config']
    train = pd.DataFrame(config['data'])
    new = pd.DataFrame(config['newdata'])
    assert model.data.to_dict(orient='records') == config['data']
    def design(frame):
        dose = frame['dose'].to_numpy(dtype=float)
        control = (frame['arm'].to_numpy() == 'control').astype(float)
        treated = (frame['arm'].to_numpy() == 'treated').astype(float)
        return np.column_stack([dose, control, treated, dose * treated])
    results = {}
    for backend in ['pymc', 'numpyro']:
        receipt = json.loads((SOURCE / 'attempt-1' / f'{backend}-netcdf-receipt.json').read_text())
        assert sha(receipt['path']) == receipt['sha256']
        idata = az.from_netcdf(receipt['path'])
        idata.posterior.load(); idata.sample_stats.load()
        post = idata.posterior
        arrays = {
            'dose': post['dose'].transpose('chain', 'draw').values,
            'arm[control]': post['arm'].sel(arm_dim='control').transpose('chain', 'draw').values,
            'arm[treated]': post['arm'].sel(arm_dim='treated').transpose('chain', 'draw').values,
            'dose:arm[treated]': post['dose:arm'].sel({'dose:arm_dim': 'treated'}).transpose('chain', 'draw').values,
            'sigma': post['sigma'].transpose('chain', 'draw').values,
        }
        beta = np.stack([arrays[label] for label in config['coefficient_labels']], axis=-1)
        prediction_checks = {}
        for name, frame, data_arg in [('fitted', train, None), ('new', new, new)]:
            expected = np.einsum('cdp,np->cdn', beta, design(frame))
            observed = model.predict(idata, newdata=data_arg, kind='mean')['mu'].transpose('chain', 'draw', '__obs__').values
            prediction_checks[name] = {'shape': list(observed.shape), 'all_finite': bool(np.isfinite(observed).all()),
                'max_abs_error': float(np.max(np.abs(expected - observed))), 'atol': 1e-10, 'rtol': 1e-10,
                'passed': bool(np.allclose(observed, expected, atol=1e-10, rtol=1e-10))}
        fit = SimpleNamespace(idata=idata, wall_seconds=receipt['fit_wall_seconds'], **receipt['fit_metadata'])
        record = ab.fit_record(model, fit, max_treedepth=10)
        write(f'{backend}-fit-record-posthoc.json', record)
        gates = {
            'original_formula': model.spec.formula == config['formula'] and record['model']['formula'] == config['formula'],
            'no_intercept_graph_names': 'Intercept' not in model.pymc_model.named_vars and 'intercept' not in model.names,
            'no_intercept_posterior': 'Intercept' not in idata.posterior and 'Intercept' not in record['posterior'],
            'no_intercept_prior': model.priors.intercept is None and 'intercept' not in record['model']['priors'],
            'record_intercept_absent': record['model']['intercept'] is False and record['model']['intercept_prior_applies_to'] is None,
            'requested_effective_centering': model.center_predictors is True and model.effective_center_predictors is False
                 and not model.design.centered and record['model']['center_predictors_requested'] is True
                 and record['model']['center_predictors'] is False,
            'raw_design_and_means': bool(np.array_equal(model.design.X_raw, design(train))
                 and np.array_equal(model.design.X_model, design(train)) and np.array_equal(model.design.x_means, np.zeros(4))),
            'coefficient_order': list(model.spec.columns) == config['coefficient_labels']
                 and record['model']['design_columns'] == config['coefficient_labels'],
            'coordinate_order': list(post.coords['arm_dim'].values) == ['control', 'treated']
                 and list(post.coords['dose:arm_dim'].values) == ['treated'],
            'exact_record_priors': record['model']['priors'] == config['record_priors'],
            'record_five_parameters': set(record['posterior']) == set(config['parameter_labels']),
            'full_level_factor_record': record['model']['categorical'] == {'arm': {'reference':'control',
                 'levels':['control','treated'],'coefficients':['control','treated'],'unused_levels':[],'coding':'full_level'}},
            'complete_retained_arrays': all(array.shape == (4, 1000) and np.isfinite(array).all() for array in arrays.values()),
            'mean_predictions': all(row['passed'] and row['all_finite'] for row in prediction_checks.values()),
        }
        gates = {key: bool(value) for key, value in gates.items()}
        mask = idata.sample_stats['diverging'].transpose('chain', 'draw').values.astype(bool)
        probs = [0., .01, .05, .25, .5, .75, .95, .99, 1.]
        summaries = {}
        for label, values in arrays.items():
            summaries[label] = {}
            for selection, selected in [('divergent_transition', values[mask]), ('nondivergent_transition', values[~mask])]:
                summaries[label][selection] = {'n': int(selected.size), 'mean': float(selected.mean()),
                    'quantiles': {str(prob): float(value) for prob, value in zip(probs, np.quantile(selected, probs))}}
        divergence_rows = []
        for chain, draw in np.argwhere(mask):
            row = {'chain': int(post.coords['chain'].values[chain]), 'draw': int(post.coords['draw'].values[draw]),
                'parameters': {label: float(array[chain, draw]) for label, array in arrays.items()},
                'sigma_empirical_cdf_all_draws': float(np.mean(arrays['sigma'] <= arrays['sigma'][chain, draw]))}
            row['sampler_statistics'] = {name: float(idata.sample_stats[name].transpose('chain', 'draw').values[chain, draw])
                for name in ['energy', 'energy_error', 'max_energy_error', 'tree_depth', 'n_steps', 'acceptance_rate']
                if name in idata.sample_stats}
            divergence_rows.append(row)
        sigma = arrays['sigma']
        sigma_location = {str(prob): {'all_draw_sigma_quantile': float(np.quantile(sigma, prob)),
            'divergent_draws_at_or_below': int((sigma[mask] <= np.quantile(sigma, prob)).sum()),
            'nondivergent_draws_at_or_below': int((sigma[~mask] <= np.quantile(sigma, prob)).sum())}
            for prob in [.01, .05, .1, .25, .5]}
        result = {'input_netcdf': receipt['path'], 'input_sha256': receipt['sha256'], 'semantic_gates': gates,
            'semantics_passed': all(gates.values()), 'predictions': prediction_checks,
            'divergences': int(mask.sum()), 'divergences_by_chain': mask.sum(axis=1).astype(int).tolist(),
            'conditional_draw_summaries': summaries, 'sigma_location': sigma_location, 'divergent_transition_rows': divergence_rows,
            'interpretation_limit': 'Conditional descriptions of retained endpoints marked by divergent transitions; not the entire divergent trajectories. Association cannot identify a cause or make the failed acceptance pass.'}
        write(f'{backend}-semantics-and-divergences.json', result)
        results[backend] = result
        print(json.dumps({'backend':backend,'semantics_passed':result['semantics_passed'],'divergences':result['divergences'],
                         'predictions':prediction_checks}), flush=True)
    _, final = verify()
    root_review = Path('/Users/andrewdou/iqss-research/artifacts/2026-09-19-autobayes-intercept-cpu-validation/autobayes-intercept-retained-review.json')
    result = {'finished': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'head': HEAD,
        'original_acceptance_remains_failed': True, 'new_posterior_fits': 0, 'new_response_draws': 0,
        'semantics_passed': all(row['semantics_passed'] for row in results.values()), 'backends': results,
        'initial_verification': initial, 'final_verification': final,
        'root_diagnostic_review': {'path':str(root_review),'sha256':sha(root_review)}}
    write('retained-semantic-review.json', result)
    return 0 if result['semantics_passed'] else 4

if __name__ == '__main__':
    try:
        code = main()
    except Exception as exc:
        write('execution-error.json', {'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        traceback.print_exc()
        code = 2
    raise SystemExit(code)
