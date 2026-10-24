"""Single fixed-rule Gaussian reference; no modeling-library imports or sampling."""
from pathlib import Path
import builtins
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from importlib import metadata

ROOT = Path('/private/tmp/autobayes-intercept-reference-1')
REPO = Path('/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/m3c2-intercept-controls')
EVALUATIONS = 0
FORBIDDEN = {'autobayes', 'bambi', 'pymc', 'numpyro'}
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in FORBIDDEN:
        raise RuntimeError('Modeling-library import forbidden: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name, value):
    with (ROOT / name).open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False); out.write('\n')
def verify(frozen):
    for entry in frozen['inputs']:
        assert sha(entry['path']) == entry['sha256'], entry['path']
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip()
    status = subprocess.check_output(['git','status','--porcelain'], cwd=REPO, text=True)
    assert head == frozen['head'] and not status, (head,status)
    return {'head':head,'status_porcelain':status,'input_hashes_verified':len(frozen['inputs'])}

def main():
    global EVALUATIONS
    frozen = json.loads((ROOT / 'freeze.json').read_text())
    before = verify(frozen)
    import numpy as np
    from scipy.linalg import cho_solve
    from scipy.special import logsumexp
    import arviz as az
    config = frozen['config']
    expected_versions = config['expected_versions']
    versions = {pkg:metadata.version(pkg) for pkg in expected_versions}
    assert versions == expected_versions
    write('environment.json', {'python':sys.version,'platform':platform.platform(),'versions':versions,
                               'forbidden_imports':sorted(FORBIDDEN),'verification':before})
    y = np.array([.2,1.1,.8,2.,1.5,2.2])
    dose = np.array([0.,1.,2.,0.,1.,2.])
    control = np.array([1.,1.,1.,0.,0.,0.]); treated = 1. - control
    X = np.column_stack([dose, control, treated, dose * treated])
    D = np.diag([1.,4.,4.,1.])
    assert y.tolist() == config['y'] and X.tolist() == config['X'] and np.diag(D).tolist() == config['D_diagonal']
    frozen_model = json.loads(Path(config['original_freeze']).read_text())['config']
    assert config['parameter_labels'] == frozen_model['parameter_labels']
    assert config['data'] == frozen_model['data'] and config['record_priors'] == frozen_model['record_priors']
    assert [row['y'] for row in config['data']] == y.tolist()
    assert [row['dose'] for row in config['data']] == dose.tolist()
    assert [row['arm'] for row in config['data']] == ['control']*3 + ['treated']*3
    kernel = X @ D @ X.T
    DXt = D @ X.T
    eye = np.eye(6)
    nodes = {n:np.polynomial.legendre.leggauss(n) for n in (128,256)}
    saved_nodes, rules = {}, []

    def target(t):
        global EVALUATIONS
        if EVALUATIONS >= config['hard_evaluation_cap']:
            raise RuntimeError('Hard marginal-target evaluation cap reached')
        EVALUATIONS += 1
        sigma = np.exp(t)
        chol = np.linalg.cholesky(sigma*sigma*eye + kernel)
        solved = cho_solve((chol,True), y, check_finite=False)
        logdet = 2*np.log(np.diag(chol)).sum()
        log_y = -.5*(6*np.log(2*np.pi) + logdet + y @ solved)
        log_sigma_prior = .5*np.log(2/np.pi) - .5*sigma*sigma
        log_t_density = log_y + log_sigma_prior + t
        conditional_means = np.concatenate([DXt @ solved, [sigma]])
        assert np.isfinite(log_t_density) and np.isfinite(conditional_means).all()
        return float(log_t_density), conditional_means

    def rule(name, lower, upper, n):
        points, weights = nodes[n]
        lo, hi = np.log(lower), np.log(upper)
        t = (lo+hi)/2 + (hi-lo)/2*points
        log_qweights = np.log(weights) + np.log((hi-lo)/2)
        evaluated = [target(float(point)) for point in t]
        log_density = np.array([item[0] for item in evaluated])
        means = np.stack([item[1] for item in evaluated])
        log_weighted = log_qweights + log_density
        log_z = float(logsumexp(log_weighted))
        normalized_weights = np.exp(log_weighted-log_z)
        integrated_means = normalized_weights @ means
        assert np.isfinite(log_z) and np.isfinite(integrated_means).all()
        assert np.isclose(normalized_weights.sum(),1.,atol=1e-14,rtol=1e-14)
        result = {'name':name,'sigma_interval':[lower,upper],'nodes':n,'log_normalizer':log_z,
            'means':dict(zip(config['parameter_labels'],map(float,integrated_means))),
            'min_log_integrand':float(log_density.min()),'max_log_integrand':float(log_density.max()),
            'weights_underflowing_to_zero':int((normalized_weights==0).sum()),
            'cumulative_target_evaluations':EVALUATIONS}
        for key,array in [('log_sigma',t),('log_quadrature_weights',log_qweights),('log_marginal_t',log_density),('conditional_means',means)]:
            saved_nodes[name+'_'+key] = array
        rules.append(result)
        with (ROOT/'rules.jsonl').open('a') as out: out.write(json.dumps(result,allow_nan=False)+'\n')
        print(json.dumps({'rule':name,'evaluations':EVALUATIONS,'log_normalizer':log_z}),flush=True)
        return result

    estimates = {}
    for spec in config['rules']:
        label, lower, upper, n = spec['label'], spec['lower'], spec['upper'], spec['nodes']
        full = rule(label+'_full',lower,upper,n)
        cdfs = {}
        for point in config['cdf_points']:
            numerator = rule(label+'_cdf_'+str(point).replace('.','p'),lower,point,n)
            cdf = float(np.exp(numerator['log_normalizer']-full['log_normalizer']))
            assert np.isfinite(cdf) and 0 <= cdf <= 1
            cdfs[str(point)] = cdf
        estimates[label] = {'log_normalizer':full['log_normalizer'],'means':full['means'],'sigma_cdf':cdfs}
    assert EVALUATIONS == config['expected_target_evaluations'] == 3840 and len(rules) == 18
    np.savez(ROOT/'quadrature-nodes.npz',**saved_nodes)
    write('quadrature-estimates.json', {'estimates':estimates,'evaluations':EVALUATIONS,'rules':rules})
    stability = {}
    for left,right in [('primary128','primary256'),('primary256','expanded256')]:
        a,b = estimates[left],estimates[right]
        differences = {'log_normalizer':abs(a['log_normalizer']-b['log_normalizer']),
            **{'mean:'+key:abs(a['means'][key]-b['means'][key]) for key in config['parameter_labels']},
            **{'cdf:'+str(point):abs(a['sigma_cdf'][str(point)]-b['sigma_cdf'][str(point)]) for point in config['cdf_points']}}
        stability[left+'_vs_'+right] = {'absolute_changes':differences,'max_absolute_change':max(differences.values()),
            'passed':all(value<=config['stability_atol'] for value in differences.values())}
    reference = estimates['expanded256']
    root_review = json.loads(Path(config['root_review']).read_text())
    comparisons = {}
    for backend in ['pymc','numpyro']:
        receipt = json.loads(Path(config['netcdf_receipts'][backend]).read_text())
        assert sha(receipt['path']) == receipt['sha256'] == root_review['backends'][backend]['sha256']
        idata = az.from_netcdf(receipt['path']); post = idata.posterior
        arrays = {'dose':post['dose'].transpose('chain','draw').values,
            'arm[control]':post['arm'].sel(arm_dim='control').transpose('chain','draw').values,
            'arm[treated]':post['arm'].sel(arm_dim='treated').transpose('chain','draw').values,
            'dose:arm[treated]':post['dose:arm'].sel({'dose:arm_dim':'treated'}).transpose('chain','draw').values,
            'sigma':post['sigma'].transpose('chain','draw').values}
        assert all(array.shape==(4,1000) and np.isfinite(array).all() for array in arrays.values())
        divergences = int(idata.sample_stats['diverging'].sum())
        assert divergences == root_review['backends'][backend]['divergences']
        means,cdfs = {},{}
        for label,array in arrays.items():
            prior_review = root_review['backends'][backend]['parameters'][label]
            assert abs(float(array.mean())-prior_review['mean'])<=1e-12
            mcse = prior_review['mcse_mean']; difference = prior_review['mean']-reference['means'][label]
            means[label] = {'reference':reference['means'][label],'retained_mean':prior_review['mean'],'retained_mcse_mean':mcse,
                'signed_difference':difference,'four_mcse':4*mcse,'within_four_mcse_descriptive':abs(difference)<=4*mcse}
        for point in config['cdf_points']:
            indicators = (arrays['sigma']<=point).astype(float)
            extracted = np.asarray(az.mcse(indicators,method='mean'))
            assert extracted.size == 1
            mcse = float(extracted.item()); empirical = float(indicators.mean())
            assert np.isfinite(mcse)
            difference = empirical-reference['sigma_cdf'][str(point)]
            cdfs[str(point)] = {'reference':reference['sigma_cdf'][str(point)],'retained_cdf':empirical,'indicator_mean_mcse':mcse,
                'signed_difference':difference,'four_mcse':4*mcse,'within_four_mcse_descriptive':abs(difference)<=4*mcse,
                'indicator_count':int(indicators.sum()),'indicator_shape':list(indicators.shape)}
        comparisons[backend] = {'netcdf_sha256':receipt['sha256'],'divergences_retained':divergences,'means':means,'sigma_cdf':cdfs}
    assert not any(name.split('.')[0] in FORBIDDEN for name in sys.modules)
    after = verify(frozen)
    passed = all(row['passed'] for row in stability.values())
    result = {'finished':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reference_checks_passed':passed,
        'target_evaluations':EVALUATIONS,'hard_cap':config['hard_evaluation_cap'],'rule_count':len(rules),
        'reference_choice':'expanded256 (prespecified)','reference':reference,'estimates':estimates,'stability':stability,
        'retained_draw_comparisons':comparisons,'initial_integrity':before,'final_integrity':after,
        'new_sampling':False,'original_pr_acceptance_remains_failed':True,
        'limitations':['Fixed-rule/interval stability is empirical, not a rigorous tail or error bound.',
            'MCSE comparisons are descriptive and not valid uncertainty guarantees for divergent chains.',
            'No AutoBayes/Bambi/PyMC/NumPyro import or fit; original sampler and runner failures unchanged.']}
    write('reference-result.json',result)
    return 0 if passed else 4

if __name__ == '__main__':
    try: code=main()
    except Exception as exc:
        write('execution-error.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'target_evaluations':EVALUATIONS})
        traceback.print_exc();code=2
    raise SystemExit(code)
