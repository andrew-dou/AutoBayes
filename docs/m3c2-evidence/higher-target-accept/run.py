"""One prospective target_accept=.99 unit; exactly two fits, no retries."""
from pathlib import Path
from importlib import metadata
import datetime
import hashlib
import inspect
import json
import runpy
import subprocess
import time
import traceback

ROOT=Path('/private/tmp/autobayes-intercept-higher-accept-1')
REPO=Path('/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/m3c2-intercept-controls')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value): return str(value)
    return value
def write(name,value):
    with (ROOT/name).open('x') as out: json.dump(clean(value),out,indent=2,allow_nan=False);out.write('\n')
def event(kind,**fields):
    row={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':kind,**fields}
    with (ROOT/'events.jsonl').open('a') as out:out.write(json.dumps(clean(row),allow_nan=False)+'\n')
    print(json.dumps(clean(row),allow_nan=False),flush=True)
def verify(frozen):
    for entry in frozen['inputs']: assert sha(entry['path'])==entry['sha256'],entry['path']
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)
    assert head==frozen['head'] and not status,(head,status)
    return {'head':head,'status_porcelain':status,'hashes_verified':len(frozen['inputs'])}
def scalar(value):
    array=np.asarray(value)
    assert array.size==1,array.shape
    return float(array.item())
def named(idata):
    p=idata.posterior
    return {'dose':p.dose.transpose('chain','draw').values,
        'arm[control]':p.arm.sel(arm_dim='control').transpose('chain','draw').values,
        'arm[treated]':p.arm.sel(arm_dim='treated').transpose('chain','draw').values,
        'dose:arm[treated]':p['dose:arm'].sel({'dose:arm_dim':'treated'}).transpose('chain','draw').values,
        'sigma':p.sigma.transpose('chain','draw').values}

def diagnostics(idata,backend,config):
    arrays=named(idata);stats=idata.sample_stats
    pars={name:{'mean':float(a.mean()),'sd':float(a.std(ddof=1)),
        'rhat':scalar(az.rhat(a,method='rank')),'ess_bulk':scalar(az.ess(a,method='bulk')),
        'ess_tail':scalar(az.ess(a,method='tail')),'mcse_mean':scalar(az.mcse(a,method='mean'))}
        for name,a in arrays.items()}
    post_checks={name:{'shape':list(a.shape),'complete':a.sizes.get('chain')==4 and a.sizes.get('draw')==1000,
        'finite':bool(np.isfinite(a.values).all())} for name,a in idata.posterior.data_vars.items()}
    stat_checks={}
    for name,a in stats.data_vars.items():
        excluded=backend=='pymc' and name in config['unused_eigenvalue_placeholders'] and bool(np.isnan(a.values).all())
        stat_checks[name]={'shape':list(a.shape),'complete':a.sizes.get('chain')==4 and a.sizes.get('draw')==1000,
            'finite':bool(np.isfinite(a.values).all()),'excluded_unused_all_nan':excluded,
            'nonfinite_count':int((~np.isfinite(a.values)).sum())}
    bfmi=np.asarray(az.bfmi(idata));div=int(stats.diverging.sum());depth=int((stats.tree_depth>=10).sum())
    flags=int(stats.reached_max_treedepth.sum()) if 'reached_max_treedepth' in stats else 0
    finite=all(np.isfinite(value) for row in pars.values() for value in row.values()) and np.isfinite(bfmi).all()
    gates={'complete_finite_posterior':all(row['complete'] and row['finite'] for row in post_checks.values())
        and all(a.shape==(4,1000) and np.isfinite(a).all() for a in arrays.values()),
        'core_stats_present':all(name in stats for name in config['core_sampler_statistics'][backend]),
        'complete_finite_active_stats':all(row['complete'] and (row['finite'] or row['excluded_unused_all_nan']) for row in stat_checks.values()),
        'finite_diagnostics':finite,'zero_divergences':div==0,
        'rhat_le_1_01':all(np.isfinite(row['rhat']) and row['rhat']<=1.01 for row in pars.values()),
        'ess_bulk_ge_400':all(np.isfinite(row['ess_bulk']) and row['ess_bulk']>=400 for row in pars.values()),
        'ess_tail_ge_400':all(np.isfinite(row['ess_tail']) and row['ess_tail']>=400 for row in pars.values()),
        'every_chain_bfmi_ge_0_3':bfmi.shape==(4,) and np.isfinite(bfmi).all() and (bfmi>=.3).all(),
        'no_depth_saturation':depth==0,'no_max_depth_flags':flags==0}
    gates={key:bool(value) for key,value in gates.items()}
    return {'parameters':pars,'posterior_checks':post_checks,'sample_stats_checks':stat_checks,'bfmi':bfmi.tolist(),
        'divergences':div,'divergences_per_chain':stats.diverging.sum('draw').values.tolist(),
        'max_depth':int(stats.tree_depth.max()),'depth_saturated_draws':depth,'max_depth_flags':flags,
        'gates':gates,'passed':all(gates.values())}

def semantics(model,fit,config):
    import pandas as pd
    train=pd.DataFrame(config['data']);new=pd.DataFrame(config['newdata']);post=fit.idata.posterior
    arrays=named(fit.idata)
    def design(frame):
        dose=frame.dose.to_numpy(dtype=float);control=(frame.arm.to_numpy()=='control').astype(float);treated=(frame.arm.to_numpy()=='treated').astype(float)
        return np.column_stack([dose,control,treated,dose*treated])
    beta=np.stack([arrays[label] for label in config['coefficient_labels']],axis=-1)
    predictions={}
    for label,frame,data_arg in [('fitted',train,None),('new',new,new)]:
        expected=np.einsum('cdp,np->cdn',beta,design(frame))
        actual=model.predict(fit.idata,newdata=data_arg,kind='mean')['mu'].transpose('chain','draw','__obs__').values
        predictions[label]={'shape':list(actual.shape),'max_abs_error':float(np.max(np.abs(actual-expected))),
            'finite':bool(np.isfinite(actual).all()),'passed':bool(np.allclose(actual,expected,atol=1e-10,rtol=1e-10))}
    record=ab.fit_record(model,fit,max_treedepth=10);write(f'{fit.backend}-fit-record.json',record)
    gates={
        'original_formula':model.spec.formula==config['formula'] and record['model']['formula']==config['formula'],
        'no_intercept_graph_names':'Intercept' not in model.pymc_model.named_vars and 'intercept' not in model.names,
        'no_intercept_posterior':'Intercept' not in post and 'Intercept' not in record['posterior'],
        'no_intercept_prior':model.priors.intercept is None and 'intercept' not in record['model']['priors'],
        'record_intercept_absent':record['model']['intercept'] is False and record['model']['intercept_prior_applies_to'] is None,
        'requested_effective_centering':model.center_predictors is True and model.effective_center_predictors is False and not model.design.centered
            and record['model']['center_predictors_requested'] is True and record['model']['center_predictors'] is False,
        'raw_design_and_means':np.array_equal(model.design.X_raw,design(train)) and np.array_equal(model.design.X_model,design(train))
            and np.array_equal(model.design.x_means,np.zeros(4)),
        'coefficient_order':list(model.spec.columns)==config['coefficient_labels'] and record['model']['design_columns']==config['coefficient_labels'],
        'coordinate_order':list(post.coords['arm_dim'].values)==['control','treated'] and list(post.coords['dose:arm_dim'].values)==['treated'],
        'exact_record_priors':record['model']['priors']==config['record_priors'],
        'record_five_parameters':set(record['posterior'])==set(config['parameter_labels']),
        'full_level_factor_record':record['model']['categorical']=={'arm':{'reference':'control','levels':['control','treated'],
            'coefficients':['control','treated'],'unused_levels':[],'coding':'full_level'}},
        'complete_retained_arrays':all(a.shape==(4,1000) and np.isfinite(a).all() for a in arrays.values()),
        'mean_predictions':all(row['passed'] and row['finite'] for row in predictions.values())}
    gates={key:bool(value) for key,value in gates.items()}
    assert len(gates)==14
    return {'gates':gates,'predictions':predictions,'passed':all(gates.values())}

def main():
    global np,az,ab
    frozen=json.loads((ROOT/'freeze.json').read_text());before=verify(frozen);config=frozen['config']
    original=json.loads(Path(config['original_freeze']).read_text())['config']
    for old,new in zip(original['fits'],config['fits']):
        assert set(old)==set(new) and [key for key in old if old[key]!=new[key]]==['target_accept']
        assert old['target_accept']==.9 and new['target_accept']==.99
    import numpy as np
    import arviz as az
    import autobayes as ab
    import pymc as pm
    from numpyro.infer import NUTS
    assert Path(ab.__file__).resolve().parent.parent==REPO
    versions={name:metadata.version(name) for name in config['expected_versions']};assert versions==config['expected_versions']
    assert inspect.signature(pm.NUTS.__init__).parameters['max_treedepth'].default==10
    assert inspect.signature(NUTS.__init__).parameters['max_tree_depth'].default==10
    write('environment.json',{'versions':versions,'environment':ab.environment(),'source_verification':before,
        'pymc_nuts_signature':str(inspect.signature(pm.NUTS.__init__)),'numpyro_nuts_signature':str(inspect.signature(NUTS.__init__))})
    model=runpy.run_path(str(REPO/'examples/m3c2_no_intercept.py'),run_name='higher_accept_example')['build_example']()
    assert model.data.to_dict(orient='records')==config['data'] and model.spec.formula==config['formula']
    priors={f'b:{name}':str(prior) for name,prior in model.priors.coefficient_priors(model.spec).items()};priors['sigma']=str(model.priors.sigma)
    assert priors==config['record_priors'] and list(model.spec.columns)==config['coefficient_labels'] and model.priors.intercept is None
    write('model-snapshot.json',{'data':model.data.to_dict(orient='records'),'formula':model.spec.formula,'priors':priors,
        'columns':list(model.spec.columns),'X_raw':model.design.X_raw.tolist(),'describe':model.describe()})
    fits,outputs={},{}
    for settings in config['fits']:
        backend=settings['backend'];event('fit_started',backend=backend,kwargs=settings)
        started=time.monotonic();fit=model.fit(**settings);fits[backend]=fit
        path=ROOT/f'{backend}.nc';assert not path.exists();fit.idata.to_netcdf(str(path))
        outputs[backend]={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'fit_wall_seconds':fit.wall_seconds,
            'fit_and_serialization_seconds':time.monotonic()-started,'metadata':{key:getattr(fit,key) for key in
            ['backend','device_requested','device_actual','platform','dtype','chain_method','chains','draws','tune','target_accept','seed','notes']}}
        write(f'{backend}-netcdf-receipt.json',outputs[backend]);event('fit_saved',backend=backend,**outputs[backend])
    ref_document=json.loads(Path(config['reference_result']).read_text());reference=ref_document['reference']
    assert ref_document['reference_choice']=='expanded256 (prespecified)' and ref_document['reference_checks_passed']
    results={}
    for backend,fit in fits.items():
        fit.idata=az.from_netcdf(outputs[backend]['path']);fit.idata.posterior.load();fit.idata.sample_stats.load()
        diag=diagnostics(fit.idata,backend,config);semantic=semantics(model,fit,config);arrays=named(fit.idata)
        reference_means={};cdf={}
        for label,row in diag['parameters'].items():
            diff=abs(row['mean']-reference['means'][label]);band=4*row['mcse_mean']+1e-8
            reference_means[label]={'reference':reference['means'][label],'mean':row['mean'],'mcse_mean':row['mcse_mean'],
                'absolute_difference':diff,'tolerance':band,'passed':bool(np.isfinite(diff) and np.isfinite(band) and diff<=band)}
        for point in [.2,.3,.4,.6,1.]:
            indicator=(arrays['sigma']<=point).astype(float);prob=float(indicator.mean());mcse=scalar(az.mcse(indicator,method='mean'))
            cdf[str(point)]={'reference':reference['sigma_cdf'][str(point)],'empirical':prob,'indicator_count':int(indicator.sum()),
                'chain_shape':list(indicator.shape),'indicator_mean_mcse':mcse,'four_mcse':4*mcse,
                'signed_difference':prob-reference['sigma_cdf'][str(point)],'descriptive_only':True}
        expected=next(row for row in config['fits'] if row['backend']==backend)
        execution={'cpu_float64':fit.platform=='cpu' and fit.dtype=='float64',
            'settings_match':all(getattr(fit,key)==expected[key] for key in ['chains','draws','tune','target_accept','seed']),
            'parallel_numpyro':fit.chain_method=='parallel' if backend=='numpyro' else True,
            'retained_hash_matches':sha(outputs[backend]['path'])==outputs[backend]['sha256']}
        results[backend]={'diagnostics':diag,'semantics':semantic,'reference_means':reference_means,'sigma_cdf_descriptive':cdf,
            'execution_gates':execution,'passed':diag['passed'] and semantic['passed'] and all(execution.values()) and all(row['passed'] for row in reference_means.values())}
        write(f'{backend}-analysis.json',results[backend]);event('backend_analyzed',backend=backend,passed=results[backend]['passed'])
    comparison={}
    for label in config['parameter_labels']:
        a=results['pymc']['diagnostics']['parameters'][label];b=results['numpyro']['diagnostics']['parameters'][label]
        diff=abs(a['mean']-b['mean']);band=4*float(np.hypot(a['mcse_mean'],b['mcse_mean']))
        comparison[label]={'pymc_mean':a['mean'],'numpyro_mean':b['mean'],'absolute_difference':diff,
            'four_combined_mcse':band,'passed':bool(np.isfinite(diff) and np.isfinite(band) and diff<=band)}
    after=verify(frozen);passed=all(row['passed'] for row in results.values()) and all(row['passed'] for row in comparison.values())
    result={'finished':datetime.datetime.now(datetime.timezone.utc).isoformat(),'this_configuration_passed':passed,'posterior_fits':len(fits),
        'backends':results,'backend_mean_comparison':comparison,'outputs':outputs,'initial_integrity':before,'final_integrity':after,
        'reference_sha256':sha(config['reference_result']),'reference_target_evaluations':0,'new_response_draws':0,
        'historical_target_accept_0_9_failure_unchanged':True,'limitations':'One prospective configuration on one Gaussian development example; no calibration, broad-family/GPU validation, rare-tail guarantee, or automatic PR acceptance.'}
    write('result.json',result);event('finished',configuration_passed=passed,posterior_fits=len(fits))
    return 0 if passed else 4

if __name__=='__main__':
    try: code=main()
    except Exception as exc:
        # Use only the standard library here, including errors before numerical imports.
        with (ROOT/'execution-error.json').open('x') as out:
            json.dump({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()},out,indent=2);out.write('\n')
        traceback.print_exc();code=2
    raise SystemExit(code)
