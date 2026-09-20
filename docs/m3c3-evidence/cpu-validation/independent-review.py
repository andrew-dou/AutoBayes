from pathlib import Path
import json,hashlib,datetime,subprocess
import numpy as np,pandas as pd,arviz as az
import autobayes as ab
p=Path('/private/tmp/autobayes-multiple-intercepts-cpu-validation-1');out=Path('/private/tmp/autobayes-m3c3-cpu-root-review');out.mkdir(exist_ok=True)
config=json.loads((p/'config.json').read_text());frozen=json.loads((p/'freeze.json').read_text());reported=json.loads((p/'result.json').read_text());completion=json.loads((p/'completion.json').read_text())
for x in frozen['inputs']:assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'],x['path']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(ab.__file__).parent.parent,text=True).strip()==frozen['head']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=Path(ab.__file__).parent.parent,text=True)
frame=pd.DataFrame(json.loads((p/'data.json').read_text()));assert len(frame)==60
assert list(zip(frame.county,frame.year,frame.replicate))==[(f'C{c}',f'Y{y}',r) for c in range(6) for y in range(5) for r in range(2)]
assert np.array_equal(frame.x,np.tile([-.75,.75],30))
noise=np.random.default_rng(731).normal(0,.7,60)
expected_y=np.array([.4+.6*r.x+[-.6,-.35,-.1,.15,.35,.55][int(r.county[1:])]+[-.3,-.1,0,.15,.25][int(r.year[1:])]+noise[i] for i,r in frame.iterrows()])
assert np.array_equal(expected_y,frame.y.to_numpy())
Zc=np.column_stack([(frame.county==f'C{i}').to_numpy(float) for i in range(6)]);Zy=np.column_stack([(frame.year==f'Y{i}').to_numpy(float) for i in range(5)])
A=np.column_stack([np.ones(60),frame.x.to_numpy(),.45*Zc,.25*Zy]);Q=np.diag([1/1.5**2,1]+[1]*11)+A.T@A/.7**2;mean=np.linalg.solve(Q,A.T@frame.y.to_numpy()/.7**2)
oracle=json.loads((p/'oracle.json').read_text());idx={'Intercept':[0],'x':[1],'1|county_offset':list(range(2,8)),'1|year_offset':list(range(8,13))};order=np.concatenate([idx[n] for n in oracle['free_variable_names']]);assert np.array_equal(order,oracle['mapped_order'])
assert np.array_equal(A,oracle['A']) and np.allclose(Q,oracle['precision'],rtol=0,atol=1e-12) and np.allclose(mean,oracle['analytic_mean'],atol=1e-12,rtol=0)
h_error=float(np.max(np.abs(np.array(oracle['negative_hessian'])-Q[np.ix_(order,order)])));assert h_error<1e-9 and np.max(np.abs(oracle['gradient']))<1e-8
m=ab.gaussian('y ~ x + (1 | county) + (1 | year)',frame,ab.Priors(intercept=ab.Normal(0,1.5),b=ab.Normal(0,1),sd={'county':ab.HalfNormal(.8),'year':ab.Exponential(2)},sigma=ab.HalfNormal(1)),center_predictors=True)
assert np.array_equal(m.design.Z,np.column_stack([Zc,Zy]));assert m.spec.groups==('county','year')
metrics={};means={};predictions={};records={};draw_hashes={}
for backend in ['pymc','numpyro']:
 path=p/(backend+'.nc');draw_hashes[backend]=hashlib.sha256(path.read_bytes()).hexdigest();assert draw_hashes[backend]==completion['outputs'][backend]['sha256']
 idata=az.from_netcdf(path);idata.load();post=idata.posterior;stats=idata.sample_stats
 assert post.sizes['chain']==4 and post.sizes['draw']==1000
 assert all(np.isfinite(a.values).all() for a in post.data_vars.values())
 for name,a in stats.data_vars.items():
  if backend=='pymc' and name in ('smallest_eigval','largest_eigval') and np.isnan(a.values).all():continue
  assert np.isfinite(a.values).all(),(backend,name)
 for required in ['diverging','energy','tree_depth','lp','acceptance_rate','step_size','n_steps']:assert required in stats
 names=['Intercept','x','1|county_sigma','1|year_sigma','sigma','1|county','1|year'];summary=az.summary(idata,var_names=names,kind='all',round_to='none')
 assert len(summary)==16 and np.isfinite(summary[['mean','sd','mcse_mean','ess_bulk','ess_tail','r_hat']]).all().all()
 bfmi=np.asarray(az.bfmi(idata));r={'divergences':int(stats.diverging.sum()),'max_rhat':float(summary.r_hat.max()),'min_bulk_ess':float(summary.ess_bulk.min()),'min_tail_ess':float(summary.ess_tail.min()),'bfmi':bfmi.tolist(),'min_bfmi':float(bfmi.min()),'max_depth':int(stats.tree_depth.max())}
 assert r['divergences']==0 and r['max_rhat']<=1.01 and r['min_bulk_ess']>=400 and r['min_tail_ess']>=400 and r['min_bfmi']>=.3 and r['max_depth']<10
 if 'reached_max_treedepth' in stats:assert not stats.reached_max_treedepth.values.any()
 for k,v in [('max_rhat',r['max_rhat']),('min_ess_bulk',r['min_bulk_ess']),('min_ess_tail',r['min_tail_ess']),('min_bfmi',r['min_bfmi'])]:assert np.isclose(v,completion['diagnostic_extremes'][backend][k],rtol=1e-12,atol=1e-12)
 metrics[backend]=r;means[backend]={n:{'mean':float(summary.loc[n,'mean']),'mcse':float(summary.loc[n,'mcse_mean'])} for n in config['scalar_parameters']}
 for g,J in [('county',6),('year',5)]:assert post[f'1|{g}'].dims==('chain','draw',f'{g}__factor_dim') and list(post.coords[f'{g}__factor_dim'].values)==[('C' if g=='county' else 'Y')+str(i) for i in range(J)]
 assert np.array_equal(idata.observed_data['y'].values,frame.y.to_numpy())
 pred=[]
 for rows in [None,[59,0,31,12,47,8]]:
  f=frame if rows is None else frame.iloc[rows].reset_index(drop=True)
  expected=post.Intercept.values[...,None]+post.x.values[...,None]*f.x.to_numpy()
  expected+=post['1|county'].values[...,f.county.str[1:].astype(int).to_numpy()]
  expected+=post['1|year'].values[...,f.year.str[1:].astype(int).to_numpy()]
  actual=m.predict(idata,newdata=None if rows is None else f,kind='mean').mu.transpose('chain','draw','__obs__').values
  assert actual.shape==expected.shape and np.allclose(actual,expected,rtol=1e-10,atol=1e-10)
  pred.append({'rows':len(f),'values':actual.size,'max_abs_error':float(np.max(np.abs(actual-expected)))})
 predictions[backend]=pred
 rec=json.loads((p/(backend+'-fit-record.json')).read_text());json.dumps(rec,allow_nan=False)
 assert rec['model']['priors']=={'intercept':'Normal(mu=0, sigma=1.5)','b:x':'Normal(mu=0, sigma=1)','sd:county':'HalfNormal(sigma=0.8)','sd:year':'Exponential(lam=2)','sigma':'HalfNormal(sigma=1)'}
 assert [g['factor'] for g in rec['model']['groups']]==['county','year'] and 'group' not in rec['model'] and 'J' not in rec['model']
 for g in rec['model']['groups']:assert g['levels']==config['levels'][g['factor']] and g['J']==len(g['levels']) and g['noncentered']
 for k,v in {'chains':4,'draws':1000,'tune':1000,'target_accept':.99,'seed':109 if backend=='pymc' else 113,'backend':backend,'platform':'cpu','dtype':'float64'}.items():assert rec['execution'][k]==v,(k,rec['execution'][k])
 assert set(rec['posterior'])==set(config['scalar_parameters'])
 for n in config['scalar_parameters']:assert np.isclose(rec['posterior'][n]['mean'],means[backend][n]['mean'],atol=1e-12,rtol=1e-12)
 records[backend]=rec
comparison={}
for n in config['scalar_parameters']:
 a=means['pymc'][n];z=means['numpyro'][n];delta=abs(a['mean']-z['mean']);tol=4*np.hypot(a['mcse'],z['mcse']);assert delta<=tol;comparison[n]={'difference':delta,'four_combined_mcse':float(tol),'passed':True}
assert ab.compare_records(records['pymc'],records['numpyro'])['agree']
result={'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'head':frozen['head'],'frozen_inputs_verified':len(frozen['inputs']),'new_fits':0,'reconstructed_model_builds':1,'oracle_reaggregation_max_precision_error':h_error,'oracle_fresh_graph_evaluation':False,'original_data_seed_and_rows_match':True,'draw_hashes':draw_hashes,'diagnostics':metrics,'mean_comparison':comparison,'prediction_checks':predictions,'total_prediction_values':sum(x['values'] for a in predictions.values() for x in a),'saved_records_verified':True,'passed':True,'limitation':'One Gaussian CPU case; retained-draw recomputation and saved-oracle review, no calibration/GPU claim.'}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
