from pathlib import Path
import json,hashlib
import numpy as np
import arviz as az
root=Path('/private/tmp/autobayes-intercept-higher-accept-1')
r=root
expected={'pymc':'35e4589927a113cd53413ce53b1d8072e4c479eaed1dc8b0a6a69b2ad0a9db66','numpyro':'83395a2419032ca4e13f08a51dd053087d212cf61332fcab9d78c130f5e81f39'}
def scalar(x):
 a=np.asarray(x);assert a.size==1
 return a.item()
report={'review_type':'Independent retained-draw review of the prospective0.99 configuration; no new fits. Historical0.9 failure remains unchanged.','backends':{}}
for backend,digest in expected.items():
 p=r/f'{backend}.nc';assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
 ds=az.from_netcdf(p);post=ds.posterior;stats=ds.sample_stats
 arrays={'dose':post.dose,'arm[control]':post.arm.sel(arm_dim='control'),'arm[treated]':post.arm.sel(arm_dim='treated'),'dose:arm[treated]':post['dose:arm'].sel({'dose:arm_dim':'treated'}),'sigma':post.sigma}
 params={}
 for name,x in arrays.items():
  a=x.transpose('chain','draw').values
  params[name]={'shape':list(a.shape),'finite':bool(np.isfinite(a).all()),'mean':scalar(a.mean()),'rhat':scalar(az.rhat(a,method='rank')),'ess_bulk':scalar(az.ess(a,method='bulk')),'ess_tail':scalar(az.ess(a,method='tail')),'mcse_mean':scalar(az.mcse(a,method='mean'))}
 finite_stats={k:bool(np.isfinite(v.values).all()) for k,v in stats.data_vars.items()}
 exclusions=[k for k,v in stats.data_vars.items() if backend=='pymc' and k in ['smallest_eigval','largest_eigval'] and np.isnan(v.values).all()]
 row={'sha256':digest,'parameters':params,'divergences':int(stats.diverging.sum()),'max_tree_depth':int(stats.tree_depth.max()),'depth_saturation':int((stats.tree_depth>=10).sum()),'bfmi':az.bfmi(ds).tolist(),'finite_sampler_statistics':finite_stats,'declared_unused_exclusions':exclusions,'no_intercept': 'Intercept' not in post}
 row['gates']={'complete_finite':all(v['shape']==[4,1000] and v['finite'] for v in params.values()),'zero_divergences':row['divergences']==0,'rhat_le_1_01':all(v['rhat']<=1.01 for v in params.values()),'bulk_ess_ge_400':all(v['ess_bulk']>=400 for v in params.values()),'tail_ess_ge_400':all(v['ess_tail']>=400 for v in params.values()),'bfmi_ge_0_3':all(v>=.3 for v in row['bfmi']),'no_depth_saturation':row['depth_saturation']==0,'active_stats_finite':all(v or k in exclusions for k,v in finite_stats.items())}
 report['backends'][backend]=row
compare={}
for name in report['backends']['pymc']['parameters']:
 a=report['backends']['pymc']['parameters'][name];b=report['backends']['numpyro']['parameters'][name]
 difference=abs(a['mean']-b['mean']);bound=4*np.hypot(a['mcse_mean'],b['mcse_mean'])
 compare[name]={'abs_difference':difference,'four_combined_mcse':float(bound),'passes':bool(difference<=bound)}
report['comparisons']=compare
reference=json.loads(Path('/private/tmp/autobayes-intercept-reference-1/reference-result.json').read_text())['reference']['means']
report['reference_mean_comparisons']={}
for b,row in report['backends'].items():
 report['reference_mean_comparisons'][b]={name:{'absolute_difference':abs(v['mean']-reference[name]),'tolerance':4*v['mcse_mean']+1e-8,'passed':abs(v['mean']-reference[name])<=4*v['mcse_mean']+1e-8} for name,v in row['parameters'].items()}
report['configuration_passed']=all(all(v['gates'].values()) for v in report['backends'].values()) and all(v['passes'] for v in compare.values()) and all(v['passed'] for r in report['reference_mean_comparisons'].values() for v in r.values())
assert report['configuration_passed']
report['conclusion']='All independently recomputed diagnostic and mean-comparison gates pass for target_accept0.99. Original0.9failure remains unchanged. This is one Gaussian example, not calibration, tail accuracy or GPU validation.'
out=Path('/private/tmp/autobayes-intercept-higher-accept-root-review.json');out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for k,row in report['backends'].items():print(k,{'divergences':row['divergences'],'max_rhat':max(v['rhat'] for v in row['parameters'].values()),'min_bulk_ess':min(v['ess_bulk'] for v in row['parameters'].values()),'min_tail_ess':min(v['ess_tail'] for v in row['parameters'].values()),'min_bfmi':min(row['bfmi']),'gates':row['gates']})
print('means_within_4MCSE',sum(x['passes'] for x in compare.values()),'of',len(compare))
print(out)
