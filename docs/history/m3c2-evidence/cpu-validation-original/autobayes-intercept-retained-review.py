from pathlib import Path
import json,hashlib
import numpy as np
import arviz as az
root=Path('/private/tmp/autobayes-intercept-cpu-validation')
r=root/'attempt-1'
expected={'pymc':'041046187d8edc38ccbfc85a04e09951dc83d0397c89c67956d262a94082a3ee','numpyro':'db33a43459b91da6bb7154a29a1acdfd357b0b644d8b3230dcb966b558a220a7'}
def scalar(x):
 a=np.asarray(x);assert a.size==1
 return a.item()
report={'review_type':'Independent retained-draw analysis; no new fits. Original attempt failed and remains unchanged.','backends':{}}
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
report['conclusion']='Failed zero-divergence gate in both backends; diagnostic failures are not proof of a library defect. Do not merge or claim validation passed. Retained-draw comparison is descriptive with divergent samples. Planned API semantic checks were not reached by the original runner.'
out=Path('/private/tmp/autobayes-intercept-retained-review.json');out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for k,row in report['backends'].items():print(k,{'divergences':row['divergences'],'max_rhat':max(v['rhat'] for v in row['parameters'].values()),'min_bulk_ess':min(v['ess_bulk'] for v in row['parameters'].values()),'min_tail_ess':min(v['ess_tail'] for v in row['parameters'].values()),'min_bfmi':min(row['bfmi']),'gates':row['gates']})
print('means_within_4MCSE',sum(x['passes'] for x in compare.values()),'of',len(compare))
print(out)
