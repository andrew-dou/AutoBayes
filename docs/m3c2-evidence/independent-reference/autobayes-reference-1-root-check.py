from pathlib import Path
import json,hashlib
import numpy as np
r=Path('/private/tmp/autobayes-intercept-reference-1');f=json.loads((r/'freeze.json').read_text());o=json.loads((r/'reference-result.json').read_text());q=json.loads((r/'quadrature-estimates.json').read_text());z=np.load(r/'quadrature-nodes.npz',allow_pickle=False)
for x in f['inputs']:assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'],x['path']
summary={};max_error=0.;evaluations=0
for row in q['rules']:
 name=row['name'];logw=z[name+'_log_quadrature_weights']+z[name+'_log_marginal_t'];m=logw.max();s=np.exp(logw-m);normal=m+np.log(s.sum());means=(s/s.sum())@z[name+'_conditional_means'];assert len(logw)==row['nodes'];evaluations+=len(logw)
 assert abs(normal-row['log_normalizer'])<1e-12
 for val,recorded in zip(means,row['means'].values()):assert abs(val-recorded)<1e-12;max_error=max(max_error,abs(float(val)-recorded))
 summary[name]={'log_normalizer':float(normal),'means':dict(zip(row['means'],map(float,means)))}
for label in ['primary128','primary256','expanded256']:
 for point in ['0.2','0.3','0.4','0.6','1.0']:
  cdf=np.exp(summary[label+'_cdf_'+point.replace('.','p')]['log_normalizer']-summary[label+'_full']['log_normalizer']);assert abs(cdf-o['estimates'][label]['sigma_cdf'][point])<1e-12
assert evaluations==3840==o['target_evaluations'];assert len(summary)==18
checks=0;flagged=[]
for backend,rows in o['retained_draw_comparisons'].items():
 for label in ['means','sigma_cdf']:
  for name,cell in rows[label].items():
   sample=cell.get('retained_mean',cell.get('retained_cdf'));delta=sample-cell['reference'];assert abs(delta-cell['signed_difference'])<1e-14
   mcse=cell.get('retained_mcse_mean',cell.get('indicator_mean_mcse'));assert abs(4*mcse-cell['four_mcse'])<1e-14
   assert (abs(delta)<=4*mcse)==cell['within_four_mcse_descriptive'];checks+=1
   if not cell['within_four_mcse_descriptive']:flagged.append([backend,label,name])
result={'frozen_input_hashes':len(f['inputs']),'saved_rules_reaggregated':len(summary),'stored_node_count':evaluations,'new_target_evaluations':0,'mean_aggregation_max_abs_error':max_error,'cdf_reaggregations':15,'comparison_arithmetic_checks':checks,'descriptive_flags':flagged,'decision':'Reference numerical stability and arithmetic supported; original sampler acceptance remains failed. Rare-event MCSE with a single NumPyro indicator hit is not a reliable tail-accuracy gate.'}
Path('/private/tmp/autobayes-reference-1-root-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
