import json
import bambi as bmb, formulae, numpy as np,pandas as pd
D=pd.DataFrame(dict(y=[1.,2,3,4,5,6], x=[-1.,0,1,2,3,5], z=pd.Categorical(['b','a','c','b','a','c'], categories=['b','a','c'],ordered=True), w=pd.Categorical(['v','u','u','v','v','u'],categories=['v','u'],ordered=True),g=['A','B','A','B','A','B']))
out=[]
for rhs in ['0+x','x-1','0+z','0+x+z','0+z+w','0+w+z','0+x+z+x:z','0+x+z+w+x:z','0+x+w+z+x:z','0+x+z+x:z+(1|g)']:
 for center in [False,True]:
  m=bmb.Model('y~'+rhs,D,center_predictors=center);m.build()
  c=m.components['mu'].design.common
  p=m.backend.model;pt=p.initial_point(); vs=p.replace_rvs_by_values([p['mu']]);fn=p.compile_fn(vs,inputs=p.value_vars,point_fn=True,on_unused_input='ignore')
  for key in pt:
   if key in c.terms:pt[key]=np.ones_like(pt[key])*.3
  matrix=np.asarray(c.design_matrix)
  actual=np.asarray(fn(pt)[0]);expected=matrix@np.full(matrix.shape[1],.3)
  out.append(dict(rhs=rhs,center_requested=center,terms={k:dict(labels=v.labels,levels=v.levels) for k,v in c.terms.items()},design=matrix.tolist(),variables=list(pt),raw_linear_predictor_error=float(abs(actual-expected).max())))
print(json.dumps(dict(bambi=bmb.__version__,formulae=formulae.__version__,cases=out),indent=2))
