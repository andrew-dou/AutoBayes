import json
from autobayes.formula import ModelSpec
s=ModelSpec('y ~ x + (1 | g)', 'y', ('x',), 'g', levels=('a','b'))
before=hash(s); keys={s:'fitted model'}
s.levels_by_group['g']=('replacement-a','replacement-b')
print(json.dumps({'hash_before':before,'hash_after':hash(s),'hash_changed':before!=hash(s),'dict_key_still_found':s in keys,'public_levels_after':s.levels,'type':type(s.levels_by_group).__name__}))
