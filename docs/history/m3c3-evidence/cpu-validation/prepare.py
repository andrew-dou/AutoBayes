"""Generate the prescribed residuals once and freeze inputs before any model build."""
from pathlib import Path
from importlib import metadata
import ast
import csv
import datetime
import hashlib
import json
import shutil
import subprocess
import numpy as np

ROOT = Path('/private/tmp/autobayes-multiple-intercepts-cpu-validation-1')
REPO = Path('/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/m3c3-multiple-intercepts')
HEAD = '40613ed44db44a7ab84af46b38cad0421f49eb25'
DISPATCH = Path('/Users/andrewdou/iqss-research/memory/dispatches/2026-09-20-autobayes-multiple-intercepts-cpu-validation.txt')
PLAN = Path('/Users/andrewdou/iqss-research/plans/2026-09-20-autobayes-multiple-intercepts-cpu-validation.md')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name, value):
    with (ROOT / name).open('x') as out: json.dump(value, out, indent=2, allow_nan=False); out.write('\n')

assert sha(DISPATCH) == 'a782617fa60d8c85a3acbe0c9ff8498eacd43d51e41891aa818875016ea1f10d'
assert sha(PLAN) == 'd3f24425812d2ea4d4baafdd53bb0f0ab5e2c2ca433c3778a42efc9a54951169'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == HEAD
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
assert not (ROOT / 'data.json').exists() and not (ROOT / 'freeze.json').exists()
for name in ['run.py', 'supervisor.py', 'prepare.py']: ast.parse((ROOT / name).read_text())
residuals = np.random.default_rng(731).normal(0, .7, 60)
county_effects = [-.6, -.35, -.1, .15, .35, .55]
year_effects = [-.3, -.1, 0., .15, .25]
data = []
for county in range(6):
    for year in range(5):
        for replicate, x in enumerate([-.75, .75]):
            y = .4 + .6*x + county_effects[county] + year_effects[year] + residuals[len(data)]
            data.append({'county': f'C{county}', 'year': f'Y{year}', 'replicate': replicate, 'x': x, 'y': float(y)})
write('data.json', data)
with (ROOT / 'data.csv').open('x', newline='') as out:
    writer = csv.DictWriter(out, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
write('generation.json', {'seed': 731, 'intercept': .4, 'slope': .6, 'residual_sd': .7,
    'county_effects': county_effects, 'year_effects': year_effects, 'residuals': residuals.tolist(),
    'row_order': ['county', 'year', 'replicate'], 'generator': 'numpy.random.default_rng(731).normal(0,0.7,60)'})
config = {'formula': 'y ~ x + (1 | county) + (1 | year)',
    'levels': {'county': [f'C{i}' for i in range(6)], 'year': [f'Y{i}' for i in range(5)]},
    'center_predictors': True, 'new_row_positions': [59, 0, 31, 12, 47, 8],
    'scalar_parameters': ['Intercept', 'x', '1|county_sigma', '1|year_sigma', 'sigma'],
    'record_priors': {'intercept': 'Normal(mu=0, sigma=1.5)', 'b:x': 'Normal(mu=0, sigma=1)',
        'sd:county': 'HalfNormal(sigma=0.8)', 'sd:year': 'Exponential(lam=2)', 'sigma': 'HalfNormal(sigma=1)'},
    'fits': [dict(backend=backend, device='cpu', chains=4, draws=1000, tune=1000, seed=seed,
                  target_accept=.99, cores=4, progressbar=False) for backend, seed in [('pymc', 109), ('numpyro', 113)]],
    'unused_eigenvalue_placeholders': ['smallest_eigval', 'largest_eigval'],
    'core_sampler_statistics': {'pymc': ['diverging', 'energy', 'tree_depth', 'lp', 'acceptance_rate', 'step_size', 'n_steps'],
        'numpyro': ['diverging', 'energy', 'tree_depth', 'lp', 'acceptance_rate', 'step_size', 'n_steps']},
    'expected_versions': {name: metadata.version(name) for name in ['pymc', 'pytensor', 'bambi', 'formulae', 'arviz', 'numpyro', 'jax', 'jaxlib', 'numpy', 'pandas', 'scipy', 'xarray']},
    'depth_source': {'pymc': {'path': str(REPO / '.venv/lib/python3.11/site-packages/pymc/step_methods/hmc/nuts.py'), 'line': 132, 'parameter': 'max_treedepth', 'default': 10},
        'numpyro': {'path': str(REPO / '.venv/lib/python3.11/site-packages/numpyro/infer/hmc.py'), 'line': 925, 'parameter': 'max_tree_depth', 'default': 10},
        'method': 'Pinned constructor defaults and runtime signature assertions; no conflicting kwargs forwarded.'},
    'gates': {'oracle_precision_atol_rtol': 1e-9, 'oracle_gradient_atol': 1e-8, 'rhat_max': 1.01,
        'ess_bulk_min': 400, 'ess_tail_min': 400, 'bfmi_min': .3, 'divergences': 0,
        'tree_depth_limit_exclusive': 10, 'five_scalar_mean_combined_mcse': 4,
        'prediction_atol_rtol': 1e-10},
    'env_overrides': {'PYTHONDONTWRITEBYTECODE': '1', 'JAX_PLATFORMS': 'cpu', 'AUTOBAYES_HOST_DEVICES': '4',
        'JAX_ENABLE_X64': 'true', 'PYTENSOR_FLAGS': f'floatX=float64,compiledir={ROOT}/pytensor-cache',
        'MPLCONFIGDIR': str(ROOT / 'mpl-cache'), 'NUMBA_CACHE_DIR': str(ROOT / 'numba-cache'),
        'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}}
write('config.json', config)
sources = [DISPATCH, PLAN, *sorted((REPO / 'autobayes').glob('*.py')), REPO / 'uv.lock', REPO / 'pyproject.toml',
           *[Path(config['depth_source'][b]['path']) for b in ['pymc', 'numpyro']]]
snapshots = ROOT / 'source-snapshots'; snapshots.mkdir()
copies = []
for i, path in enumerate(sources):
    copied = snapshots / f'{i:02d}-{path.name}'; shutil.copy2(path, copied); copies.append(copied)
local_inputs = [ROOT / name for name in ['data.json', 'data.csv', 'generation.json', 'config.json', 'run.py', 'supervisor.py', 'prepare.py']]
write('freeze.json', {'unit': 'autobayes-multiple-intercepts-cpu-validation-1', 'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'head': HEAD, 'dirty_state': '', 'inputs': [{'path': str(p), 'sha256': sha(p)} for p in sources + copies + local_inputs]})
print(json.dumps({'head': HEAD, 'data_json_sha256': sha(ROOT / 'data.json'), 'data_csv_sha256': sha(ROOT / 'data.csv'),
                  'config_sha256': sha(ROOT / 'config.json'), 'freeze_sha256': sha(ROOT / 'freeze.json'), 'rows': len(data)}, indent=2))
