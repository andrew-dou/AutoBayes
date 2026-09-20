# PR10 intercept-control CPU acceptance: failed attempt

2026-09-19. Exact unchanged head: `6a58eb3c9b8ab510aa8bd5d3d04936e64990dace`.

The one authorized attempt executed both posterior fits exactly once and saved both InferenceData files. **This attempt did not pass acceptance.** Sampler logs report 3 post-tuning divergences for PyMC and 6 for NumPyro, violating the prespecified zero-divergence requirement. A separate analysis-runner error then prevented completion of the five-parameter diagnostic, comparison, and semantic tables. No repair, retry, extra fit, new seed, or threshold change was performed.

## Prespecified execution and retained evidence

The example, six observations, priors, four coefficient labels plus sigma, settings, scripts, locked package versions, and 19 source/script input hashes were frozen before launch in `freeze.json`. SHA-256: `b4872cbbe9ce8759b0391e48a4fa170e48948fc43f3077f2741855ae8a1cb996`.

Both CPU fits used four chains, 1000 tuning plus 1000 retained draws per chain, target acceptance 0.9, and inspected maximum-tree-depth defaults of 10. PyMC used seed 37 and four cores; NumPyro used seed 41 and parallel CPU chains. Pinned depth sources are `pymc/step_methods/hmc/nuts.py:132` (PyMC 5.28.5) and `numpyro/infer/hmc.py:925` (NumPyro 0.21.0), with hashes and runtime-signature verification recorded. No conflicting depth keyword was forwarded.

Outer command: `/usr/bin/python3 /private/tmp/autobayes-intercept-cpu-validation/supervisor.py`.

One exclusive `attempt-1` directory; 600-second child-process-group deadline and 3-second termination grace. Actual wall time **53.160824416 seconds**, no timeout, child and supervisor exit **2**. Launch at 2026-09-19T20:18:04.606021Z; finish at 20:18:57.767164Z. Exact argv, environment, timings, and return codes: `attempt-1/exit-receipt.json`.

| Backend | Recorded fit wall seconds | Retained file | SHA-256 |
|---|---:|---|---|
| PyMC | 20.10914212500211 | `attempt-1/pymc.nc` | `041046187d8edc38ccbfc85a04e09951dc83d0397c89c67956d262a94082a3ee` |
| NumPyro | 2.8630223749787547 | `attempt-1/numpyro.nc` | `db33a43459b91da6bb7154a29a1acdfd357b0b644d8b3230dcb966b558a220a7` |

Files were serialized and hashed immediately after each fit. Per-backend netCDF receipts and append-only events retain their precise metadata. `attempt-1/environment.json` records versions and CPU/float64 environment; `attempt-1/model-snapshot.json` records the actual model, priors, raw design, and absent intercept before fitting.

## Separate analysis-runner failure

After reopening the retained PyMC file, `acceptance.py:88` attempted `float(az.mcse(value, method='mean'))`. The returned non-scalar NumPy array cannot be converted using `float` in this stack. The resulting `TypeError: only 0-dimensional arrays can be converted to Python scalars` is preserved in `attempt-1/execution-error.json` and stderr. This is a defect in this external acceptance script, not a confirmed AutoBayes defect. The failure was recorded correctly as exit 2; it was not hidden by catching SystemExit.

The script stopped before writing per-backend acceptance results, fit records, named-parameter diagnostic tables, or backend-mean comparisons. Therefore those checks are **unresolved**, not passed or failed on their numerical merits. The logged divergence counts have not been independently recomputed in this worker after the stop. Neither semantic agreement nor a semantic regression can be claimed from this failed analysis stage.

The frozen scripts and original evidence remain untouched. A separately authorized read-only evaluation of the retained draws could finish the unchanged gates without any additional sampling. The known divergence failures would remain failures. No merge was attempted. This small Gaussian example cannot establish calibration, all-family sampling behavior, or GPU validity.
