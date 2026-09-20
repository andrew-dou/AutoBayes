# PR10 higher-target-accept CPU diagnostic — this configuration passed

Executed 2026-09-20 UTC (September 19 America/New_York) at unchanged head `6a58eb3c9b8ab510aa8bd5d3d04936e64990dace`.

**All prespecified gates passed for this single prospective target_accept=0.99 configuration.** This does not relabel the original target_accept=0.9 fits: their 3/6 divergences and original external analysis-runner TypeError remain immutable failures. No merge was performed; root independent review remains separate.

## Exact execution

Exactly two CPU float64 fits of the unchanged six-observation `examples/m3c2_no_intercept.py::build_example()`: PyMC then NumPyro, four chains, 1000 tuning and 1000 retained draws per chain, seeds 37/41, PyMC cores 4 and NumPyro parallel with four host devices. The script mechanically verified that **target_accept 0.9→0.99 was the only changed sampler keyword**. Formula, data, priors, labels, original four prediction rows, and max-depth defaults 10 stayed fixed. Pinned source hashes and runtime signatures verified the PyMC/NumPyro depth defaults; no incompatible depth keyword was forwarded.

One exclusive invocation under a 600-second process-group deadline (TERM and 3-second KILL grace). Actual wall time **48.110692916000005 seconds**, exit **0**, no timeout. Both complete netCDF files were serialized and hashed immediately after their fits, then reopened for every posterior diagnostic and semantic check. No retry, extra fit, new response draw, fresh reference evaluation, network, install, GPU run, or repository edit occurred.

| Backend | Fit wall seconds | Draw file | SHA-256 |
|---|---:|---|---|
| pymc | 21.173541500000283 | `pymc.nc` | `35e4589927a113cd53413ce53b1d8072e4c479eaed1dc8b0a6a69b2ad0a9db66` |
| numpyro | 2.8736890839936677 | `numpyro.nc` | `83395a2419032ca4e13f08a51dd053087d212cf61332fcab9d78c130f5e81f39` |

## Diagnostic values

| Backend | Divergences | Max rank R-hat | Min bulk ESS | Min tail ESS | Min chain BFMI | Max depth | Depth-10 saturation / flags |
|---|---:|---:|---:|---:|---:|---:|---|
| pymc | 0 | 1.00347463072 | 909.580641075 | 1670.28637129 | 0.704669114929 | 7 | 0 / 0 |
| numpyro | 0 | 1.00368148655 | 1013.78200915 | 1602.79212798 | 0.664013732769 | 7 | 0 / 0 |

All posterior arrays and active sampler statistics have complete 4×1000 chain/draw axes and finite values. Only PyMC’s previously declared, unused, entirely NaN `smallest_eigval`/`largest_eigval` fields were excluded. No active diagnostic was discarded.

| Backend / parameter | Mean | Mean MCSE | Rank R-hat | Bulk ESS | Tail ESS |
|---|---:|---:|---:|---:|---:|
| pymc / dose | 0.250699634592 | 0.00951932048803 | 1.00186657995 | 1532.04892091 | 2009.42751457 |
| pymc / arm[control] | 0.438246472341 | 0.0127632673651 | 1.00088143011 | 1675.73820405 | 1670.28637129 |
| pymc / arm[treated] | 1.65271679463 | 0.0128263165704 | 1.00347463072 | 1712.35198731 | 1767.7143218 |
| pymc / dose:arm[treated] | -0.0628058043472 | 0.0137046805828 | 1.00218801265 | 1278.02425309 | 1682.37564502 |
| pymc / sigma | 0.630007739789 | 0.00924393808556 | 1.00144144296 | 909.580641075 | 2035.05446729 |
| numpyro / dose | 0.272782032215 | 0.0097460909542 | 1.00054125908 | 1531.72150179 | 1602.79212798 |
| numpyro / arm[control] | 0.394097945908 | 0.0141845721158 | 1.00292042092 | 1470.72771579 | 1655.48481389 |
| numpyro / arm[treated] | 1.63953781147 | 0.0144619171992 | 1.00368148655 | 1608.56237633 | 1604.15067953 |
| numpyro / dose:arm[treated] | -0.0681395503438 | 0.0126775812445 | 1.00127624567 | 1523.84477348 | 2061.4775509 |
| numpyro / sigma | 0.650068351532 | 0.00982770264475 | 1.00344867804 | 1013.78200915 | 1786.22735943 |

## Mean comparisons

All five backend differences meet four combined mean MCSE. Each backend also meets `absolute difference ≤ 4×its mean MCSE + 1e-8` against the frozen expanded256 independent reference; the 1e-8 numerical allowance is not a rigorous error bound. No quadrature or target evaluations were repeated.

| Parameter | Backend absolute difference | Four combined MCSE | PyMC reference difference / tolerance | NumPyro reference difference / tolerance |
|---|---:|---:|---|---|
| dose | 0.022082397623 | 0.0544945870987 | 0.0116838259592 / 0.0380772919521 | 0.0103985716639 / 0.0389843738168 |
| arm[control] | 0.0441485264325 | 0.0763259410624 | 0.0181714811472 / 0.0510530794605 | 0.0259770452852 / 0.056738298463 |
| arm[treated] | 0.0131789831664 | 0.0773212980587 | 0.013708081257 / 0.0513052762818 | 0.000529098090593 / 0.0578476787969 |
| dose:arm[treated] | 0.00533374599664 | 0.0746768329362 | 0.00329197249902 / 0.0548187323311 | 0.00204177349762 / 0.0507103349779 |
| sigma | 0.0200606117425 | 0.053968009873 | 0.0164335868831 / 0.0369757623422 | 0.00362702485942 / 0.039310820579 |

## Semantic checks and full gate table

All 14 semantic checks pass per backend: absent Intercept in graph/names/prior/posterior/record, requested centering true and effective centering false, exact raw design, coefficient and coordinate labels, explicit priors, five named parameters, full-level coding, and mean predictions. All 80,000 compared mean values are finite; maximum error is 0 in-sample and 8.881784197001252e-16 on the frozen new rows, below the unchanged 1e-10 absolute/relative tolerances.

| Gate | PyMC | NumPyro |
|---|---|---|
| diagnostic: complete_finite_posterior | pass | pass |
| diagnostic: core_stats_present | pass | pass |
| diagnostic: complete_finite_active_stats | pass | pass |
| diagnostic: finite_diagnostics | pass | pass |
| diagnostic: zero_divergences | pass | pass |
| diagnostic: rhat_le_1_01 | pass | pass |
| diagnostic: ess_bulk_ge_400 | pass | pass |
| diagnostic: ess_tail_ge_400 | pass | pass |
| diagnostic: every_chain_bfmi_ge_0_3 | pass | pass |
| diagnostic: no_depth_saturation | pass | pass |
| diagnostic: no_max_depth_flags | pass | pass |
| semantic: original_formula | pass | pass |
| semantic: no_intercept_graph_names | pass | pass |
| semantic: no_intercept_posterior | pass | pass |
| semantic: no_intercept_prior | pass | pass |
| semantic: record_intercept_absent | pass | pass |
| semantic: requested_effective_centering | pass | pass |
| semantic: raw_design_and_means | pass | pass |
| semantic: coefficient_order | pass | pass |
| semantic: coordinate_order | pass | pass |
| semantic: exact_record_priors | pass | pass |
| semantic: record_five_parameters | pass | pass |
| semantic: full_level_factor_record | pass | pass |
| semantic: complete_retained_arrays | pass | pass |
| semantic: mean_predictions | pass | pass |
| execution: cpu_float64 | pass | pass |
| execution: settings_match | pass | pass |
| execution: parallel_numpyro | pass | pass |
| execution: retained_hash_matches | pass | pass |
| reference mean: dose | pass | pass |
| reference mean: arm[control] | pass | pass |
| reference mean: arm[treated] | pass | pass |
| reference mean: dose:arm[treated] | pass | pass |
| reference mean: sigma | pass | pass |

| Backend-agreement gate | Result |
|---|---|
| dose | pass |
| arm[control] | pass |
| arm[treated] | pass |
| dose:arm[treated] | pass |
| sigma | pass |

## Sigma CDF: descriptive only, no acceptance gate

| Sigma point | Numerical reference | PyMC probability (hits/4000) | PyMC indicator MCSE | NumPyro probability (hits/4000) | NumPyro indicator MCSE |
|---|---:|---|---:|---|---:|
| 0.2 | 0.00141271496819 | 0.00175 (7) | 0.00089304218664 | 0.00025 (1) | 0.000249498496989 |
| 0.3 | 0.0531610918197 | 0.05525 (221) | 0.00508019390663 | 0.04975 (199) | 0.00508533193158 |
| 0.4 | 0.201243123517 | 0.202 (808) | 0.0105857868424 | 0.2025 (810) | 0.0104717369423 |
| 0.6 | 0.54018588765 | 0.5525 (2210) | 0.0141262280704 | 0.53 (2120) | 0.0151155525622 |
| 1.0 | 0.879836321707 | 0.8975 (3590) | 0.00698814579797 | 0.878 (3512) | 0.00850282573131 |

The sigma≤0.2 counts are only 7 (PyMC) and 1 (NumPyro); such sparse indicators can make MCSE misleading. The NumPyro probability again differs from the reference by more than its estimated four-MCSE band; this was prespecified as descriptive only, is not a reliable tail-accuracy gate, and was not used to pass or fail this configuration. No tail guarantee, frequentist coverage, scientific calibration, broad-family behavior, or GPU validation is established.

## Integrity and evidence

All **68 frozen input hashes** and the clean exact worktree head verified before and after execution, preserving the original fit/semantic/reference bundles and their failures. Known scalar extraction was corrected only in the new analysis code using a size-one check followed by `.item()`; no historical script was modified.

Plan SHA-256: `6b660aa002c48ca4d1fdd8ceb8f408ca4e498bbea628cc21bde254c67dea4259`. Freeze SHA-256: `da9214548acf597a2866c91f42c083b18fad2e018e3a791057ba465efa028410`. Reference result SHA-256: `3be11b0e8e4d6ecc6910b33d965130ecd492d6024e28b3f1fc34d1a1b0821884`.

Historical outer command: `/usr/bin/python3 /private/tmp/autobayes-intercept-higher-accept-1/supervisor.py`. Do not rerun. The new runner identity, exact configuration and all source hashes are frozen in `freeze.json`; `launch.json`/`exit-receipt.json` retain actual argv, environment, start/end, process ID and exit.

- `result.json`: all full-precision diagnostics, 73 gate results, comparison values, CDF descriptions, hashes and integrity checks.
- `pymc.nc`, `numpyro.nc` and per-backend netCDF receipts: complete retained InferenceData, saved immediately.
- `pymc-analysis.json`, `numpyro-analysis.json`, and per-backend fit records: diagnostics and public API semantic evidence.
- `run.py`, `supervisor.py`, `freeze.json`, `preparation-checkpoint.json`: exact prospective implementation/configuration.
- `stdout.txt`, `stderr.txt`, `events.jsonl`, `exit-receipt.json`: unedited execution evidence.

This unit is complete and stopped. It authorizes no further tuning, diagnostic, or merge. Root independently reviews these artifacts before deciding PR acceptance.
