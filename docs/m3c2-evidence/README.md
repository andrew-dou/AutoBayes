# Intercept-control evidence

Decision/probes preceded feature code. The initial targeted failure and failed full suite are preserved. Final changes after the full suite touch test assertions and documentation only. See ../m3c2-decision.md for the test map and outcome interpretation.

Full local invocation artifacts, including inherited regression fixture outputs, remain at /Users/andrewdou/iqss-research/artifacts/2026-09-19-autobayes-intercept-validation/. `external-validation-manifest.sha256` is relative to that directory. This directory includes only logs/receipts plus the relevant source probe/supervisor, not posterior draws. The supervisor is an executed historical record with a fixed unique output path; do not rerun it into that path.

The later independent review below adds bounded Gaussian CPU evidence. Synthetic fixed-draw panels remain labelled stubs, not convergence evidence. GPU validation is still absent.

## Later CPU evidence

The original target_accept0.9 failure, read-only semantic analysis, independent numerical reference and separate successful target_accept0.99 configuration are retained in the four named subdirectories. See ../m3c2-decision.md for the acceptance decision and limits. Scripts, frozen configuration, reports, logs and JSON outcomes are committed; posterior netCDF files and the quadrature NPZ remain in the corresponding `/Users/andrewdou/iqss-research/artifacts/2026-09-19-autobayes-intercept-*` bundles.

`COMMITTED-FILES.sha256` checks each committed subset. `FULL-EXTERNAL-BUNDLE.sha256` is the original manifest relative to its full external bundle and includes intentionally uncommitted binary files. It is not a manifest to run against this subset. Historical scripts retain original absolute paths and must not be rerun into their exclusive output directories. Original failures and thresholds were not overwritten.
