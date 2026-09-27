# Fit provenance and legacy draws

AutoBayes captures the built model's target identity before dispatching either sampler. FitResult.target_identity returns a defensive copy. The same JSON is stored in the returned InferenceData root attribute autobayes_target_identity_json so supported NetCDF round-trips retain it. This is provenance bookkeeping against accidental pairing, not authentication of arbitrary externally edited metadata.

fit_record(model, fit) and predict(model, fit) reject identities that are malformed, conflict between the result and draws, or differ from the supplied model. An independently rebuilt identical target is acceptable. Backend and seed differences do not change target identity. Direct graph mutation remains unsupported.

Raw InferenceData without provenance remains accepted by predict, but its output has fit_binding_status="unverified". Valid metadata present on raw draws is checked. No provenance is inferred from variable names or dimensions.

Creating a record from an unbound historical fit requires fit_record(model, fit, allow_unverified=True). Such a record includes requested_target_identity and fit_binding_status="unverified", but omits target_identity. Default compare_records therefore refuses to claim matching targets; its separate allow_legacy=True switch permits an explicitly unverified descriptive comparison. Unknown provenance cannot be repaired merely by pairing the old draws with a reconstructed model of the right shape. Preserve original evidence and establish provenance externally rather than fabricating metadata.

Tests use fixed synthetic draws and stubbed Bambi sampler calls, including the NumPyro dispatch path on CPU; they do not constitute new MCMC runs or GPU validation. Existing synthetic fixture provenance is deliberately labeled in test construction, not inferred in production.
