# Data provenance

**Status:** Active
**Last updated:** (set on first entry)

## Summary

Where every input came from, which file is canonical, and what is derived from what.

Write this before the first model. It is the page most often needed later and the most
painful to reconstruct, because the knowledge lives in whoever ran the pipeline and
decays fast. Six months on, "which of these four CSVs did we actually fit on" is a
question with a real answer that nobody remembers.

## Canonical inputs

One row per input file. **Canonical** means: the file a fit reads. If two files could
plausibly be the training data, that ambiguity is itself the finding — record it.

| File | Source | Obtained | Rows × cols | Canonical? | Notes |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

## Derivation chain

How raw inputs become the fitted dataset. Name the script for each arrow, so the chain
is executable rather than remembered.

```
raw_source_a  ──[scripts/stage1.py]──►  intermediate/  ──┐
                                                         ├─[scripts/merge.py]─►  training.csv
raw_source_b  ──[scripts/stage2.R]───►  intermediate/  ──┘
```

## Filters and exclusions

Every row-dropping decision, with its justification and its count. A filter applied in
preprocessing is a modelling assumption wearing a disguise — an exclusion nobody wrote
down will be rediscovered as a mysterious sample-size mismatch.

| Filter | Where applied | Rows dropped | Why |
| --- | --- | --- | --- |
| | | | |

## Known problems

Quality issues that survive into the fitted data: mislabelled keys, silent imputation,
coverage gaps by period or group, units that change partway through a series. Record
them even when unfixed — an unfixed known problem is far better than an unknown one.

## Unresolved

Inputs whose producer cannot be identified, files with no reproducible chain, anything
you inherited and cannot regenerate. Be explicit: an unreproducible input is a real
constraint on what the project can claim.

## Related

[[data/dictionary]]
