# Syntax fixes from the audit (F08)

The syntax audit (#85, docs/brms-parity/reports/syntax-audit.pdf) read 248 brms formulas with brms 2.23 and with AutoBayes.
F08 fixes what it found in the formula language; every changed corpus entry (`tests/brms_syntax/corpus.json`) now matches
brms's `standata()` or is refused with a message naming its construct. After F08 the corpus has 134 matches, 113 named
refusals and one documented difference (D9), and no silent misreading, unclear refusal or crash.

**Ordered factors** are coded with R's polynomial contrasts, as brms codes them: [ordered factors](ordered-factors.md).

**Group-level terms as brms reads them.** The intercept is implied: `(x | g)` is `(1 + x | g)` and `(x || g)` is
`(1 + x || g)` (the same terms, the same model). `gr(g, cor = FALSE)` is `||` and `gr(g, id = "p")` is `|p|` (an ID is
offered between the response formula and a distributional formula, D106). A term without its varying intercept,
`(0 + x | g)` or `(x - 1 | g)`, stays refused, and the refusal now gives brms's reading where there is one: next to `(1 | g)`
it is the uncorrelated intercept and slope, `(1 + x || g)`; two terms of `g` that share an ID are one correlated block,
`(1 + x + z | g)`.

**Refusals that name their construct.** R's `.`, backtick-quoted names (`` `x.dot` ``: write `x.dot`), brms's reserved
`Intercept` (`0 + Intercept + x`), the namespace operator `::` (`splines::ns(x)`), `%in%` and other `%...%` operators, the `^`
expansion (`(x + z + w)^2`; G03 reads it now, docs/term-algebra.md), smooth terms `s()`, `t2()`, `te()`, `ti()` (their own message, no longer the transforms'), a
transformed response (`log(y) ~ x`), a `cbind()` response of a family of one response column (with the `mvbind()` spelling of
several Gaussian responses), `disc ~` of an ordinal family (D38), a non-linear formula of a distributional parameter (brms's
`nlf(sigma ~ ...)`, D62) and a non-linear parameter nested in another (`nlf(a ~ c * z)`, D60).

**Cheap spellings of what was already offered.**

- `trials(<number>)`: `k | trials(15) ~ x`, the same number of trials on every row (brms's Stan data `trials`); the number is
  `ModelSpec.trials`'s text, `"15"`, and every frame the model reads (fitted rows, new rows) gets it as a column of that name
  (`formula.with_trials`).
- brms's censoring labels: `cens(col)` reads `"left"`, `"none"`, `"right"` and `"interval"` (with `cens(col, y2)`) as -1, 0,
  1 and 2, and a logical column as 1 (`TRUE`, right-censored) or 0, as brms does; any other label is refused (D24).
- `cat(K)`, brms's deprecated spelling of an ordinal response's number of categories, for the cumulative, sratio, cratio and
  acat families: taken where `K` is the data's number of categories (it then changes nothing); another `K`, with which brms
  has thresholds no row reaches, is refused (D169), as is `cat()` with `thres()` or in another family.

**Documentation.** D9 now says that brms's own Gamma link is `log`, as `ab.gamma`'s, and that only `stats::Gamma()` (brms's
`family = Gamma()`) defaults to `inverse`; D20 that `gr()` is taken since D08 and its `cor` and `id` since F08.

**The audit's data2 entries.** `tests/brms_syntax/audit.py` hands the entries that name `data2` (`car()`, `sar()`,
`fcor()`) the matrices `brms_reading.R` builds (`audit.make_data2`) and compares the terms' Stan data (`Nloc`, `Jloc`, the
edges, `Nneigh`, `eigenMcar`; `Msar`, `eigenMsar`, `N_tg`; `Mfcor`): the three entries match brms.
