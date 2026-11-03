# Special terms in interactions and with ids; category-specific expressions and group terms

brms's special terms beyond the main effects `mo(x)` (docs/monotonic.md) and `me(x, sdx)` (docs/missing-and-measurement-error.md):
`mo()` in interactions and with `id`, `mo()` in a distributional formula, `me(x, sdx, gr = g)` and `me()` in interactions; and
brms's category-specific `cs()` on an expression and in a group-level term (docs/ordinal-categorical.md).

```python
import autobayes as ab
priors = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1),
                   simo={"ord": ab.Dirichlet(2, 2, 2), "mo(ord):x": ab.Dirichlet(1.5)})
model = ab.gaussian("y ~ x + mo(ord) + mo(ord):x + (1 | g)", data, priors)
both = ab.gaussian("y ~ mo(ord) * mo(o2) + (1 | g)", data,
                   ab.Priors(..., simo={"ord": ..., "o2": ..., "mo(ord):mo(o2)": {"ord": ..., "o2": ...}}))
shared = ab.gaussian('y ~ mo(ord, id = "a") + mo(ord, id = "a"):x + (1 | g)', data, ab.Priors(..., simo={"a": ab.Dirichlet(2)}))
latent = ab.gaussian("y ~ me(xg, sdxg, gr = g) + (1 | h)", data, ab.Priors(..., meanme=ab.Normal(0, 2), sdme=ab.HalfNormal(1)))
acat = ab.acat("ord ~ x + (cs(1) | g)", data, ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5), sd=ab.HalfNormal(1),
                                                       cor={"g": ab.LKJ(2)}))
print(model.describe())
```

## Special products

brms reads every term that holds `mo()` or `me()` and is not a plain main effect as one special term: one coefficient (its entry
of `bsp`) times the product of its parts, `mo(simo, Xmo)` for each `mo()` and the latent values `Xme` for each `me()`, and of its
other variables (brms's `Csp`, the covariate):

    mo(ord):x          b_mo(ord):x * mo(simo_mo(ord):x, ord_i) * x_i                        (Csp = x)
    mo(ord):mo(o2)     b * mo(simo_mo(ord):mo(o2)[ord], ord_i) * mo(simo_mo(ord):mo(o2)[o2], o2_i)
    me(xobs, sdx):w    b * Xme_xobs,i * w_i                                                 (the me() main effect's latent values)

`mo(ord) * mo(o2)` is `mo(ord) + mo(o2) + mo(ord):mo(o2)`, R's expansion, so it has four simplexes (one per main effect and one
per `mo()` of the product) and three coefficients, as brms has. A plain `mo(ord)` keeps B03's machinery; every other special term
is a `PopTerm` of kind `"special_product"` (`autobayes.special_terms.SpecialProduct`, `spec.sp_terms`): no design column, its
coefficient keyed by its name in `Priors(b={...})` and named `b_mo(ord):x` (brms's class `b`, coefficient `moord:x`), never
centred. The terms come in R's term order, which is brms's order of `bsp`, `Xmo_k` and `Csp_k`.

Simplexes. Each `mo()` part of a special product has a simplex of its own (Stan's stick-breaking coordinates, as B03's), keyed in
`Priors(simo=...)`, `value_point(simo=...)` and `model.names["simo:<key>"]` by the term's name (`"mo(ord):x"`; PyMC `simo_mo(ord):x`)
or, for a term with several `mo()` parts, by the term and the part's column (`{"mo(ord):mo(o2)": {"ord": ..., "o2": ...}}`; flat
key `mo(ord):mo(o2)[ord]`). brms's prior rows are class `simo`, coefficient `moord:x1`, `moord:moo21`, `moord:moo22` (the name and
the part's position); `ab.default_priors()` gives brms's `dirichlet(1)` for each. `value_point(bsp={"ord": ..., "mo(ord):x": ...})`
takes the coefficients, plain terms by column and special terms by name.

## `mo(x, id = "a")`

The `mo()` parts of one formula with the same id share one simplex (brms's documented rule, `?mo`: "within one formula"), keyed by
the id (`simo={"a": ...}`, PyMC `simo_a`, brms class `simo`, coefficient `a`); each term keeps its coefficient (`moordidEQa`,
`moordidEQa:x`). An id names one column here (brms also shares a simplex between columns with the same number of values; refused,
divergence D198). A distributional formula's ids are its own: `bf(y ~ mo(ord, id = "a"), sigma ~ mo(ord, id = "a"))` has two
simplexes, as brms does (`gauss_mo_id_sigma`).

`mo()` in a distributional formula (with or without an id) is a special product of that formula, added to its linear predictor
before its log link (`special_terms.special_link`): coefficient `sigma_mo(ord)`, simplex `sigma_simo_<key>`, priors
`Priors(sigma={"intercept": ..., "b": ..., "simo": {...}})` (brms's rows with `dpar = sigma`). The log link only (sigma, shape,
phi); others are refused (D198).

## A special term times a factor (D197)

`mo(ord):z` with a declared factor `z` is one special term per column of R's coding of the term: an indicator per level when the
term without the factor is not in the formula (`x + mo(ord):z`: `mo(ord):zp`, `mo(ord):zq`, `mo(ord):zr`, each with its own
coefficient and simplex, priors `simo={"mo(ord):zp": ..., ...}`), the non-reference levels' when it is (`mo(ord) * z`:
`mo(ord):zq`, `mo(ord):zr`). The same holds for `me(x, sdx):z`. brms 2.23.0 gets both wrong: without the main effect its Stan data
hold each level's indicator (`Csp_1..3`) but its program multiplies every level's term by `Csp_1`, so the rows of the other
levels get no effect and the terms are not identified apart; with it, brms stops ("replacement has 2 rows, data has 3"). This
library builds the interaction as written. The frozen case `gauss_mo_colon_factor` keeps per-gate expected failures on G2 (log
density, gradient), G3 and G4, which brms's program reaches; G1 (Stan data), G5 and G6 pass; `tests/test_special_term_ext.py`
checks the model's density against an independent computation.

## `me(x, sdx, gr = g)` and `me()` interactions

`me(xg, sdxg, gr = g)`: one latent value per level of `g` (`Xme_xg` over `Xme_xg_dim`, the sorted levels; brms's `Nme_1`), the
noisy value and noise of each level observed once (`Xn_1`, `noise_1`), and row i reading its level's (brms's `Jme_1`). The noisy
values and noise must be the same on every row of a level, as brms requires (refused naming the level otherwise).
`value_point(zme=...)` takes one standardised value per level. Predictions on the fitted rows only (D74).

`me(xobs, sdx) * w` is `me(xobs, sdx) + w + me(xobs, sdx):w`: the product multiplies the main effect's latent values (brms's one
`Xme_1` in both terms). An `me()` part needs its `me()` main effect in the formula (D198).

## `cs()` on an expression and in a group-level term

`cs(x + z)` is a category-specific term per variable, `cs(x) + cs(z)`, with a declared factor coded by treatment contrasts:
brms's `Xcs` = `(x, zq, zr)`, coefficients `cs(x)`, `cs(z)[q]`, `cs(z)[r]` (one per threshold each; brms class `b`, coefficients
`x`, `zq`, `zr`; `value_point(bcs={"x": ..., "zq": ..., "zr": ...})`). `cs(z)` of a factor alone is the same construction.

`(cs(1) | g)` (acat, sratio, cratio) gives each level of `g` one intercept per threshold, `Intercept[1]`, ..., `Intercept[K - 1]`,
correlated (brms's `sd_1[K - 1]`, `L_1`, `cor | g`), added to that threshold's predictor `mu_cs[i, k]` (brms's `mucs`): the term is
built by the model (`GroupTerm.thresholds`), its SDs under one `sd` prior or a mapping by coefficient, its correlation under
`cor={"g": ...}`. `predict(kind="linear")` has the threshold dimension when the term's effects are in it; without group effects
(`include_group_effects=False`) and without `cs()` population terms it is `mu`, as brms's `posterior_linpred(re_formula = NA)`.

## `cs()` in the cumulative family stays refused (D89)

brms takes it ("experimental") but does not keep each row's thresholds `Intercept - x bcs` ordered: its log density is NaN at 568
of 800 points drawn as the fixtures' are, and its `posterior_epred` has negative probabilities (smallest -0.2386 at the frozen
`cumulative_cs` points). The case keeps its gate.

## Refused, naming the construct

A special product next to an addition term, `gp()`, an autocorrelation, SAR, CAR, `fcor()` or residual correlation term, or a
group-level term the model builds itself; in a multivariate, non-linear or multinomial model, a mixture or the categorical family;
`mo()` and `me()` in one term or one formula (D73); two factors in one special term; a variable twice in one term; an id given to two
columns; an `me()` part without its main effect; `mo()` in a distributional formula without the log link; a special product in
`conditional_effects()` and `prior_draws()`; `(cs(1) | g)` with `||`, slopes, `gr()` options, an ID, a combined factor, another term
of its factor or another term the model builds; `cs(1)` in the population-level formula and expressions inside `cs()` other than
`+` (D89, D198).

## Evidence

Frozen cases (brms 2.23.0) pass gates G1-G6: `gauss_mo_interaction`, `gauss_mo_mo_interaction`, `gauss_mo_id`, `gauss_mo_id_sigma`,
`gauss_me_gr`, `gauss_me_interaction`, `acat_cs_expression` and `acat_cs_group`; `gauss_mo_colon_factor` passes G1, G5 and G6 with
expected failures on the gates brms's error reaches (D197). G1 compares every `Xmo_k`, `Jmo`, `Imo`, `Ksp` and `con_simo_k` in brms's
order and every `Csp_k` (`gates.check_special_covariates`), and `Jme_1`, `Nme_1` (`gates.check_me_groups`); G2 maps `bsp`,
`bsp_sigma`, every declared `simo_k` (its stick-breaking coordinates recomputed and checked) and `zme_1` per level; G3 checks
sigma against brms's own special terms (`adapter.brms_special_predictor`). `tests/test_special_term_ext.py` checks the parsing, the
coding, every refusal, priors, names, `value_point`, predictions, `describe()`, and the densities by hand.
