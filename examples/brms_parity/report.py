#!/usr/bin/env python
"""Write the benchmark report (LaTeX) from compare.py's comparison-<date>.json.

    python examples/brms_parity/report.py <comparison-<date>.json> [--previous results/comparison-2026-10-17.json]
        [--out docs/brms-parity/reports/benchmark-<date>.tex] [--pdf]

Every number and every sentence that depends on one is generated from the JSON; the fixed text is the method and its
limits (README.md). ``--previous`` adds a table of the models both runs have, end to end, then and now. ``--pdf`` runs
pdflatex twice in the output's directory and removes the auxiliary files. A comparison built from smoke runs is marked as
such on the first page; do not commit a report built from one.

Given compare.py --scaling's scaling-<date>.json instead, it writes the scaling report,
docs/brms-parity/reports/scaling-<date>.tex: each library's best configuration per model and size, then the full sweep.
"""
import argparse
import json
import statistics
import subprocess
from pathlib import Path

LABELS = {
    "B1": r"Bernoulli, \code{(1 + x | g)}", "B2": r"Gaussian, \code{(1 + x | g)}", "B3": "cumulative, five categories",
    "B4": "zero-inflated Poisson", "B5": "Weibull", "B6": r"zero-inflated neg.\ binomial", "B7": "sratio, five categories",
    "B8": "Gaussian, horseshoe on 20", "B9": "beta-binomial, trials", "B10": r"Gaussian, \code{(1 | mm(g1, g2))}",
    "B11": "mixture of two Gaussians", "B12": r"exact GP, \code{gp(x)}", "B13": r"AR(1), \code{ar(t, g)}",
    "B14": r"non-linear, \code{a * exp(b * x)}", "B15": r"\code{mvbind(y1, y2)}, rescor", "B16": r"monotonic \code{mo(edu)}",
    "B17": r"Weibull, \code{cens()}", "B18": r"distributional, \code{sigma \textasciitilde{} z}",
}
FACTOR_WORDS = {"steps": "fewer leapfrog steps", "gradient": "cheaper gradients", "overhead": "less work per step beyond the "
                "gradient", "parallelism": "more cores kept busy"}

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{booktabs,array,enumitem,caption,float}
\usepackage{xcolor,hyperref}
\sloppy
\hypersetup{colorlinks=true, linkcolor=blue!50!black, urlcolor=blue!50!black}
\newcommand{\code}[1]{\texttt{#1}}
\newcommand{\file}[1]{\texttt{#1}}
\setlist{nosep, leftmargin=1.4em}
"""


def esc(s: str) -> str:
    return "".join({"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}.get(c, c) for c in s)


def short(model: str) -> str:
    return model.split("_")[0]


def f(v, nd=1) -> str:
    """A number for a table: '--' when missing; one decimal by default, two below 1, none above 1000."""
    if v is None:
        return "--"
    v = float(v)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) < 1 and nd == 1:
        return f"{v:.2f}"
    return f"{v:.{nd}f}"


def x(v) -> str:
    return "--" if v is None else f"{float(v):.2f}"


def mean(v):
    v = [t for t in (v if isinstance(v, list) else [v]) if t is not None]
    return statistics.mean(v) if v else None


def table(caption: str, label: str, cols: str, head: list[str], body: list[list[str]], size: str = r"\footnotesize",
          sep: str = "4pt") -> str:
    lines = [r"\begin{table}[H]", size, rf"\setlength{{\tabcolsep}}{{{sep}}}", r"\begin{center}",
             rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule"] + head + [r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in body]
    lines += [r"\bottomrule", r"\end{tabular}\end{center}", rf"\caption{{\small {caption}}}\label{{{label}}}",
              r"\end{table}", ""]
    return "\n".join(lines)


def ess_rate(row, side, kind="min_ess"):
    T = row["stages"][side]["decomposition"]["T"]
    return row[side][kind] / T if T else None


def summary(rows) -> str:
    tot = [r["brms"]["total"] / r["autobayes"]["total"] for r in rows]
    ex = [r["factors"]["speedup"] for r in rows]
    largest = [(r["factors"]["largest"], r["factors"][r["factors"]["largest"]] > 1) for r in rows if r["factors"]["largest"]]
    counts = {k: largest.count(k) for k in set(largest)}
    parts = [f"the {k} factor ({'for' if up else 'against'} AutoBayes) on {n} model{'s' if n > 1 else ''}"
             for (k, up), n in sorted(counts.items(), key=lambda t: -t[1])]
    slower = [short(r["model"]) for r in rows if r["factors"]["speedup"] < 1]
    text = (rf"\noindent\textbf{{The answer.}} Fitting each model from scratch (compiling and sampling), brms took "
            rf"{min(tot):.1f} to {max(tot):.1f} times as long as AutoBayes (median {statistics.median(tot):.1f}). Its "
            rf"sampler alone (warm-up and sampling of every chain, compilation excluded) took {min(ex):.1f} to "
            rf"{max(ex):.1f} times as long (median {statistics.median(ex):.1f}). Splitting that ratio into its four factors "
            rf"(Section~\ref{{sec:decomp}}), the largest was "
            rf"{', '.join(parts)}.")
    if slower:
        text += f" brms's sampler was the faster on {', '.join(slower)}."
    return text + "\n"


def stage_tables(rows) -> str:
    body_b, body_a = [], []
    for r in rows:
        sb, sa = r["stages"]["brms"], r["stages"]["autobayes"]
        m = short(r["model"])
        body_b.append([m, LABELS.get(m, ""), f"{r['n']:,}", f(sb["codegen"]), f(sb["compile"]), f(sb["exec"]),
                       f(sb["sample_overhead"]), f(sb["total"])])
        body_a.append([m, f(sa["build"]), f(sa["fit_pre"]), f(sa["compile"]), f(sa["compile_events_sampler"]),
                       f(sa["exec"]), f(sa["fit_post"]), f(sa["total"]),
                       x(r["brms"]["total"] / r["autobayes"]["total"]), x(r["factors"]["speedup"])])
    tb = table("brms's stages, seconds. \\emph{compile}: \\code{brm(chains = 0)}, code generation, C++ compilation and "
               "loading; \\emph{codegen}: the code generation alone, run again afterwards; \\emph{exec}: the slowest chain's "
               "warm-up and sampling as Stan reports them; \\emph{around}: the rest of \\code{update()} (forking the chains, "
               "brms's data preparation and its renaming of the fit).", "tab:brms", "llrrrrrr",
               [r"& model & rows & codegen & compile & exec & around & total \\"], body_b)
    ta = table("AutoBayes's stages, seconds. \\emph{build}: the model; \\emph{pre}: the fit up to NumPyro (PyTensor's graph "
               "turned into JAX, initial points); \\emph{compile}: the NumPyro run up to the end of its largest XLA "
               "compilation, with \\emph{JAX} the part of it JAX's monitoring reports as tracing, lowering and "
               "compiling; \\emph{exec}: the rest of the run, every chain's warm-up and sampling; \\emph{post}: the draws "
               "moved to the constrained scale and the InferenceData. The last two columns are brms's time over "
               "AutoBayes's, end to end and for the samplers' execution.", "tab:autobayes", "lrrrrrrrrr",
               [r"& build & pre & compile & JAX & exec & post & total & \multicolumn{2}{c}{faster by} \\",
                r"\cmidrule(lr){9-10}& & & & & & & & total & exec \\"], body_a)
    return tb + ta


def work_table(rows) -> str:
    body = []
    for r in rows:
        db, da = r["stages"]["brms"]["decomposition"], r["stages"]["autobayes"]["decomposition"]
        sb, sa = r["stages"]["brms"], r["stages"]["autobayes"]
        body.append([short(r["model"]), f"{db['S_warmup']:,.0f}", f"{db['S_sampling']:,.0f}", f(mean(sb["tree_depth_sampling"]), 2),
                     f(mean(sb["step_size_final"]), 3), f"{da['S_warmup']:,.0f}", f"{da['S_sampling']:,.0f}",
                     f(mean(sa["tree_depth_sampling"]), 2), f(mean(sa["step_size_final"]), 3), x(db["S"] / da["S"])])
    return table("Sampler work: leapfrog steps summed over the chains, for warm-up and for sampling; the mean tree depth "
                 "while sampling and the final step size, averaged over the chains. The last column is brms's steps over "
                 "AutoBayes's.", "tab:work", "lrrrrrrrrr",
                 [r"& \multicolumn{4}{c}{brms} & \multicolumn{4}{c}{AutoBayes} & \\",
                  r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}& warm-up & sampling & depth & step & warm-up & sampling & depth "
                  r"& step & ratio \\"], body)


def gradient_table(rows) -> str:
    body = []
    for r in rows:
        db, da = r["stages"]["brms"]["decomposition"], r["stages"]["autobayes"]["decomposition"]
        sa = r["stages"]["autobayes"]
        body.append([short(r["model"]), str(r["stages"]["brms"]["n_unconstrained"]), f(db["g_host_us"]),
                     f(db["g_direct_us"]), f(db["g_dispatch_us"]), f(db["g_us"]), f(da["g_host_us"]), f(da["g_dispatch_us"]),
                     f(da["g_us"]), f(sa["grad_lower"] + sa["grad_compile"], 2), x(db["g_us"] / da["g_us"])])
    return table("The cost of one log density and gradient, microseconds, at a posterior draw on the unconstrained scale "
                 "(the number of coordinates first). brms: \\code{rstan::grad\\_log\\_prob} (\\emph{host}), the model "
                 "instance's method called directly (\\emph{direct}), a call that does no work (\\emph{null}), and "
                 "\\emph{direct} less \\emph{null}, the estimate used below. AutoBayes: the jitted potential and gradient "
                 "called from Python (\\emph{host}), a jitted call that does no work (\\emph{null}), and the cost per "
                 "evaluation inside one compiled loop (\\emph{loop}), the estimate used below; \\emph{compile} is "
                 "seconds to trace, lower and compile it. Medians of 2,000 calls after a first one.", "tab:grad",
                 "lrrrrrrrrrr",
                 [r"& & \multicolumn{4}{c}{brms} & \multicolumn{4}{c}{AutoBayes} & \\",
                  r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}& coords & host & direct & null & used & host & null & loop & "
                  r"compile & ratio \\"], body)


def decomposition(rows) -> str:
    body = []
    for r in rows:
        db, da, fa = r["stages"]["brms"]["decomposition"], r["stages"]["autobayes"]["decomposition"], r["factors"]
        body.append([short(r["model"]), f(db["T"]), f(da["T"]), x(fa["speedup"]), x(fa["steps"]), x(fa["gradient"]),
                     x(fa["overhead"]), x(fa["parallelism"]), x(db["O"]), x(da["O"]), x(db["U"]), x(da["U"])])
    # the worked example: the model with the median sampler speed-up
    ex = sorted(rows, key=lambda r: r["factors"]["speedup"])[len(rows) // 2]
    db, da, fa = ex["stages"]["brms"]["decomposition"], ex["stages"]["autobayes"]["decomposition"], ex["factors"]
    m = short(ex["model"])

    def side(name, d):
        return (rf"{name} took $T = {d['T']:.2f}$ s and {d['S']:,.0f} leapfrog steps over all chains; at "
                rf"$g = {d['g_us']:.1f}\,\mu$s a gradient those steps alone are $S g = {d['S_g']:.2f}$ CPU seconds. Its "
                rf"chains used $C = {d['cpu']:.2f}$ CPU seconds, so each step cost $O = C/(Sg) = {d['O']:.2f}$ gradients "
                rf"and $U = C/T = {d['U']:.2f}$ cores were busy on average.")
    text = [r"\section{Where the sampling time goes}\label{sec:decomp}", "",
            r"For each side, $T$ is the samplers' execution time (warm-up and sampling of every chain, compilation "
            r"excluded), $S$ the leapfrog steps summed over the chains, $g$ the cost of one log density and gradient "
            r"inside the sampler (Table~\ref{tab:grad}) and $C$ the CPU seconds spent over $T$. With $O = C/(Sg)$, the "
            r"CPU cost of one step counted in gradients (1 if a step costs exactly a gradient), and $U = C/T$, the cores "
            r"kept busy, $T = S g O / U$ holds exactly, so brms's $T$ over AutoBayes's is the product",
            r"\[ \frac{T_b}{T_a} = \underbrace{\frac{S_b}{S_a}}_{\text{steps}} \cdot "
            r"\underbrace{\frac{g_b}{g_a}}_{\text{gradient}} \cdot \underbrace{\frac{O_b}{O_a}}_{\text{overhead}} \cdot "
            r"\underbrace{\frac{U_a}{U_b}}_{\text{parallelism}}. \]",
            rf"\textbf{{A worked example, {m}}} (the median sampler speed-up). " + side("brms", db) + " " + side("AutoBayes", da)
            + rf" Then ${db['T']:.2f}/{da['T']:.2f} = {fa['speedup']:.2f}$, which is {fa['steps']:.2f} (steps) times "
            rf"{fa['gradient']:.2f} (gradient) times {x(fa['overhead'])} (overhead) times {x(fa['parallelism'])} "
            r"(parallelism).", "",
            table("The four factors of brms's sampler time over AutoBayes's, and each side's $O$ and $U$. A factor above 1 "
                  "favours AutoBayes.", "tab:decomp", "lrrrrrrrrrrr",
                  [r"& \multicolumn{2}{c}{$T$ (s)} & & \multicolumn{4}{c}{factor} & \multicolumn{2}{c}{$O$} & "
                   r"\multicolumn{2}{c}{$U$} \\",
                   r"\cmidrule(lr){2-3}\cmidrule(lr){5-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}& brms & AutoBayes & "
                   r"ratio & steps & gradient & overhead & parallel & brms & AutoBayes & brms & AutoBayes \\"], body),
            r"\noindent Model by model, the largest factor:", r"\begin{itemize}"]
    for r in rows:
        fa = r["factors"]
        if fa["largest"] is None:
            continue
        text.append(rf"\item \textbf{{{short(r['model'])}}} ({LABELS.get(short(r['model']), '')}): sampler "
                    rf"{fa['speedup']:.2f} times faster in AutoBayes; largest factor {fa['largest']} "
                    rf"({fa[fa['largest']]:.2f}, {FACTOR_WORDS[fa['largest']] if fa[fa['largest']] > 1 else 'against AutoBayes'}).")
    return "\n".join(text + [r"\end{itemize}", ""])


def efficiency_table(rows) -> str:
    body = []
    for r in rows:
        body.append([short(r["model"]), f(r["autobayes"]["min_ess"], 0), f(r["brms"]["min_ess"], 0),
                     f(r["autobayes"]["min_ess_tail"], 0), f(r["brms"]["min_ess_tail"], 0), f(ess_rate(r, "autobayes"), 0),
                     f(ess_rate(r, "brms"), 0), str(r["divergent"]["autobayes"]), str(r["divergent"]["brms"]),
                     f(r["max_abs_z"], 2)])
    return table("Sampling efficiency and agreement: the smallest bulk and tail ESS over the parameters both libraries "
                 "report (the same ArviZ estimators for both), the smallest bulk ESS per second of the samplers' "
                 "execution $T$, the divergent transitions after warm-up, and the largest $|z|$, the difference of the "
                 "posterior means over their combined Monte Carlo standard error.", "tab:eff", "lrrrrrrrrr",
                 [r"& \multicolumn{2}{c}{min bulk ESS} & \multicolumn{2}{c}{min tail ESS} & \multicolumn{2}{c}{bulk ESS/s} "
                  r"& \multicolumn{2}{c}{divergent} & max $|z|$ \\",
                  r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}& AutoBayes & brms & AutoBayes "
                  r"& brms & AutoBayes & brms & AutoBayes & brms & \\"], body, size=r"\small")


def post_table(rows) -> str:
    fns = ["posterior_epred", "log_lik", "loo", "waic", "bayes_R2", "hypothesis", "conditional_effects"]

    def cell(v):
        if not v:
            return "--"
        if "refused" in v:
            return r"\textit{ref.}"
        return f"{v['median']:.2f}"
    body = [[short(r["model"])] + [f"{cell(r['post'][fn]['autobayes'])} / {cell(r['post'][fn]['brms'])}" for fn in fns]
            for r in rows]
    head = [r"& \code{posterior\_epred} & \code{log\_lik} & \code{loo} & \code{waic} & \code{bayes\_R2} & \code{hypothesis} "
            r"& \code{cond.\_effects} \\"]
    return table("Post-processing, seconds, AutoBayes / brms: the median of three calls after a first one. \\textit{ref.}: "
                 "the function refused the model (the message is in the JSON).", "tab:post", "lccccccc", head, body,
                 sep="3pt")


def previous_table(rows, prev) -> str:
    old = {r["model"]: r for r in prev}
    body = [[short(r["model"]), f(old[r["model"]]["brms"]["total"]), f(r["brms"]["total"]),
             f(old[r["model"]]["autobayes"]["total"]), f(r["autobayes"]["total"])] for r in rows if r["model"] in old]
    if not body:
        return ""
    return table("End-to-end seconds in the previous run and in this one, for the models both have.", "tab:prev", "lrrrr",
                 [r"& \multicolumn{2}{c}{brms} & \multicolumn{2}{c}{AutoBayes} \\",
                  r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}& before & now & before & now \\"], body, size=r"\small")


LIMITS = r"""\section{What these numbers do not show}

\begin{itemize}
\item \textbf{One run each}, on one machine whose load is not controlled: a single number may be off by 10--30\%.
\item \textbf{Step counts differ for reasons beyond the gradient.} Stan and NumPyro both run NUTS with a diagonal metric
  and dual-averaging step-size adaptation, but their warm-up schedules, windows and step-size searches are not the same,
  and they start from different points, so they end on different step sizes and tree depths. The steps factor measures
  that difference; it says nothing about either library's gradient. Neither count includes the gradients spent searching
  for an initial step size.
\item \textbf{The gradient costs are estimates.} The host-call medians include a call overhead the samplers do not pay
  (for a jitted JAX function on the CPU, tens of microseconds on a busy machine); the factors use AutoBayes's cost inside
  a compiled loop and rstan's direct call less a call that does no work. brms's point is a draw of a short extra chain
  (brms renames its fit's parameters, so the fit's draws cannot be mapped back to Stan's coordinates); AutoBayes's is the
  last draw of its fit. For these models the cost hardly depends on the point.
\item \textbf{CPU time and parallelism.} AutoBayes's $C$ is the whole process's CPU time over the run's execution
  (XLA's threads included); brms's is the CPU time of the forked chain processes over \code{update()}, which includes
  each child's small R-side setup. $U$ above the number of chains means a chain used more than one core.
\item \textbf{Compilation is measured from the timeline.} AutoBayes's sampler compilation is the time from the start of
  the NumPyro run to the end of its largest XLA compilation; the part JAX reports as tracing, lowering and compiling is
  shown beside it. brms's codegen is a warm re-run of \code{stancode()} and \code{standata()}.
\item \textbf{Repeated fits.} brms compiles once per model and can refit new data without recompiling; for many fits of
  one model only the samplers' time matters.
\item \textbf{Other setups.} rstan only (cmdstanr is not installed), no within-chain threading, the CPU only.
\end{itemize}

\section{Reproducing}

\begin{verbatim}
python examples/brms_parity/benchmark_data.py DATA
Rscript examples/brms_parity/benchmark_brms.R DATA OUT
python examples/brms_parity/benchmark_autobayes.py DATA OUT
python examples/brms_parity/compare.py OUT
python examples/brms_parity/report.py OUT/comparison-DATE.json --pdf
\end{verbatim}
"""


def what_was_run(rows) -> str:
    r0 = rows[0]
    sa, sb = r0["settings"]["autobayes"] or {}, r0["settings"]["brms"] or {}
    va, vb = r0["versions"]["autobayes"] or {}, r0["versions"]["brms"] or {}
    return "\n".join([
        r"\section{What was run}", "",
        rf"Each of the {len(rows)} models was fitted by both libraries with the same formula, family and explicit priors, "
        r"on the same simulated data (\file{examples/brms\_parity/benchmark\_data.py}, fixed seeds), with the same sampler "
        rf"settings: {sa.get('chains', '?')} chain{'s in parallel' if sa.get('chains') != 1 else ''}, "
        rf"{sa.get('tune', '?')} warm-up and {sa.get('draws', '?')} "
        rf"kept draws each, target acceptance {sa.get('target_accept', '?')}.",
        r"\begin{itemize}",
        rf"\item \textbf{{brms}} {esc(str(vb.get('brms', '?')))} with rstan {esc(str(vb.get('rstan', '?')))} "
        rf"(R {esc(str(vb.get('R', '?')))}), {sb.get('cores', '?')} cores; compiled with \code{{brm(chains = 0)}} and "
        r"sampled with \code{update(recompile = FALSE)}.",
        rf"\item \textbf{{AutoBayes}} (commit \code{{{esc(str(va.get('commit', '?')))}}}) with NumPyro "
        rf"{esc(str(va.get('numpyro', '?')))}, JAX {esc(str(va.get('jax', '?')))} "
        rf"and PyMC {esc(str(va.get('pymc', '?')))} on the CPU ({esc(str(va.get('machine', '?')))}, "
        rf"{va.get('cpu_count', '?')} cores), \code{{ab.fit}} unchanged and instrumented from outside "
        r"(\file{examples/brms\_parity/README.md}).",
        r"\end{itemize}", ""])


def _med(c, key, nd=1) -> str:
    v = c and c.get(key)
    return f(v["median"], nd) if v else "--"


def _range(c, key) -> str:
    v = c and c.get(key)
    return f"{f(v['median'], 0)} [{f(v['lo'], 0)}, {f(v['hi'], 0)}]" if v else "--"


SCALING_METHOD = r"""\section{What was run}

Five models (\file{benchmark\_data.py --scaling N}: S1 logistic \code{y \textasciitilde{} x1 + x2 + (1 + x1 | g)}, S2 Gaussian
\code{(1 + x1 + x2 | g)}, S3 Poisson \code{(1 | g1) + (1 | g2)}, S4 cumulative \code{(1 | g)}, S5 \code{gp(x, k = 30)}) at
each size, with $J = \min(N/100, 1000)$ groups (the cap is AutoBayes's memory: formulae's dense $N \times J$ indicator per
group term), the same explicit priors in both libraries, four chains of 1000 warm-up and 1000 kept draws, target acceptance
0.9, three sampler seeds per configuration. ESS is the smallest over the parameters every record of the model reports
(ArviZ's rank-normalised bulk and tail ESS for both); per second means per second of the samplers' execution (warm-up and
sampling of every chain, compilation excluded) unless marked end to end.

\textbf{Cores.} brms runs four chains in parallel, each with $t$ threads under \code{threading(t)} (Stan's
\code{reduce\_sum}, cmdstanr): $4t$ cores, the SLURM allocation. AutoBayes's \code{--cores c} sets the process's CPU affinity
to $c$ cores before JAX starts (the enforced part), sizes the thread pools to $c$, and gives XLA one CPU device per chain;
how XLA uses the $c$ cores inside a chain is its own decision. $U$ is the CPU seconds over the samplers' time, the cores
kept busy on average; a value above the configuration's cores is flagged.
"""


def scaling_report(data: dict, out: str | None, pdf: bool) -> Path:
    date, configs, best = data["date"], data["configs"], data["best"]
    root = Path(__file__).resolve().parents[2]
    dest = Path(out) if out else root / "docs" / "brms-parity" / "reports" / f"scaling-{date}.tex"
    by = {(c["model"], c["n"], c["config"]): c for c in configs}
    head = [r"& rows & \multicolumn{3}{c}{brms, best} & \multicolumn{3}{c}{AutoBayes, best} & \multicolumn{2}{c}{ratio} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}& & configuration & bulk ESS/s & tail ESS/s & "
            r"configuration & bulk ESS/s & tail ESS/s & bulk & end to end \\"]
    body = []
    for b in best:
        cb, ca = by.get((b["model"], b["n"], b["brms"])), by.get((b["model"], b["n"], b["autobayes"]))
        body.append([short(b["model"]), f"{b['n']:,}", esc(b["brms"] or "--"), _range(cb, "ess_bulk_per_s"),
                     _med(cb, "ess_tail_per_s", 0), esc(b["autobayes"] or "--"), _range(ca, "ess_bulk_per_s"),
                     _med(ca, "ess_tail_per_s", 0), x(b["ratio_bulk"]), x(b["ratio_total"])])
    t1 = table("Each library's best configuration (the highest median bulk ESS per second over the seeds): bulk ESS per "
               "second as the median [smallest, largest] over the seeds, tail ESS per second the median. Ratio: "
               "AutoBayes's over brms's, per second of the samplers' time and end to end.", "tab:best", "lrlrrlrrrr",
               head, body, size=r"\scriptsize", sep="3pt")
    sweep = [r"\begin{longtable}{@{}llrrrrrrrrr@{}}", r"\caption{\small The full sweep, medians over the seeds. cores: "
             r"of the configuration; $U$: cores kept busy; ms/step: the slowest chain's time per leapfrog step; $T$ the "
             r"samplers' time and total the end-to-end time, seconds. $^*$: CPU time above the configuration's cores in "
             r"some seed.}\label{tab:sweep}\\", r"\toprule",
             r"& rows & configuration & cores & $U$ & bulk ESS/s & tail ESS/s & $T$ & total & ms/step & max $\hat R$ \\",
             r"\midrule\endhead"]
    for c in configs:
        sweep.append(" & ".join([short(c["model"]), f"{c['n']:,}", esc(c["config"]) + ("$^*$" if c["over_cap"] else ""),
                                 str(c["cores"]), _med(c, "U", 2), _med(c, "ess_bulk_per_s", 0), _med(c, "ess_tail_per_s", 0),
                                 _med(c, "T"), _med(c, "total"), _med(c, "step_ms", 3), _med(c, "max_rhat", 3)]) + r" \\")
    sweep += [r"\bottomrule", r"\end{longtable}", ""]
    parts = [PREAMBLE.replace(r"\usepackage{booktabs,", r"\usepackage{longtable,booktabs,"),
             r"\title{AutoBayes against brms: scaling}", rf"\author{{Benchmark run of {esc(date)}, prepared for Andrew}}",
             rf"\date{{{esc(date)}}}", "", r"\begin{document}", r"\maketitle", ""]
    if data.get("smoke"):
        parts.append(r"\begin{center}\fbox{\parbox{0.9\textwidth}{\textbf{Smoke run.} These numbers check the scripts "
                     r"and mean nothing about speed. Do not commit this report.}}\end{center}" + "\n")
    parts += [SCALING_METHOD, r"\section{Best against best}", "", t1, r"\section{The sweep}", "", "\n".join(sweep),
              r"\end{document}", ""]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(parts))
    print(f"wrote {dest}")
    if pdf:
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", dest.name], cwd=dest.parent,
                           check=True, stdout=subprocess.DEVNULL)
        for ext in (".aux", ".log", ".out", ".toc"):
            dest.with_suffix(ext).unlink(missing_ok=True)
        print(f"wrote {dest.with_suffix('.pdf')}")
    return dest


def main(path: str, previous: str | None, out: str | None, pdf: bool) -> Path:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "best" in data:              # compare.py --scaling's output
        return scaling_report(data, out, pdf)
    rows = [r for r in data if "stages" in r]
    if not rows:
        raise SystemExit(f"{path} has no model with stage records (compare.py output of the staged benchmark)")
    date = Path(path).stem.removeprefix("comparison-")
    smoke = any(r["settings"]["smoke"] for r in rows)
    root = Path(__file__).resolve().parents[2]
    dest = Path(out) if out else root / "docs" / "brms-parity" / "reports" / f"benchmark-{date}.tex"
    parts = [PREAMBLE, rf"\title{{AutoBayes against brms: where the speed comes from}}",
             rf"\author{{Benchmark run of {esc(date)}, prepared for Andrew}}", rf"\date{{{esc(date)}}}", "",
             r"\begin{document}", r"\maketitle", ""]
    if smoke:
        parts.append(r"\begin{center}\fbox{\parbox{0.9\textwidth}{\textbf{Smoke run.} One chain, 50 warm-up and 50 kept "
                     r"draws per model: these numbers check the scripts and mean nothing about speed. Do not commit this "
                     r"report.}}\end{center}" + "\n")
    parts += [summary(rows), what_was_run(rows), r"\section{Time to a posterior, by stage}", "", stage_tables(rows),
              r"\section{The samplers' work}", "", work_table(rows), gradient_table(rows), decomposition(rows),
              r"\section{Is the faster sampler sampling as well?}", "", efficiency_table(rows), r"\section{Post-processing}",
              "", post_table(rows)]
    if previous:
        parts += [r"\section{Against the previous run}", "", previous_table(rows, json.loads(Path(previous).read_text()))]
    parts += [LIMITS, r"\end{document}", ""]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(parts))
    print(f"wrote {dest}")
    if pdf:
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", dest.name], cwd=dest.parent,
                           check=True, stdout=subprocess.DEVNULL)
        for ext in (".aux", ".log", ".out", ".toc"):
            dest.with_suffix(ext).unlink(missing_ok=True)
        print(f"wrote {dest.with_suffix('.pdf')}")
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("comparison")
    ap.add_argument("--previous", help="an earlier comparison JSON, for a then-and-now table")
    ap.add_argument("--out", help="the .tex to write (default docs/brms-parity/reports/benchmark-<date>.tex)")
    ap.add_argument("--pdf", action="store_true", help="also run pdflatex twice and remove the auxiliary files")
    a = ap.parse_args()
    main(a.comparison, a.previous, a.out, a.pdf)
