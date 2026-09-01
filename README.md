# bayes-agent-harness

Scaffolding for running a long Bayesian model-iteration program with a coding agent
in the loop, on a cluster you pay for by the GPU-hour.

This is the harness behind an election-forecasting project that fit ~60 successive
versions of a hierarchical model on an NVIDIA H200, each conditional on the last.
The modelling was the research; the harness is what made the modelling survivable
once an agent was doing most of the typing. It is extracted here with the project
specifics removed.

## The problem it solves

An agent with a long-running research codebase has three failure modes that
compound:

1. **It forgets.** Context is finite. By week three the agent no longer knows that
   AR(1) on the year slopes was tried and abandoned, so it proposes AR(1) on the
   year slopes.
2. **It is confidently wrong about the past.** Asked "did we test this?", an agent
   with no durable record will reconstruct a plausible answer. Plausible is worse
   than absent, because you act on it.
3. **It burns real money doing it.** A wrong sbatch script fails seven seconds into
   a queued H200 allocation, and you find out an hour later.

None of these is fixed by a better prompt. They are fixed by writing things down in
a place the agent is required to read, and by making the expensive path go through
a checklist.

## What is here

| Piece | What it does |
| --- | --- |
| `AGENTS.md` / `CLAUDE.md` | The contract the agent reads first. Wiki-first workflow, hard rules, the decision gate. |
| `DECISIONS.md` | An append-only log. **No entry, no job.** Each entry answers what/why, how we will know it worked, and what we do if it fails. |
| `wiki/` | Compiled, agent-maintained knowledge. Raw sources stay immutable; everything learned is written here. Adapted from Karpathy's LLM Wiki pattern. |
| `wiki/hypotheses/ruled-out.md` | The single highest-value file. Approaches already killed, with the evidence. Read before proposing anything. |
| `cluster/` | SLURM templates with the failure modes already encoded, plus a preflight that catches them before you queue. |
| `scripts/` | `new_decision.py` (enforces the gate), `wiki_lint.py` (finds stale and contradictory pages), `run_registry.py` (config → output dir → wall clock). |

## Quickstart

```bash
git clone <this repo> my-project && cd my-project && rm -rf .git && git init
```

Then:

1. Edit `AGENTS.md` — the Hard Rules section is project-specific and is the part
   that actually does the work. Delete the examples, write yours.
2. Point your agent at it (`CLAUDE.md` is a pointer to `AGENTS.md`, so both
   Claude Code and Codex pick it up).
3. Before the first expensive run, `python3 scripts/new_decision.py "first fit"`.

## The decision gate

The rule is one line: **before submitting any job that costs real compute, append an
entry to `DECISIONS.md`.**

It is not bureaucracy, it is a forcing function. Writing "how we will know if it
worked" before the run is what converts *I'll look at the output and see* into a
falsifiable prediction. Half the entries in the original project's log changed the
run before it was submitted, because writing down the diagnostic revealed that the
run as specified could not produce it.

`scripts/new_decision.py` refuses to write an entry with an empty field.

## The wiki

Raw sources — model code, logs, data, fitted objects — are **read-only** to the
agent. Everything the agent learns goes into `wiki/`, cross-referenced, with an
index. On any new question the agent reads `wiki/index.md` first and only falls
through to raw sources when the wiki is insufficient; when it does, it writes what
it found back.

The rule that matters most in practice: **write to the wiki before context runs
out, not at the end of the session.** An agent that waits until it is done loses
everything to compaction. `AGENTS.md` states this explicitly.

`scripts/wiki_lint.py` checks for broken `[[links]]`, pages that claim `PENDING`
long after their run completed, and ruled-out hypotheses that have reappeared as
active proposals.

## Cluster templates

`cluster/slurm_gpu_fit.sbatch` is a working template carrying the traps that cost
the original project the most time:

- `source activate <env>` does **not** export `LD_LIBRARY_PATH` on a compute node.
  Jobs die about seven seconds in on `ImportError: libffi.so.8`. The export is in
  the template, immediately after activation.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` — otherwise JAX grabs the whole device and
  anything else on the node dies.
- Fresh output directory per run, always. Never write into an existing fit
  directory; the trace already there is somebody's comparison baseline.
- `.sh` versus `.sbatch` matters if you dispatch by extension. A misnamed batch
  script runs on the login node and looks exactly like a hung SSH session.

`cluster/preflight.py` runs the cheap version of the job first — build the model,
evaluate one log-density and one gradient, assert both finite, print the free
parameter count — so that a specification error surfaces in thirty seconds on a
login node rather than after an hour in the queue.

## What this is not

Not a workflow engine, not an experiment tracker, not a replacement for MLflow or
W&B. Those record what a run did. This is upstream of that: it constrains what gets
run, and keeps what was learned somewhere an agent will actually find it.

## Licence

MIT.
