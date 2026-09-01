# Why this harness

Notes on the reasoning, kept separate from the instructions so `AGENTS.md` stays
short enough that an agent actually reads all of it.

## The failure this is built around

A research program that runs for months, with an agent doing most of the typing,
accumulates knowledge much faster than any context window holds it. The knowledge
that matters most is negative: the twelve parameterizations that did not fix the
funnel, the covariate whose posterior was indistinguishable from zero, the sampler
setting that looked like a win and was a measurement artifact.

Negative results are exactly what an agent will not reconstruct. Asked whether an
approach was tried, an agent with no record does not say "I don't know" — it
produces a plausible answer. Plausible is worse than absent, because you act on it.

So the harness is organized around one question: *where does a negative result get
written down, such that the agent is required to read it before proposing work?*
The answer is `wiki/hypotheses/ruled-out.md`, and `AGENTS.md` makes reading it a
precondition. Everything else supports that.

## Why the decision gate is upstream of the run, not downstream

Experiment trackers record what a run did. That is necessary and not sufficient,
because by the time you are recording, the compute is spent.

The gate asks for the diagnostic and its threshold *before* submission. In the
original project this changed the run more often than it changed the record:
writing "how will we know it worked" would reveal that the planned run could not
produce the diagnostic — the output it saved did not contain the quantity, or the
comparison needed a baseline that had never been fit. Discovering that costs
nothing on a login node and half a day in the queue.

The third field — what we do if it fails — exists because "try something else" is
how a research program turns into a random walk. Naming the next action in advance
means a failed run still advances the program.

## Why "clone, don't edit in place"

An agent editing a working script in place, in a repository without branches, can
destroy the only copy of something that worked, in a way that is invisible until
the next run produces different numbers for a reason nobody can reconstruct. The
rule is cheap and the failure is expensive and silent. It costs disk and a suffix.

## Why fitted objects never move between machines

They are large, they go stale invisibly, and a stale trace looks exactly like a
fresh one. Every quantity derived from one is then wrong in a way no diagnostic
catches, because the trace itself is internally consistent — it is just the answer
to a question you stopped asking three model versions ago. Regenerate where needed.

## On replicate counts and honest figures

A benchmark with five replicates per cell does not support an uncertainty interval,
and a min-to-max bar drawn across those five points reads as one. Plot the
replicates and the median, print the `n` on the figure, and let the reader see the
spread. If two measurement campaigns disagree — and they will — the disagreement is
the finding, not something to be averaged away.

The same applies to a speedup ratio. A ratio computed across two runs that differ
in model, sampler settings, and hardware is not a device speedup; it is a number.
Report the formulation the ratio was obtained with, or do not report the ratio.

## What was deliberately left out

No workflow engine, no DAG, no scheduler abstraction. Every one of those adds a
layer between the agent and the thing that failed, and the debugging loop is
already long enough. The templates are shell scripts you can read in one screen,
and the scripts are single files with no dependencies beyond the standard library.
