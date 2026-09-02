# Agent contract

You are working in a long-running research codebase. Sessions are short, the project
is not. Everything below exists because context is finite and compute costs money.

Read this file first, every session.

---

## 1. Wiki-first workflow (mandatory)

**Before any work:**

1. Read `wiki/index.md` to find the relevant pages.
2. Read those pages.
3. Go to raw sources (model code, logs, fits, data) only when the wiki is
   insufficient — and when you do, write what you found back into the wiki.

**Background you do not need to rediscover.** The AutoBayes `reference/` pack covers
the fixed cost of the field — sampler theory and its citations, `brms`→PyMC porting,
the JAX/NumPyro GPU path, diagnostic thresholds, and a failure atlas keyed by symptom.
Read it rather than deriving it again. It is a fixed input: never write findings back
into it. Anything you learn goes in `wiki/` or `research/`.

**Two places for project knowledge, and they are not interchangeable.** `wiki/` is
compiled state, optimized for an agent to retrieve — short pages, cross-referenced,
current. `research/` is long-form and dated: the investigation write-up, the derivation,
the argument a human reads end to end. A finding usually produces both — a `research/`
document with the full reasoning, and a wiki page that states the conclusion and links
to it.

**Before proposing any new model, experiment, or expensive job:**

1. Read `wiki/hypotheses/ruled-out.md`. **Do not revisit anything on it.** If you
   believe a ruled-out approach deserves another look, say so explicitly and give
   the reason the original evidence no longer applies. Do not simply re-propose it.
2. Read `wiki/hypotheses/remaining-viable.md` and prefer those directions.
3. Read `DECISIONS.md` for the full record.
4. Append a decision entry **before** any code is written or any job submitted.
5. Fill in that entry's Result section once the run completes.

**After any run or investigation:**

1. Update the relevant `wiki/` page(s).
2. Update `wiki/index.md` if a status changed.
3. Append one line to `wiki/log.md`.

## 2. Write to the wiki before you run out of context

Do **not** save wiki updates for the end of the session. Context compaction will
erase your working memory first.

- After any significant finding, result, or decision: write it down immediately.
- If you have been working a while and have not written to the wiki yet: stop and
  do it now.
- Before any long-running operation: flush what you know first.
- When context is getting large: proactively write everything you have learned.

If you learned it this session and it is not in the wiki, it did not happen.

## 3. The decision gate

**No entry in `DECISIONS.md`, no job.**

Every entry answers four questions, and an entry that does not is not an entry:

1. **What** are we changing, and **why** do we believe it will help?
2. **How will we know if it worked?** A specific diagnostic or metric with a
   threshold, decided before the run. Not "we will look at the output".
3. **What is the kill criterion?** The outcome that means this *direction* is dead, not
   just this run. Separate from (2) on purpose: a run can miss its success threshold
   and still be worth iterating on, and the distinction is only honest if you draw it
   beforehand.
4. **What do we do if it fails?** A concrete next action, not "try something else".

Use `python3 scripts/new_decision.py "<short title>"`. It will refuse to write an
entry with a field left empty.

The point is not the record. The point is that writing (2) before the run regularly
reveals that the run as specified cannot produce the diagnostic — which is much
cheaper to discover now.

## 4. Hard rules

> **These are the project-specific ones. Replace them with yours.** They are the
> part of this file that does real work; the examples below are the originals from
> the project this harness came out of, kept as worked examples of the right
> altitude and specificity.

- **Clone, don't edit in place.** To change a working script, copy it with a
  version suffix and edit the copy. Never edit a known-good script in place.
- **Never overwrite canonical data files.** Write derived files to new locations;
  leave originals untouched.
- **Fresh output directory per run.** A new fit never writes into an existing fit
  directory. The trace already there is a comparison baseline.
- **Never transfer fitted objects between machines.** Regenerate where needed. They
  are large, they go stale silently, and a stale trace looks exactly like a fresh
  one.
- **Read-only dependencies stay read-only.** Source from the upstream repo, never
  edit it; write overrides locally.
- **Flag any training or fine-tuning run before launching it.** Sampling and
  inference runs just get narrated.
- **Usage limits are a hard stop.** Persist state — job IDs, output paths, what was
  in flight — report, and resume after reset. Do not retry into the limit.

## 5. Running jobs

- Templates are in `cluster/`. Read `cluster/README.md` before writing a new one;
  the failure modes that cost the most time are already encoded there.
- Run `python3 cluster/preflight.py` before queueing anything expensive. It builds
  the model, takes one log-density and one gradient, and asserts both are finite.
  Thirty seconds on a login node beats an hour in the queue followed by a
  seven-second failure.
- Record the job ID immediately and report it.
- Monitor with accounting and log tails. Do not poll faster than the job can
  plausibly change state.
- On a failure inside the first minute, read the error log tail **first**. Most
  fast failures are environment, not model.
- On completion, register the run: `python3 scripts/run_registry.py add ...`.
  Configuration, output directory, and wall clock, in one place. You will be asked
  "how long did that take" months later and the logs will have rotated.

## 6. Reporting

After a working session, state: what changed (one line per file), what you
deliberately left alone, and what needs following up. Do not narrate the process.

When you are uncertain, say so before the claim, not after it. A hedge after a
confident sentence does not undo the confident sentence.
