# Wiki schema

> Adapted from [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).
> The wiki is a persistent, compounding knowledge artifact maintained by the agent.

## Architecture

```
raw sources (READ-ONLY)         compiled wiki (AGENT-MAINTAINED)
────────────────────────        ────────────────────────────────
models/                  ──►    wiki/models/          one page per model iteration
experiments/             ──►    wiki/concepts/        core technical concepts
data/                    ──►    wiki/hypotheses/      tested and remaining hypotheses
logs/                    ──►    wiki/bugs/            known bugs and their fixes
DECISIONS.md             ──►    wiki/infrastructure/  deployment and tooling
                                wiki/index.md         master catalog
                                wiki/log.md           append-only activity log
```

**Raw sources are immutable.** The agent reads from them and never modifies them.
All compiled knowledge goes into `wiki/`.

## Operations

**Ingest** — when new information arrives:
1. Read the raw source.
2. Write or update the relevant wiki page.
3. Add cross-references (`[[page-name]]`) both ways.
4. Update the `wiki/index.md` catalog entry.
5. Append one line to `wiki/log.md`.

**Query** — when answering a question about the project:
1. Read `wiki/index.md`.
2. Read the specific pages needed.
3. Fall through to raw sources only if the page is insufficient.
4. If the raw source reveals something not in the wiki, write it back.

**Lint** — periodically, and before any handoff:
```bash
python3 scripts/wiki_lint.py
```
Checks for broken `[[links]]`, stale `PENDING` claims, orphan pages missing from
the index, and ruled-out hypotheses that have reappeared as active proposals.

## Page format

Every page uses this structure. `templates/wiki-page.md` is the copyable version.

```markdown
# Page title

**Status:** Active | Resolved | Failed | Superseded by [[other-page]]
**Last updated:** YYYY-MM-DD

## Summary
Two or three sentences. What a reader needs if they read nothing else.

## Detail
The substance.

## Evidence
Numbers, diagnostics, file paths. Anything asserted here must be checkable.

## Related
[[other-page]], [[another-page]]
```

## Rules

- A claim without a number or a file path is not evidence.
- `PENDING` carries the date it was written. `wiki_lint.py` flags stale ones.
- Never delete a page. Mark it `Superseded by [[...]]` and keep it — the record of
  what was tried is the point.
- Cross-reference in both directions. A page nothing links to will not be found.
