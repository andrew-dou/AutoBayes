# research/

Long-form, dated write-ups. The investigation with its reasoning intact, the
derivation, the argument someone reads end to end.

## How this differs from `wiki/`

They are not redundant, and using one for the other's job degrades both.

|  | `wiki/` | `research/` |
| --- | --- | --- |
| Read by | an agent, retrieving | a human, following an argument |
| Length | short — a screen | as long as the argument needs |
| Tense | current state | what was true when written |
| Edited | in place, kept current | never; superseded by a new document |
| Answers | "what do we believe now?" | "how did we come to believe it?" |

A finding usually produces both: a `research/` document with the full reasoning, and a
wiki page stating the conclusion and linking to it. Compressing a real investigation
into a wiki page loses the reasoning that makes it checkable. Leaving it only in
`research/` means nobody finds it — a document nothing links to does not exist.

## Convention

Number sequentially, never reuse a number, never renumber:

```
research/
  01_prior_art_and_approach.md
  02_data_provenance_audit.md
  03_funnel_diagnosis.md
  04_censored_likelihood_derivation.tex
```

The number is a stable handle. `DECISIONS.md` entries and wiki pages cite documents by
number, and renumbering silently breaks every one of those references.

Markdown or LaTeX both work. LaTeX earns itself once there is real mathematics; keep
the compiled PDF alongside the source.

## What belongs here

- The write-up of an investigation, including the paths that did not work
- A derivation the model depends on
- A comparison between two approaches, with the numbers
- A teaching document explaining a subsystem to a collaborator
- Prior art for a specific problem — as distinct from the general background in the
  AutoBayes `reference/` pack, which is a fixed input you do not add to

## What does not

- Current status of anything — that is `wiki/`, and a status claim here goes stale
  invisibly
- Decisions and their rationale — that is `DECISIONS.md`
- Raw output. Reference it by path; do not paste it

## Rules

- **Date every document** and never silently revise a conclusion. If it is wrong,
  write a new document and mark the old one superseded. The record of what was believed
  when is frequently the thing that explains a later result.
- **Assertions carry a number or a file path.** A claim with neither is not a finding.
- **Negative results belong here in full.** They are the least reconstructible thing
  you produce, and the reason the next agent does not repeat the month you just spent.
  Record the mechanism, not only the outcome — "it did not work" does not stop anyone.
