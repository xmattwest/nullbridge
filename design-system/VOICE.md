# Voice

**Say what it is, show the number, stop.**

This guide comes from what Matt West kept and cut: 2,303 OpenCode prompts across 51 projects (June to November 2026), the approved copy on the Dawn State site, the personal homepage and OpenCode Mirror, and the edits he asked agents to make. The same handful of corrections came up in project after project. They are rules now, so nobody has to ask again.

## The rules

1. **Say what it is in the first line.** Use words a smart outsider already knows. A STROBE entry that never says "protein" has failed, however precise it is.
2. **Keep the workshop out of public copy.** Gates, phases, checkpoints, run IDs, claim IDs and internal codenames stay in plans and handoffs. Public copy says what was done and what was found.
3. **Let the number carry the confidence.** Give the figure, its unit and its scope. Drop the adjective that was doing the number's job.
4. **Write one point where you drafted three.** If a section has three bullets, check whether one of them is the point and the other two are padding.
5. **Vary sentence length.** Short sentences are welcome. A paragraph of identical medium-length sentences reads as generated.
6. **No em dashes.** Use a period, comma, colon or parentheses. This applies to published prose, UI copy, captions and social posts.
7. **Define a technical term once, in passing.** "LLM behavioral economics, which studies how large language models make economic decisions." Then use the term.
8. **Disclose what, not how.** For Dawn State, describe capability and results. Methods, model stacks and simulation names are trade secrets unless you choose to publish them.
9. **No launch theatre.** No "Introducing", no slogans as headings, no status dots or badges as decoration, no emojis, no titles on social posts.
10. **End handoffs with the next move.** A command, a link, a question or a decision. Not a summary of what was just said.

## Registers

One set of facts, three ways of speaking. Pick the register from the reader, not the project.

| Register | Reader | Sounds like | Avoid |
| --- | --- | --- | --- |
| Dawn State | Partners, funders, reviewers | Confident, factual, sparing. "Proteins move. Most screening workflows begin with a single static structure." | Process detail, hedged adjectives, method disclosure |
| Matt West | People who found the site or a post | Dry, specific, a little odd, not too serious. "I work in commodities, do AI research, and tend to take an interesting question a little further than I meant to." | Career-page polish, mission statements, humblebrag lists |
| Research | Quantitative readers, traders, reviewers | Neutral and exact. What was done, what was found, how sure, what it does not show. | Experiment tracking language, enthusiasm, causal claims the design cannot support |

Prompts to agents are often lowercase shorthand. That is a working register between a person and a tool. Published work does not copy it.

## Kept and cut

Real edits, before and after.

| Before | After | What changed |
| --- | --- | --- |
| Calibrated on 39 experimentally validated apo/holo pairs (Gate 0 passed with 27/43 judge-verified captures), scaling to proteome-scale agricultural and fungal targets. | Verified on 39 experimentally validated apo/holo pairs with 27 judge-verified open-state captures, scaling across proteome-scale agricultural and fungal targets. | The gate is gone; the numbers stay. |
| The research combines LLM behavioral economics—how large language models make economic decisions—with mechanistic interpretability: looking inside one model for clues about what drives its choices. | The research combines LLM behavioral economics, which studies how large language models make economic decisions, with mechanistic interpretability, which looks inside one model for clues about what drives its choices. | Em dashes become definitions in passing. |
| Your work, reflected. | OpenCode Mirror. Your working style, conversation fingerprint and daily rhythm, drawn from your local archive. | A slogan becomes a name and a description. |
| Introducing STROBE (with a glowing dot) | STROBE / generative conformational sampling | Launch theatre becomes a plain label. |
| Three bullets on continuous learning | One sentence | "We don't need three points. Just one concise one will do." |

## Phrases to cut on sight

- **Inflated verbs:** spearheaded, leveraged, synergized, elevated, empowered, unlocked, revolutionized, transformed.
- **Empty adjectives:** robust, seamless, cutting-edge, holistic, dynamic, pivotal, passionate, world-class, powerful.
- **Stock frames:** "In today's fast-paced world", "It's not just X, it's Y", "Here's the thing", "Let's dive in", "The result?", "game-changer", "at the intersection of".
- **Self-description:** results-driven, detail-oriented, thought leader, track record, seasoned.
- **Hedge stacks:** "may potentially", "could possibly help to", "it is important to note that".

A banned word is a symptom. Replace it with the specific fact it was hiding, or delete the sentence.

## By document

- **Headlines and titles** state the subject or the finding in plain words. A short display line may end with a period, as Mirror's "Builder." does. Display type is for that one line; everything else uses ordinary headings.
- **Ledes** say who it is for and what they get, in one or two sentences.
- **Figure titles** state the finding. Subtitles state the measure, unit, sample and period. See [data.md](adapters/data.md).
- **Buttons and links** name the action or the destination: "Download Mirror", "Read the preprint". Never "Learn more" or "Click here".
- **Errors** name the problem and the fix.
- **Social posts** have no title, no hashtag block, no emojis. Open with the finding or the odd detail. Keep it short and a little humble.
- **Applications and forms** answer the question asked, at the length asked, in the order asked.
- **Handoff prompts** read like a work order: files to read, goal, where we stopped, constraints, first action.

## Check before publishing

- Does the first line say what this is, for someone who has never heard of it?
- Is there any gate, phase, run ID or codename a stranger would not understand?
- Is there an em dash anywhere?
- Is every adjective earning its place, or could a number replace it?
- Read it aloud. Does every sentence have the same rhythm?
- Would you be comfortable if the most expert person in the field read it?
