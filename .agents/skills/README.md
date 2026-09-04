# VeOmni Agent Skills

Reusable workflow definitions for AI coding agents working on VeOmni. Skills follow the [Agent Skills](https://agentskills.io) open standard and are auto-discovered by compatible agents (Cursor, Claude Code, Codex, Junie, etc.).

## Structure

Each skill is a folder containing a `SKILL.md` with YAML frontmatter:

```
.agents/skills/
├── veomni-develop/
│   └── SKILL.md          # name + description frontmatter, then instructions
├── veomni-debug/
│   └── SKILL.md
├── veomni-patchgen-model/
│   ├── SKILL.md          # the spine: applies to every model
│   └── references/       # loaded on demand, per model category
│       ├── model-examples.md
│       ├── moe.md
│       └── multimodal.md
└── ...
```

Agents use the `description` field to decide when a skill is relevant. Users can also invoke skills manually with `/skill-name` in chat.

## Keeping a skill readable

A skill is read in full every time it fires, so length is a real cost. When a
skill grows past a few hundred lines, check whether the bulk of it is
*conditional* — only relevant to some subset of the cases it covers. If so,
move each conditional block into `references/` and leave a routing table near
the top of `SKILL.md` saying which file to load when.

`veomni-patchgen-model` is the worked example: it was 1055 lines, of which
roughly a third only applied to MoE models and another third only to VLM/Omni
models. A text-only dense model now reads the ~710-line spine and skips both.
Keep the spine self-contained — a reference file adds to a phase, it never
replaces one, so nobody has to reconstruct the procedure from fragments.

## Adding a Skill

1. Create `.agents/skills/<skill-name>/SKILL.md` with `name` and `description` frontmatter.
   - `name` must match the folder name (lowercase, hyphens only).
   - `description` should explain what the skill does and when to use it.
2. Add the skill to the dispatch table in `AGENTS.md`.
3. If the skill requires domain knowledge, add it to `.agents/knowledge/`.
4. Optional: add `scripts/`, `references/`, or `assets/` subdirectories.

See the [Agent Skills specification](https://agentskills.io/specification) for the full format.

## Skill Index

The dispatch table in [`AGENTS.md`](../../AGENTS.md) is the single index — it maps
a task to the skill to use, and every agent already loads it. Each skill's own
`description` frontmatter is the authoritative statement of when it applies.

Keeping a second copy of the index here only produced drift, so it is gone.
