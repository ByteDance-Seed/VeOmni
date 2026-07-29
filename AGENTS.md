# VeOmni Development Guide

> Instructions for AI coding agents working on this repository.

**VeOmni** is a modular distributed training framework for multi-modality models (text, vision, audio, diffusion, omni) across various accelerators (GPUs, NPUs). Developed by ByteDance Seed Team.

- Homepage: https://github.com/ByteDance-Seed/VeOmni
- Python: `>=3.11, <3.13`
- Package: `veomni`

**Language**: Match user's language (English).

## Context Loading

On session start, read the following:
- `.agents/knowledge/constraints.md` — hard constraints to check before any code change
- `.agents/knowledge/architecture.md` — module map, trainer hierarchy, data flow
- `.agents/knowledge/uv.md` — dependency management architecture (uv, extras, lockfile)

---

## Core Principles

- **Challenge First, Execute Second**: Spot logic flaws or simpler alternatives? Raise concerns before executing.
- **Explain, Don't Assume**: Explain **why** (motivation, tradeoffs), not just what. Cite files and line numbers.
- **Ask When Stuck**: 3+ approaches fail? Stop, summarize, ask user. No hacks.
- **Search Before You Act**: On unexpected behavior, search codebase + check constraints + review `git log` before attempting fixes.
- **Planning Discipline**: Complex tasks (multi-file, >30 min) -> TodoWrite. Plan must state which skills will be used (e.g. `/veomni-develop` + `/veomni-review`). Simple tasks -> just do them.
- **Cross-modality Awareness**: Changes in shared code (`BaseTrainer`, `data_collator`, `distributed/`) affect all modalities.
- **No Patchgen Edits**: Never edit files under `veomni/models/transformers/*/generated/`.

---

## Setup

```bash
uv sync --extra gpu --dev
source .venv/bin/activate
```

This installs `transformers==5.9.0` via the `transformers-stable` dependency
group. `gpu` / `npu` / `npu_aarch64` are the only extras — each a complete
superset, mutually exclusive. New code must target transformers v5 and FSDP2.
See `.agents/knowledge/uv.md` and `.agents/knowledge/constraints.md`.

---

## Development Commands

```bash
source .venv/bin/activate
make style          # ruff fix + format
make quality        # ruff check (CI gate)
make commit         # style + quality
make patchgen       # regenerate model patches
pytest tests/       # all tests
pytest tests/<mod>/ # specific module
```

---

## PR Guidelines

Title: `[{modules}] {type}: {description}`

- Allowed modules and types are defined in `.github/workflows/check_pr_title.yml` (the CI source of truth).
- Breaking: prepend `[BREAKING]`

---

## Commit Flow

1. Complete and verify the change.
2. Update related documentation: `docs/`, `README.md`, `.agents/knowledge/`, config examples — if the change introduces, modifies, or removes any API, config field, or workflow.
3. Run `/veomni-review` skill (subagent code review).
4. **safe** -> commit. **risky** -> report to user, wait for approval.
5. Each fix -> immediate commit. Do not batch unrelated changes.
6. Run `make quality` before every commit.
7. **Commit messages must NOT mention Claude/AI/Co-Authored-By.**
8. **Skill gap check**: If the task didn't match any existing skill, briefly assess after completion: Was this a one-off, or a repeatable pattern? If repeatable, suggest creating a new skill to the user.

---

## Skills

Skills follow the [Agent Skills](https://agentskills.io) open standard. Each skill is a folder in `.agents/skills/<name>/` containing a `SKILL.md` with YAML frontmatter (`name`, `description`). Skills are auto-discovered by compatible agents (Cursor, Claude Code, Codex, etc.) and can also be invoked manually with `/skill-name` in chat.

| Task | Skill |
|------|-------|
| Feature / refactoring | `/veomni-develop` |
| Bug fix / debugging | `/veomni-debug` |
| Code review (pre-commit) | `/veomni-review` |
| Add new model | `/veomni-new-model` |
| Migrate existing model to transformers v5 | `/veomni-migrate-transformers-v5` |
| Add new op/kernel | `/veomni-new-op` |
| Update dependencies (uv) | `/veomni-uv-update` |
| Performance profiling | `/veomni-profile` |
| Create or update a pull request | `/create-pr` |

### Quick Decision Guide

- **"Add support for model X"** → `/veomni-new-model`
- **"Migrate X to transformers v5" / "port X to patchgen" / "convert monkey patch to generated modeling"** → `/veomni-migrate-transformers-v5`
- **"Add a new kernel / fused op"** → `/veomni-new-op`
- **"Fix this error" / "training hangs" / "wrong results"** → `/veomni-debug`
- **"Add a new capability" / "refactor" / "clean up"** → `/veomni-develop`
- **"Update package X" / "bump uv" / "upgrade torch"** → `/veomni-uv-update`
- **"Analyze this trace" / "why is training slow" / "profile" / "MFU"** → `/veomni-profile`
- **"Create a PR" / "submit PR"** → `/create-pr`

---

## Cursor Cloud specific instructions

The Cursor Cloud VM is **CPU-only (no GPU/NPU), x86_64, Python 3.12**. The
startup update script already runs `uv sync --extra gpu --dev`, so `.venv` is
ready — activate it with `source .venv/bin/activate` before any command. `uv`
is installed under `~/.local/bin` (on `PATH` via `~/.bashrc`); the `gpu` extra's
CUDA torch wheels install fine here and `torch.cuda.is_available()` is `False`.

**What works CPU-only (use these to validate changes without hardware):**
- Lint gate: `make quality` (and `make style` to auto-fix) — see `Makefile`.
- Patchgen drift: `patchgen --check` (CI equivalent of the check_patchgen job).
- Device API check: `python tests/special_sanity/check_device_api_usage.py -d {veomni,tasks,tests}`.
- The CPU subset of `pytest` (registry/ops-gate/eager/data/lora-unit/converter/
  balance/DPO/checkpoint-callback tests). GPU-only tests self-skip via
  `IS_CUDA_AVAILABLE` / `@pytest.mark.skipif(device_count < N)`, but many
  files in `tests/models`, `tests/distributed`, `tests/e2e`, and most
  `tests/parallel/ulysses` and `tests/lora` integration tests need real
  GPUs — do not expect the full `pytest tests/` (a.k.a. `make test`) to pass here.

**What does NOT work here:** real training (`train.sh` / `tasks/train_*.py`),
fused CUDA/Triton kernels (flash-attn, quack, tilelang, FlashMLA), and any
multi-GPU/FSDP2/Ulysses/EP test. Those require the self-hosted GPU CI runners.

**Gotchas:**
- To build a model on this box, force all-eager ops and `init_device="cpu"`
  (flash-attn/triton are unavailable). `tasks/infer/*.py` show the eager
  `OpsImplementationConfig` pattern (`is_flash_attn_2_available()` → `eager`).
- `make build` targets a non-existent `setup.py`; packaging is via
  `pyproject.toml` (`python -m build`), not `make build`.
- Re-running `uv sync` is cheap and idempotent; prefer it over `pip install`.

### Lark / Feishu notifications for PR review tasks

When a cloud-agent **review task** is triggered by a new PR on the remote repo,
notify the maintainer on Feishu/Lark with a short PR summary (title, author,
link, key changes, review verdict).

- **Provisioning:** `scripts/cloud/lark_notify_setup.sh` runs on VM startup (wired
  into the environment update script). It installs `@larksuite/cli` globally to
  `~/.npm-global` (on `PATH` via `~/.bashrc`), adds the `lark-shared` + `lark-im`
  skills to `~/.agents/skills/`, and configures a **bot** identity. It is a
  no-op (exit 0) when the Lark secrets are absent, so startup never breaks.
- **Required secrets:** `LARK_APP_ID`, `LARK_APP_SECRET` (bot app credentials);
  optional `LARK_BRAND` (`feishu` or `lark`, default `feishu`). Set a recipient
  the bot can reach, e.g. `LARK_NOTIFY_RECEIVER` (an `open_id` / `user_id` /
  `email` / `chat_id`) — the maintainer must have opened a chat with the bot or
  be in a shared group for DMs to deliver.
- **How to send (verified):** first `Read` `~/.agents/skills/lark-im/SKILL.md` and
  `~/.agents/skills/lark-shared/SKILL.md`. Self-check with `lark-cli doctor`
  (`bot_identity` must be `pass` / `ok: true`) — if it fails, the secrets are
  wrong (see below). Then send with the bot identity, e.g. to a group chat:

  ```bash
  lark-cli im +messages-send --as bot --chat-id "$LARK_NOTIFY_RECEIVER" \
    --markdown "**PR review summary** ...markdown here..."
  ```

  Use `--chat-id` for an `oc_...` value and `--user-id` for an `ou_...` value
  (do NOT also pass `--msg-type`; `--markdown` auto-infers `post`). Discover the
  target group with `lark-cli im +chat-list --as bot` (the review bot lives in
  the **Open-VeOmni-PR-Review** group).
- **Verified config:** `LARK_BRAND` must be **`feishu`** (this app lives on
  `open.feishu.cn`; `lark`/larksuite.com returns `invalid_client`). The four
  Lark secrets are DISTINCT values — do not reuse the `cli_...` App ID for the
  App Secret or the receiver.
- **Gotcha:** `npm i -g` needs the user prefix `~/.npm-global` (set in
  `~/.npmrc`); the setup script exports it so it never hits `EACCES`.

**Review-task flow (self-contained — works even on PR branches predating this
doc):** on a review task, after posting the GitHub review comment, send a group
summary. This does not rely on any repo file:

1. Self-check: `lark-cli doctor`. If `bot_identity` is not `pass`, configure from
   the injected secrets first:

   ```bash
   printf '%s' "$LARK_APP_SECRET" | lark-cli config init \
     --app-id "$LARK_APP_ID" --app-secret-stdin --brand feishu
   lark-cli config default-as bot
   ```

2. Send to the review group — the `$LARK_NOTIFY_RECEIVER` secret holds its
   `oc_...` chat id (the **Open-VeOmni-PR-Review** group; discover via
   `lark-cli im +chat-list --as bot` if unset). Use `--markdown`, no `--msg-type`:

   ```bash
   lark-cli im +messages-send --as bot --chat-id "$LARK_NOTIFY_RECEIVER" \
     --markdown "<PR #, title, author, branch, link; one line per review
     criterion (solves-problem / comments-docs-tests / cleanliness-conflicts);
     overall verdict + 1-3 key issues>"
   ```

3. Confirm the response is `"ok": true`; if not, surface the error rather than
   skipping silently.
