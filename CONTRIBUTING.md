# Contributing to VeOmni

Use this guide for code and documentation changes. For the published training
guides, start at the [documentation](https://veomni.readthedocs.io/en/latest/).

## Set up a checkout

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/<your-user>/VeOmni.git
cd VeOmni
git remote add upstream https://github.com/ByteDance-Seed/VeOmni.git
git fetch upstream
git switch -c my-change upstream/main
```

## Set up the environment

Use Python 3.11 or 3.12 and the uv version range declared in
[pyproject.toml](pyproject.toml). For NVIDIA development:

```bash
uv sync --locked --extra gpu --dev
source .venv/bin/activate
pre-commit install
```

For Ascend, follow the [x86](docs/get_started/installation/install_ascend_x86.md)
or [ARM](docs/get_started/installation/install_ascend_arm.md) installation guide
and include `--dev` in the uv sync command. Hardware extras are mutually
exclusive. [ROCm](docs/hardware_support/rocm/README.md) and
[MLU](docs/hardware_support/mlu/README.md) use their own accelerator environments;
do not install the NVIDIA extra into them.

Documentation-only changes need only Python 3.12 and the dependencies in
[docs/README.md](docs/README.md), plus Ruff for `make quality`; they do not require
the training environment. For CPU-only code checks, see the
[environment notes](.agents/knowledge/cpu_only_env.md).

## Make and verify a change

- Keep each change focused, and update the corresponding guide when an API,
  configuration field, or workflow changes.
- Never edit generated modeling under `veomni/models/transformers/*/generated/`
  by hand. Follow the [patchgen guide](docs/design/patchgen.md).
- Run the tests relevant to the changed behavior. The [testing guide](docs/testing.md)
  explains the suites; the [test wiring rules](.agents/knowledge/testing.md)
  explain how to include a new test in CI.

```bash
make style
make quality
python3 scripts/ci/check_doc_task_paths.py
python3 scripts/ci/check_agent_doc_paths.py
```

For documentation changes, also follow the build and authoring checks in
[docs/README.md](docs/README.md). When moving a task, configuration, or page,
update its references in the same change. A passing documentation build does
not verify that a training command runs successfully; report hardware validation
separately.

## Submit a pull request

```bash
git add <changed-files>
git commit -m "[docs] refactor: clarify training documentation"
git push -u origin my-change
```

Open a PR against `ByteDance-Seed/VeOmni:main`, using the
[PR template](.github/PULL_REQUEST_TEMPLATE.md). Explain the problem, resulting
behavior, and validation, including anything that could not be checked.
Titles follow `[{modules}] {type}: {description}`; the allowed values live in
[check_pr_title.yml](.github/workflows/check_pr_title.yml).

When updating your branch, fetch `upstream` and integrate `upstream/main`.
For a stacked PR, target the preceding branch and name the dependency in the
description so reviewers can inspect and merge the stack in order.

## Code review

GitHub PRs use [CodeRabbit](https://docs.coderabbit.ai) when its GitHub App is
installed. Configuration lives in [.coderabbit.yaml](.coderabbit.yaml).
On an existing PR, `@coderabbitai review` requests review of new commits and
`@coderabbitai full review` requests a full review. See `@coderabbitai help` for
the command list.

For coding agents, [AGENTS.md](AGENTS.md) defines the repository constraints,
skills, and pre-PR review gate.
