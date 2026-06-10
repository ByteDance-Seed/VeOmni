# Docker

This directory ships the Dockerfiles used to build VeOmni runtime images.

```
docker/
├── matrix.yaml            variant matrix (versions, toggles)
├── generate.py            PEP 723 script that renders Dockerfiles
├── templates/             Jinja2 source templates
│   ├── cuda.Dockerfile.j2
│   └── ascend.Dockerfile.j2
├── cuda/                  ← generated, do not hand-edit
│   └── Dockerfile.cu130
└── ascend/                mix of generated + hand-maintained (see "Ascend" below)
    ├── Dockerfile.ascend_8.3.rc2_a2.x86   ← generated, do not hand-edit
    └── Dockerfile.ascend_*                  hand-maintained (pip-based)
```

## CUDA — generated from a template

`docker/cuda/Dockerfile.*` are **generated** from
[`docker/templates/cuda.Dockerfile.j2`](templates/cuda.Dockerfile.j2) +
[`docker/matrix.yaml`](matrix.yaml). The default shape of the template is the
`cu130` variant; other variants would override only the bits that differ
(`base_image`, `release_notes_url`).

CI runs [`check_docker_generate.yml`](../.github/workflows/check_docker_generate.yml)
on every PR that touches `docker/`. It re-renders the templates and fails if any
on-disk generated Dockerfile drifts from the source — so hand-edits to a
generated file (`cuda/Dockerfile.*` or the templated `ascend/` image) will be
flagged.

### Updating

1. Edit one of:
   - [`matrix.yaml`](matrix.yaml) — base image, mirrors, `uv_version`
     (concrete pin; must stay inside `pyproject.toml` `required-version`).
   - [`templates/cuda.Dockerfile.j2`](templates/cuda.Dockerfile.j2) —
     structural change shared by all variants.
2. Regenerate:
   ```bash
   python docker/generate.py
   ```
3. Commit both the source change and the regenerated `cuda/Dockerfile.*`.

### Adding a new CUDA variant

Add an entry under `images:` in [`matrix.yaml`](matrix.yaml). Example:

```yaml
- name: cu131
  output: docker/cuda/Dockerfile.cu131
  template: cuda.Dockerfile.j2
  base_image: "nvcr.io/nvidia/pytorch:26.02-py3"
  release_notes_url: "https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-02.html"
  uv_extras_preset: cuda
```

Then re-run `python docker/generate.py`.

### Local prerequisites

`generate.py` only needs `jinja2` and `pyyaml`. The repo `.venv` already has
both (transitively), so the simplest invocation from a fresh checkout is:

```bash
uv sync                       # one-time, sets up .venv
.venv/bin/python docker/generate.py
```

If you don't want to set up the project venv, install the two packages into
any Python 3.11+ interpreter and run plain `python`:

```bash
pip install jinja2 pyyaml
python docker/generate.py
```

PEP 723 inline metadata at the top of `generate.py` also lets you do
`uv run --no-project --script docker/generate.py` **from outside the repo
root** (the project's `[tool.uv].required-version` pin would otherwise force
an exact uv version on you).

## Ascend

The ascend images split into two groups:

- **`Dockerfile.ascend_8.3.rc2_a2.x86`** — uv-based, **generated** from
  [`templates/ascend.Dockerfile.j2`](templates/ascend.Dockerfile.j2) +
  [`matrix.yaml`](matrix.yaml), same workflow and CI drift check as the CUDA
  images. Don't hand-edit it; edit the template / matrix and re-run
  `python docker/generate.py`.
- **`Dockerfile.ascend_*_a2.arm` / `Dockerfile.ascend_*_a3`** — still
  **hand-maintained**. They use `pip install -e .[npu_aarch64]` and build
  `torchcodec` from source against CANN, which doesn't map onto the uv-based
  template, so they're intentionally left out of the matrix.

The ascend template reuses the same context as CUDA plus two ascend-specific
knobs in the `matrix.yaml` image entry:

- `header_comments` — the `# docker hub: ... / # cann web: ... / # git repo: ...`
  provenance lines rendered above `FROM`.
- `uv_extra_flags` — raw flags appended to `uv sync` (the CANN base image needs
  `--allow-insecure-host github.com` / `--allow-insecure-host pythonhosted.org`
  to fetch the github / pythonhosted wheel sources that `--locked` revalidates).

To template another uv-based ascend variant (e.g. the `9.0.0_a2.x86` image),
add an entry under `images:` pointing at `ascend.Dockerfile.j2` with the right
`base_image` / `header_comments`, then re-run `python docker/generate.py`.
