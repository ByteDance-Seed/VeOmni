# Docker

This directory ships the Dockerfiles used to build VeOmni runtime images.

```
docker/
├── matrix.yaml            variant matrix (versions, toggles)
├── generate.py            PEP 723 script that renders Dockerfiles
├── templates/             Jinja2 source templates
│   └── cuda.Dockerfile.j2
├── cuda/                  ← generated, do not hand-edit
│   ├── Dockerfile.cu128
│   └── Dockerfile.cu129
└── ascend/                hand-maintained for now (see "Ascend" below)
```

## CUDA — generated from a template

`docker/cuda/Dockerfile.*` are **generated** from
[`docker/templates/cuda.Dockerfile.j2`](templates/cuda.Dockerfile.j2) +
[`docker/matrix.yaml`](matrix.yaml). The default shape of the template is the
`cu129` variant; other variants override only the bits that differ
(`base_image`, `release_notes_url`).

CI runs [`check_docker_generate.yml`](../.github/workflows/check_docker_generate.yml)
on every PR that touches `docker/`. It re-renders the templates and fails if the
on-disk Dockerfiles drift from the source — so hand-edits to `cuda/Dockerfile.*`
will be flagged.

### Updating

1. Edit one of:
   - [`matrix.yaml`](matrix.yaml) — `defaults.uv_version` (the pinned uv
     version baked into the image), base image, mirrors, extras list, etc.
     `uv_version` here is the docker-image pin, intentionally a single
     concrete release rather than the range declared in `pyproject.toml`'s
     `[tool.uv].required-version`. Bump in lockstep with that range when
     raising the floor or tightening the ceiling.
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
- name: cu130
  output: docker/cuda/Dockerfile.cu130
  template: cuda.Dockerfile.j2
  base_image: "nvcr.io/nvidia/pytorch:25.09-py3"
  release_notes_url: "https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-09.html"
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
