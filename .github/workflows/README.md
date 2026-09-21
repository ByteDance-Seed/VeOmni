### Adding a New Workflow

When adding a new workflow for continuous integration (CI), you have two runner options: a fixed runner or a machine from the vemlp.

- **Fixed Runner**: To use a self-hosted fixed runner, specify it via the `runs-on` keyword. Common label sets:
  - GPU, any free L20-8 host: `runs-on: [self-hosted, l20-8]`
  - GPU, a specific host (e.g. for repro): `runs-on: [self-hosted, l20-3]`
  - NPU, any free 910B-8 slice: `runs-on: [self-hosted, 910b-8]`
  - NPU, a specific 8-card slice: `runs-on: [self-hosted, 910b-2]`
  - NPU, a specific physical machine (either of its two splits): `runs-on: [self-hosted, 910b-host-1]`

  See `github_runner/README.md` for the full label scheme.
- **Vemlp Runner**: Opting for a Vemlp machine allows you to launch tasks elastically.

Here is a template to assist you. This template is designed for using Vemlp machines. Currently, for each workflow, you need to create a `setup` and a `cleanup` job. When using this template, the main parts you need to modify are the `IMAGE` environment variable and the specific `job steps`.

```yaml
name: Your Default Workflow

on:
  push:
    branches:
      - main
      - v0.*
  pull_request:
    branches:
      - main
      - v0.*
    paths:
      - "**/*.py"
      - ".github/workflows/template.yml"

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}

permissions:
  contents: read

env:
  IMAGE: "your vemlp image" # e.g. "verl-ci-cn-beijing.cr.volces.com/verlai/verl:sgl055.dev2"
  DYNAMIC_RUNNER_URL: "https://sd4hav466omp034ocfn0g.apigateway-cn-beijing.volceapi.com/veomni/runner" # public veFaas api

jobs:
  setup:
    if: github.repository_owner == 'ByteDance-Seed'
    runs-on: ubuntu-latest
    outputs:
      runner-label: ${{ steps.create-runner.outputs.runner-label }}
      task-id: ${{ steps.create-runner.outputs.task-id }}
    steps:
      - uses: actions/checkout@v4
      - id: create-runner
        uses: volcengine/vemlp-github-runner@v1
        with:
          mode: "create"
          faas-url: "${{ env.DYNAMIC_RUNNER_URL }}"
          image: "${{ env.DEFAULT_IMAGE }}"

  your_job:
    needs: setup
    runs-on: ["${{ needs.setup.outputs.runner-label || 'default-runner' }}"]
    steps:
      xxxx # your jobs

  cleanup:
    runs-on: ubuntu-latest
    needs: [setup, your_job]
    if: always()
    steps:
      - id: destroy-runner
        uses: volcengine/vemlp-github-runner@v1
        with:
          mode: "destroy"
          faas-url: "${{ env.DYNAMIC_RUNNER_URL }}"
          task-id: "${{ needs.setup.outputs.task-id }}"
```

### Model and Dataset
To avoid CI relies on network, we pre-download dataset on a NFS on the CI machine. The path for models are \${HOME}/models and the path for dataset is \${HOME}/models/hf_data.
