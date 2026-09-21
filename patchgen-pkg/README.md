# patchgen

AST-based patchgen for HuggingFace modeling files. Discovers
`*_patch_gen_config.py` files in your project, rewrites the targeted
`transformers` modeling module, and writes the result to a `generated/`
sibling directory.

## Install

```bash
pip install patchgen
```

## Configure

Add to your project's `pyproject.toml`:

```toml
[tool.patchgen]
search_root = "myproj/models"
package_prefix = "myproj.models"
ruff_isolated = true
ruff_extra_ignore = ["E501"]
```

## Run

```bash
patchgen --list                  # discover configs
patchgen --all --diff            # regen everything
patchgen <patch_module> -v       # regen one
patchgen --check                 # drift gate (CI)
patchgen --check --fix           # adopt regen output
```
