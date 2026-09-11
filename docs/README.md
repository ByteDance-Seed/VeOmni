# VeOmni documents

## Build the docs

Use Python 3.12, matching the `Check docs build` GitHub Actions workflow. The lock file pins the
complete dependency closure used by CI. Run these commands from the repository root.

```bash
# Install dependencies.
python -m pip install -r docs/requirements-lock.txt

# Build the docs with the same warnings-as-errors contract as CI.
make -C docs clean
make -C docs html SPHINXOPTS=-W
```

## Open the docs with your browser

```bash
python -m http.server -d docs/_build/html/
```
Launch your browser and open localhost:8000.
