## Contributing

1. Clone the repo to local disk

```bash
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni
```

2. Create a new branch

```bash
git checkout -b dev_your_branch
```

3. Set up a development environment

```bash
pip install -e ".[dev]"
```

4. Check code before commit

```bash
make commit
make style && make quality
# make test
```

When you **move or rename** scripts under `tasks/`, search the repo for old paths (e.g. `grep -r tasks/` in `docs/`) and update examples. CI runs a lightweight check; validate locally with:

```bash
python3 scripts/ci/check_doc_task_paths.py
```

5. Submit changes

```bash
git add .
git commit -m "commit message"
git fetch origin
git rebase origin/master
git push -u origin dev_your_branch
```

6. Create a merge request from your branch `dev_your_branch`
