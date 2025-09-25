### What does this PR do?

> Add **concise** overview of what this PR aims to achieve or accomplish. Reference related GitHub issues and PRs that help with the review.

### Checklist Before Starting

- [ ] Search for similar PRs. Paste at least one query link here: ...
- [ ] Format the PR title as `[{modules}] {type}: {description}` (This will be checked by the CI)
  - `{modules}` include `misc`, `ci`, `config`, `docs`, `core`, `data`, `dist`, `omni`, `logging`, `model`, `optim`, `ckpt`, `release`, `task`, `
  - If this PR involves multiple modules, separate them with `,` like `[megatron, torch, liger-kernals,fsdp, doc]`
  - `{type}` is in `feat`, `fix`, `refactor`, `chore`, `test`
  - If this PR breaks any API (CLI arguments, config, function signature, etc.), add `[BREAKING]` to the beginning of the title.
  - Example: `[BREAKING][fsdp, torch, liger-kernals] feat: dynamic batching`

### Test

> For changes that can not be tested by CI (e.g., algorithm implementation, new model support), validate by experiment(s) and show results like training curve plots, evaluation results, etc.

### API and Usage Example

> Demonstrate how the API changes if any, and provide usage example(s) if possible.

```python
# Add code snippet or script demonstrating how to use this
```

### Design & Code Changes

> Demonstrate the high-level design if this PR is complex, and list the specific changes.

### Checklist Before Submitting

> [!IMPORTANT]
> Please check all the following items before requesting a review, otherwise the reviewer might deprioritize this PR for review.

- [ ] Read the [Contribute Guide](https://github.com/ByteDance-Seed/VeOmni/blob/main/CONTRIBUTING.md).
- [ ] Apply [pre-commit checks](https://github.com/ByteDance-Seed/VeOmni/blob/main/CONTRIBUTING.md?plain=1#L22): `make commit && make style && make quality`
- [ ] Add / Update [the documentation](https://github.com/ByteDance-Seed/VeOmni/blob/main/docs).
- [ ] Add unit or end-to-end test(s) to [the CI workflow](https://github.com/ByteDance-Seed/VeOmni/tree/main/.github/workflows) to cover all the code. If not feasible, explain why: ...
