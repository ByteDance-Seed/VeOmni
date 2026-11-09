# FLOPs metadata callbacks

`EnvironMeter` uses callbacks to extract FLOPs lengths for DiT models. A
model-specific callback takes priority; other DiT models use the default DiT
callback. HF models retain their existing metadata handling without a callback.

## Callback contract

Register a function under the config's `model_type` in
[`FLOPS_METADATA_CALLBACKS`](../../veomni/utils/flops_meter.py#L73). The meter
selects it during [`EnvironMeter.__init__`](../../veomni/utils/helper.py#L228)
and calls it from [`add`](../../veomni/utils/helper.py#L269), once per input
micro-batch. Callbacks receive the same input supplied to `add`: a batch
dictionary or a list of batch dictionaries, depending on the training path.
The default DiT callback preserves the existing dictionary-based latent path.

[`get_flops_metadata_callback`](../../veomni/utils/flops_meter.py#L79) returns
`None` for configs without `condition_model_type`. For DiT configs it looks up
the model type, falling back to the registered `__default_dit__` callback,
which uses the existing [`_compute_wan_seqlens`](../../veomni/utils/flops_meter.py#L18)
latent shape calculation.

The callback returns `(batch_seqlens, extra_flops_kwargs)`:

- `batch_seqlens`: a list of valid main sequence lengths, one per packed sample.
- `extra_flops_kwargs`: a dictionary mapping estimator argument names to lists of
  per-sample lengths. The meter extends each list across micro-batches and
  passes the dictionary to the configured FLOPs counter.

The callback is stateless. The meter accumulates lengths, and
[`step`](../../veomni/utils/helper.py#L351) estimates FLOPs and sums achieved
FLOPS, tokens and samples across the data-parallel group, then resets the
per-step state. Inputs for which the callback returns no lengths contribute no
FLOPs. Missing metadata on some inputs or ranks can therefore make the reported
MFU an underestimate. All ranks in the group must use the same callback.

The configured counter must explicitly accept the names supplied in
`extra_flops_kwargs`. Registering a callback alone does not add an estimator or
extend a counter's signature. In particular, internal alternative counters
need to support the same arguments before a callback can be used with them.

## Example

```python
def extract_example_metadata(batch):
    # The example counter must accept `text_lengths`.
    samples = batch if isinstance(batch, list) else [batch]
    lengths, text_lengths = [], []
    for sample in samples:
        lengths.extend(sample["lengths"])
        text_lengths.extend(sample["text_lengths"])
    return lengths, {"text_lengths": text_lengths}


FLOPS_METADATA_CALLBACKS["ExampleModel"] = extract_example_metadata
```

For MiniMax-H3, [`_extract_minimax_h3_metadata`](../../veomni/utils/flops_meter.py#L43)
extracts main and text-refiner lengths from offline `packed` data. The existing
[`VeomniFlopsCounter`](../../veomni/utils/count_flops.py#L73) calculates the
formula. H3 online training may create `packed` after the meter runs; if no
rank has usable lengths, the reported MFU is zero.

The callback path and H3 data handling are covered by
[`tests/utils/test_count_flops.py`](../../tests/utils/test_count_flops.py#L257).
