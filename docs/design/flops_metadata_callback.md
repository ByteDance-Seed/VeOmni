# FLOPs metadata callbacks

`EnvironMeter` uses callbacks to extract FLOPs lengths for DiT models. A
model-specific callback takes priority; other DiT models use the default DiT
callback. HF models retain their existing metadata handling without a callback.

## Callback contract

Register a function under the config's `model_type` in
[`FLOPS_METADATA_CALLBACKS`](../../veomni/utils/flops_meter.py#L82). The meter
selects it during [`EnvironMeter.__init__`](../../veomni/utils/helper.py#L211)
and calls it from [`add`](../../veomni/utils/helper.py#L276), once per input
micro-batch. Callbacks receive the same input supplied to `add`: a batch
dictionary or a list of batch dictionaries, depending on the training path.
The default DiT callback preserves the existing dictionary-based latent path.

[`get_flops_metadata_callback`](../../veomni/utils/flops_meter.py#L88) returns
`None` for configs without `condition_model_type`. For DiT configs it looks up
the model type, falling back to the registered `__default_dit__` callback,
which uses the existing [`_compute_wan_seqlens`](../../veomni/utils/flops_meter.py#L23)
latent shape calculation.

The callback returns `(batch_seqlens, model_metadata)`:

- `batch_seqlens`: a list of valid main sequence lengths, one per packed sample.
- `model_metadata`: a dictionary mapping model-specific field names to lists of
  per-sample lengths. The meter extends each list across micro-batches and
  passes the dictionary as `model_metadata` to the configured FLOPs counter.

The callback is stateless. The meter accumulates lengths, and
[`step`](../../veomni/utils/helper.py#L358) estimates FLOPs and sums achieved
FLOPS, tokens and samples across the data-parallel group, then resets the
per-step state. Inputs for which the callback returns no lengths contribute no
FLOPs. Missing metadata on some inputs or ranks can therefore make the reported
MFU an underestimate. All ranks in the group must use the same callback.

`VeomniFlopsCounter.estimate_flops` forwards `model_metadata` to the estimator
selected by `model_type`; that estimator owns interpretation and validation of
its fields. Registering a callback alone does not add an estimator. Internal
alternative counters must also accept `model_metadata` before a callback that
returns extra fields can be used with them. The meter omits `model_metadata`
when the accumulated dictionary is empty. Estimators that consume this argument
must therefore provide a default (for example, `model_metadata=None`) and
handle missing metadata according to their model's contract. Existing estimator
calls without metadata retain their previous signature.

## Example

```python
def extract_example_metadata(batch):
    # The example model's estimator reads `text_lengths` from model_metadata.
    samples = batch if isinstance(batch, list) else [batch]
    lengths, text_lengths = [], []
    for sample in samples:
        lengths.extend(sample["lengths"])
        text_lengths.extend(sample["text_lengths"])
    return lengths, {"text_lengths": text_lengths}


FLOPS_METADATA_CALLBACKS["ExampleModel"] = extract_example_metadata
```

The corresponding estimator must be added to `VeomniFlopsCounter.estimate_func`.
Its signature accepts `(tokens_sum, batch_seqlens, delta_time, model_metadata=None)`;
when metadata is available it reads `model_metadata["text_lengths"]` and returns
estimated TFLOPS. It also handles `None` for a step with no metadata. The counter
supplies device peak FLOPS separately. See the integration test below for a
complete callback and estimator example.

For MiniMax-H3, [`_extract_minimax_h3_metadata`](../../veomni/utils/flops_meter.py#L48)
extracts main and text-refiner lengths from offline `packed` data. The existing
[`VeomniFlopsCounter`](../../veomni/utils/count_flops.py#L73) calculates the
formula. H3 online training may create `packed` after the meter runs; if no
rank has usable lengths, the reported MFU is zero.

The callback path and H3 data handling are covered by
[`tests/utils/test_count_flops.py`](../../tests/utils/test_count_flops.py#L263).
