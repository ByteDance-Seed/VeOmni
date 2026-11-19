# Custom DPO tokenizer and template contracts

DPO segment accounting needs the physical chosen and rejected lengths before
packing. An all-one attention mask cannot prove that a producer did not insert
padding. The producer's owner must audit and declare that its single-example
output is unpadded, with stable tokenization, labels and truncation semantics.
The declaration is an application promise, not an automatic proof of behavior.

For a custom tokenizer or chat template, declare
`dpo_unpadded_output_contract = True` on the **concrete class** and provide a
nonempty stable `dpo_unpadded_output_contract_id` string or zero-argument method.
A subclass must declare the flag itself. Include all configuration that changes
tokenization, special tokens, templates, truncation or supervision in the ID,
and change it whenever those semantics change. Recreate the same producer and
ID on every rank and after restart. Do not replace its methods on an instance.

For example, this template retains the built-in `DefaultTemplate` encoding and
makes an application-owned revision explicit:

```python
from veomni.data.chat_template import DefaultTemplate


class AuditedProjectTemplate(DefaultTemplate):
    dpo_unpadded_output_contract = True

    def __init__(self, tokenizer, revision):
        super().__init__(tokenizer)
        self.revision = revision

    def dpo_unpadded_output_contract_id(self):
        return f"example.default-template.{self.revision}"


chat_template = AuditedProjectTemplate(tokenizer, revision="v1")
```

This example does not audit the supplied tokenizer. Its owner must separately
satisfy the tokenizer contract, or audit an eligible built-in fast tokenizer and
provide `dpo_config.audited_unpadded_tokenizer_sha256` as described in the
[DPO guide](../examples/qwen3_dpo.md#experimental-explicit-segment-provenance).
The template's `encode_messages(messages, max_seq_len=...)` must return flat,
unpadded `input_ids`, `attention_mask`, and `labels`; labels must mask the
shared prompt and non-assistant messages. Chosen and rejected conversations
must share their prompt, and truncation must retain compatible context and
scoreable completion tokens. Plain-text producers must also provide a
verifiable prompt boundary. Already padded data needs a separately validated
unpadding transform before this interface.

[`dpo_declared_producer_identity`](../../veomni/utils/dpo_provenance.py#L160)
checks the concrete declaration and binds its ID, class and live methods.
[`dpo_producer_identity`](../../veomni/data/data_transform.py#L286) checks the
tokenizer and template together, including a template-owned tokenizer. The
standard DPO transform reuses the attestation within each process, and checks
again on a worker's first sample after fork or spawn. Producers, their methods,
configuration, and the template-owned tokenizer must remain immutable for the
process lifetime. Call `dpo_producer_identity(..., force_refresh=True)` to
explicitly check the current live objects. The
[trainer initialization](../../veomni/trainer/text_dpo_trainer.py#L768) captures
the identity; its [checkpoint state](../../veomni/trainer/text_dpo_trainer.py#L808)
and [restore validation](../../veomni/trainer/text_dpo_trainer.py#L874) bind it
alongside objective, data and parallel geometry. Changed or missing provenance
is rejected for exact trainer-state recovery; use weights-only initialization
or an independently verified migration when the old identity cannot be proven.

The identity detects changes to the declared ID and observed methods; it does
not inspect every external dependency or prove that the producer's declaration
is true. The application owns the audit and versioning of external state.
