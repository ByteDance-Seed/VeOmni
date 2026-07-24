# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Two-call varlen Generalized Causal Attention (GCA) fast path.

The single-final-image constraint of ``single_gen_t2i_v1`` permits an exact
decomposition of GCA into two padding-free Flash-Attention calls, so the
production path never allocates a dense ``[B, 1, T, T]`` mask:

* ``prefix``       — ``FA(Q[:P], K[:P], V[:P], causal=True)``
* ``image_suffix`` — ``FA(Q[P:P+I], K[:P+I], V[:P+I], causal=False)``

``P`` (causal prefix) runs through ``<timestep>``; ``I`` (image suffix) is the
projected latent payload plus ``<eoi>``. For packed batches the compiler lays
samples out contiguously and precomputes the ``cu_seqlens`` and gather/scatter
indices consumed here (see
``sequence_compiler.compile_single_gen_t2i_packed``). This dispatcher stays
model-local and reuses the configured low-level Flash-Attention backend rather
than registering a new global attention op.
"""

# Metadata keys the packed compiler must provide for the fast path. Kept here so
# the attention forward and the collator hook validate against one source.
GCA_VARLEN_METADATA_KEYS = (
    "prefix_gather_index",
    "image_suffix_gather_index",
    "cu_seqlens_q_prefix",
    "cu_seqlens_k_prefix",
    "cu_seqlens_q_image_suffix",
    "cu_seqlens_k_full",
    "max_prefix_length",
    "max_image_suffix_length",
    "max_full_length",
    "sequence_length",
)


def gca_varlen_attention_forward(
    module,
    attention_interface,
    query_states,
    key_states,
    value_states,
    gca_metadata,
    *,
    scaling,
    dropout=0.0,
):
    """Run the two-call varlen GCA decomposition on a packed sequence.

    ``query_states`` / ``key_states`` / ``value_states`` are ``[1, heads, T, hd]``
    (``B == 1``; under Ulysses SP ``T`` is the full sequence and ``heads`` is the
    rank-local head shard, i.e. the tensors are already post all-to-all). The
    returned attention output is ``[1, T, heads, hd]`` to match the reshape the
    attention module applies before ``o_proj``.
    """
    if query_states.ndim != 4 or query_states.shape[0] != 1:
        raise ValueError("The packed GCA fast path expects query_states with shape [1, heads, T, head_dim].")

    logical_length = gca_metadata["sequence_length"]
    prefix_index = gca_metadata["prefix_gather_index"]
    image_suffix_index = gca_metadata["image_suffix_gather_index"]

    # --- causal prefix: FA(Q[:P], K[:P], V[:P], causal=True) ---
    prefix_query = query_states.index_select(2, prefix_index)
    prefix_key = key_states.index_select(2, prefix_index)
    prefix_value = value_states.index_select(2, prefix_index)
    prefix_output = attention_interface(
        module,
        prefix_query,
        prefix_key,
        prefix_value,
        attention_mask=None,
        dropout=dropout,
        scaling=scaling,
        is_causal=True,
        cu_seq_lens_q=gca_metadata["cu_seqlens_q_prefix"],
        cu_seq_lens_k=gca_metadata["cu_seqlens_k_prefix"],
        max_length_q=gca_metadata["max_prefix_length"],
        max_length_k=gca_metadata["max_prefix_length"],
    )[0]

    # --- image suffix: FA(Q[P:P+I], K[:P+I], V[:P+I], causal=False) ---
    # The suffix keys/values are the padding-free, per-sample-contiguous full
    # sequence; ``cu_seqlens_k_full`` sums only the logical lengths, so any
    # trailing sequence-parallel padding beyond ``logical_length`` is dropped and
    # never attended.
    suffix_query = query_states.index_select(2, image_suffix_index)
    full_key = key_states[:, :, :logical_length]
    full_value = value_states[:, :, :logical_length]
    image_suffix_output = attention_interface(
        module,
        suffix_query,
        full_key,
        full_value,
        attention_mask=None,
        dropout=dropout,
        scaling=scaling,
        is_causal=False,
        cu_seq_lens_q=gca_metadata["cu_seqlens_q_image_suffix"],
        cu_seq_lens_k=gca_metadata["cu_seqlens_k_full"],
        max_length_q=gca_metadata["max_image_suffix_length"],
        max_length_k=gca_metadata["max_full_length"],
    )[0]

    # Scatter both outputs back to their original packed positions. Prefix and
    # suffix indices are disjoint and cover every logical token; any padding row
    # stays zero and is discarded downstream.
    sequence_length = query_states.shape[2]
    num_heads = query_states.shape[1]
    head_dim = query_states.shape[3]
    attention_output = query_states.new_zeros((1, sequence_length, num_heads, head_dim))
    attention_output.index_copy_(1, prefix_index, prefix_output.to(attention_output.dtype))
    attention_output.index_copy_(1, image_suffix_index, image_suffix_output.to(attention_output.dtype))
    return attention_output, None


def resolve_base_attention_implementation(attn_implementation):
    """Map a VeOmni Ulysses-SP attention name to its plain Flash-Attention base.

    The GCA dispatcher performs its own all-to-all around the two calls, so it
    must invoke the backend without the wrapper's built-in single-call SP
    exchange. Non-SP names pass through unchanged.
    """
    sp_to_base = {
        "veomni_flash_attention_2_with_sp": "flash_attention_2",
        "veomni_flash_attention_3_with_sp": "flash_attention_3",
        "veomni_flash_attention_4_with_sp": "flash_attention_4",
    }
    return sp_to_base.get(attn_implementation, attn_implementation)


__all__ = [
    "GCA_VARLEN_METADATA_KEYS",
    "gca_varlen_attention_forward",
    "resolve_base_attention_implementation",
]
