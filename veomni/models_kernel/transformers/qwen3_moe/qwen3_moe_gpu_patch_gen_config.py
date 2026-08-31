# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""
Patch configuration for Qwen3Moe VeomniKernel replacements.

Regen command:
patchgen veomni.models_kernel.transformers.qwen3_moe.qwen3_moe_gpu_patch_gen_config -o veomni/models_kernel/transformers/qwen3_moe/generated --diff

Keeps the models/ VeOmni patches (router replay, fused loss, parallel plan).
Only the OpSlot guards become local VeomniKernel calls.
"""

from functools import partial
from typing import Optional

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.kernels import VeomniKernel
from veomni.models_kernel.utils.kernel_utils import empty_bias, linear_bias, resolve_kernel_impl, resolve_moe_impl
from veomni.models_kernel.utils.loss_utils import ForCausalLMLoss
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import MoeCausalLMOutputWithLogProbs
from veomni.utils.moe_router_replay import get_active_replay, maybe_replay_indices


config = PatchConfig(
    source_module="transformers.models.qwen3_moe.modeling_qwen3_moe",
    target_file="patched_modeling_qwen3_moe_gpu.py",
    description="Qwen3Moe with VeOmni patches and VeomniKernel replacements",
)

# Surface ``MoeCausalLMOutputWithLogProbs`` so the patched ``forward`` can return
# per-token log-probs / entropy as constructor fields. Mutating ``output.log_probs``
# / ``output.entropy`` after constructing ``MoeCausalLMOutputWithPast`` would
# bypass ModelOutput pytree flattening, breaking FSDP2's pre-backward unshard
# hook on ``lm_head`` and triggering ``setStorage … storage of size 0`` in
# ``chunk_logprobs.backward`` (parallels VeOmni #731's qwen3_5_moe fix).
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "MoeCausalLMOutputWithLogProbs"],
)
config.drop_import_names("MoeCausalLMOutputWithPast")
config.add_import("veomni.utils.moe_router_replay", names=["get_active_replay", "maybe_replay_indices"])
config.add_import("functools", names=["partial"])
config.add_import("veomni.kernels", names=["VeomniKernel"])
config.add_import(
    "veomni.models_kernel.utils.kernel_utils",
    names=["empty_bias", "linear_bias", "resolve_kernel_impl", "resolve_moe_impl"],
)
config.add_import(
    "veomni.models_kernel.utils.loss_utils",
    names=["ForCausalLMLoss"],
)


# ── RMSNorm (always call local VeomniKernel) ─────────────────────────────────


@config.override_method(
    "Qwen3MoeRMSNorm.__init__",
    description="Construct a local rms_norm VeomniKernel",
)
def qwen3_moe_rmsnorm_init_patched(self, hidden_size, eps: float = 1e-6) -> None:
    nn.Module.__init__(self)
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
    self.veomni_rms_norm = VeomniKernel("rms_norm", "standard", resolve_kernel_impl("rms_norm_implementation"))


@config.override_method(
    "Qwen3MoeRMSNorm.forward",
    description="Always call the local rms_norm VeomniKernel",
)
def qwen3_moe_rmsnorm_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return self.veomni_rms_norm(hidden_states, self.weight, eps=self.variance_epsilon)


# ── SwiGLU MLP (always call local VeomniKernel) ──────────────────────────────


@config.override_method(
    "Qwen3MoeMLP.__init__",
    description="Construct a local swiglu_mlp VeomniKernel",
)
def qwen3_moe_mlp_init_patched(self, config, intermediate_size=None):
    nn.Module.__init__(self)
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
    self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    self.act_fn = ACT2FN[config.hidden_act]
    self.veomni_swiglu_mlp = VeomniKernel("swiglu_mlp", "standard", resolve_kernel_impl("swiglu_mlp_implementation"))


@config.override_method(
    "Qwen3MoeMLP.forward",
    description="Always call the local swiglu_mlp VeomniKernel",
)
def qwen3_moe_mlp_forward_patched(self, x):
    return self.veomni_swiglu_mlp(
        x,
        self.gate_proj.weight,
        linear_bias(self.gate_proj),
        self.up_proj.weight,
        linear_bias(self.up_proj),
        self.down_proj.weight,
        linear_bias(self.down_proj),
    )


@config.replace_class("Qwen3MoeExperts", description="Always call moe_experts VeomniKernel on v5 gate_up_proj weights")
class PatchedQwen3MoeExperts(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = torch.nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]
        self.veomni_moe = VeomniKernel("moe_experts", "standard", resolve_moe_impl())

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        unused = empty_bias(self.gate_up_proj)
        return self.veomni_moe(
            hidden_states,
            top_k_weights,
            top_k_index,
            unused,
            unused,
            self.down_proj,
            self.gate_up_proj,
            num_experts=self.num_experts,
        )


@config.override_method(
    "Qwen3MoeTopKRouter.forward",
    description=(
        "Return raw pre-softmax logits as `router_logits` so HF's "
        "`load_balancing_loss_func` (which applies softmax internally) "
        "stays consistent with the HF aux-loss baseline."
    ),
)
def qwen3_moe_topk_router_forward_patched(self, hidden_states: torch.Tensor):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    # Return raw pre-softmax logits as `router_logits`; HF's
    # `load_balancing_loss_func` applies softmax internally. The post-softmax
    # tensor is kept locally as `routing_weights` for top-k selection only.
    router_logits = torch.nn.functional.linear(hidden_states, self.weight)
    routing_weights = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
    router_top_value, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
    # Cast ``router_top_value`` back to the raw-logits dtype, matching HF's
    # reference ``Qwen3MoeTopKRouter.forward``: transformers v5.8 keeps
    # ``router_logits`` bound to the pre-softmax ``F.linear`` output (it no
    # longer re-binds it to the fp32 post-softmax tensor), so this cast lands
    # on the model dtype rather than being a no-op. The fused MoE call site
    # casts to ``final_hidden_states.dtype`` regardless; matching HF here
    # keeps the generated modeling bitwise-equal to vanilla HF.
    router_top_value = router_top_value.to(router_logits.dtype)
    return router_logits, router_top_value, router_indices


@config.replace_function("apply_rotary_pos_emb", description="Always call rope full VeomniKernel")
def apply_rotary_pos_emb_patched(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    del position_ids
    rope = VeomniKernel("rope", "full", resolve_kernel_impl("rotary_pos_emb_implementation"))
    return rope(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)


# Dummy reference resolved at codegen time from the generated module.
rotate_half = None  # noqa: E305


@config.override_method("Qwen3MoeModel.forward", description="Support SP in Qwen3MoeModel.forward")
def qwen3_moe_model_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> MoeModelOutputWithPast:
    r"""
    cache_position (`torch.LongTensor`, *optional*):
        Indices depicting the position of the input sequence tokens in the sequence.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
    # transformers 5.9 dropped ``cache_position`` from these constructors
    # ("Deprecated and unused" — see masking_utils.py:917).
    causal_mask = mask_function(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)

    return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


@config.override_method(
    "Qwen3MoeForCausalLM.__init__",
    description="Bind ForCausalLMLoss and load_balancing_loss VeomniKernels",
)
def qwen3_moe_forcausallm_init_patched(self, config):
    super().__init__(config)
    self.model = Qwen3MoeModel(config)
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.router_aux_loss_coef = config.router_aux_loss_coef
    self.num_experts = config.num_experts
    self.num_experts_per_tok = config.num_experts_per_tok
    impl = resolve_kernel_impl("cross_entropy_loss_implementation", npu_as="chunk_loss")
    self.veomni_ce = VeomniKernel("cross_entropy_loss", "standard", impl)
    self.loss_function = partial(ForCausalLMLoss, kernel=self.veomni_ce)
    self.veomni_lb = VeomniKernel(
        "load_balancing_loss",
        "standard",
        resolve_kernel_impl("load_balancing_loss_implementation"),
    )
    self.post_init()


@config.override_method(
    "Qwen3MoeForCausalLM.forward",
    description="Always call ForCausalLMLoss and load_balancing_loss VeomniKernels",
)
def qwen3_moe_forcausallm_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    output_router_logits: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> MoeCausalLMOutputWithLogProbs:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    cache_position (`torch.LongTensor`, *optional*):
        Indices depicting the position of the input sequence tokens in the sequence.
    """
    output_router_logits = (
        output_router_logits if output_router_logits is not None else self.config.output_router_logits
    )

    outputs: MoeModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_router_logits=output_router_logits,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        loss, logits, fused_linear_aux = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.vocab_size,
            hidden_states=hidden_states,
            weights=self.lm_head.weight,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states)

    aux_loss = None
    if output_router_logits:
        router_logits = outputs.router_logits
        if router_logits is None or not isinstance(router_logits, tuple):
            aux_loss = 0
        else:
            gate = torch.cat([layer.reshape(-1, layer.shape[-1]) for layer in router_logits], dim=0)
            mask = attention_mask if isinstance(attention_mask, torch.Tensor) else gate.new_empty(0)
            aux_loss = self.veomni_lb(gate, mask, top_k=self.num_experts_per_tok)
        if labels is not None and isinstance(aux_loss, torch.Tensor):
            loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

    return MoeCausalLMOutputWithLogProbs(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
        fused_linear_aux=fused_linear_aux,
    )


@config.override_method(
    "Qwen3MoeForCausalLM.get_parallel_plan",
    description="Register Qwen3Moe expert parallel plan for v5 generated modeling",
)
def qwen3_moe_get_parallel_plan_patched(self):
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan()


@config.override_method(
    "Qwen3MoeSparseMoeBlock.forward",
    description="Call maybe_replay_indices so RL frameworks can record / replay MoE routing decisions",
)
def qwen3_moe_sparse_moe_block_forward_patched(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
    router_logits, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
    # MoE router replay: when an RL framework has installed a manager via
    # ``set_active_replay``, the manager may substitute ``selected_experts``
    # with previously recorded target indices. The manager's sole
    # responsibility is choosing indices; all model-specific post-topk
    # weight math (softmax recompute, gather, renorm, dtype cast) is
    # replicated here so the cross-framework controller stays
    # model-agnostic. After upstream #715, ``Qwen3MoeTopKRouter`` returns
    # pre-softmax ``router_logits`` for HF's load_balancing_loss_func
    # consistency and discards its internal post-softmax matrix after
    # top-k, so we recompute ``softmax`` here to feed the RR contract.
    # The guarded block runs only when a manager is installed — the
    # default training path pays zero extra compute.
    if get_active_replay() is not None:
        target_dtype = routing_weights.dtype
        routing_scores = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        selected_experts = maybe_replay_indices(self.gate, routing_scores, selected_experts)
        routing_weights = routing_scores.gather(1, selected_experts)
        if self.gate.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(-1, keepdim=True)
        routing_weights = routing_weights.to(target_dtype)
    final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
    return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
