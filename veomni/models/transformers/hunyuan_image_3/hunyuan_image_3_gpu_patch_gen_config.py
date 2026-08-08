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

"""HunyuanImage 3 GPU patchgen configuration.

Adds the model-local ``single_gen_t2i_v1`` packed-varlen forward: two-call
generalized causal attention, generalized 2D RoPE, Ulysses sequence
parallelism, and the flow objective. The dense edge mask the two calls encode
is available as a correctness oracle via ``dense_reference_attention`` on the
compiled metadata (tests only).

Regen command:
patchgen veomni.models.transformers.hunyuan_image_3.hunyuan_image_3_gpu_patch_gen_config \
  -o veomni/models/transformers/hunyuan_image_3/generated --diff
"""

from patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.hunyuan_v1_moe.modeling_hunyuan_v1_moe",
    target_file="patched_modeling_hunyuan_image_3_gpu.py",
    description="HunyuanImage 3 official-layout import with the single_gen_t2i_v1 packed varlen forward",
    transformers_version="5.9.0",
)
config.add_import(
    "veomni.models.transformers.hunyuan_image_3.component_policy",
    names=["HunyuanImage3ComponentPolicy"],
)
config.add_import(
    "veomni.models.transformers.hunyuan_image_3.configuration_hunyuan_image_3",
    names=["HunyuanImage3Config"],
)
config.add_import(
    "veomni.models.transformers.hunyuan_image_3.parallel_plan",
    names=["get_parallel_plan"],
)
config.add_import(
    "veomni.schedulers.flow_matching_loss",
    names=["derive_flow_seed", "flow_matching_loss", "prepare_reference_flow_batch"],
)
config.add_import(
    "veomni.utils.model_outputs",
    names=["HunyuanImage3ReferenceOutput"],
)
config.add_import(
    "veomni.models.transformers.hunyuan_image_3.image_projection",
    names=["TimestepEmbedder", "UNetDown", "UNetUp"],
)
config.add_import(
    "veomni.models.transformers.hunyuan_image_3.rope_2d",
    names=["build_2d_rope"],
)
config.add_import(
    "veomni.models.transformers.hunyuan_image_3.generalized_causal_attention",
    names=[
        "GCA_VARLEN_METADATA_KEYS",
        "GCA_VARLEN_TENSOR_KEYS",
        "build_packed_gca_dense_mask",
        "gca_varlen_attention_forward",
        "resolve_base_attention_implementation",
    ],
)
config.add_import(
    "veomni.distributed.sequence_parallel",
    names=[
        "gather_heads_scatter_seq",
        "gather_outputs",
        "gather_seq_scatter_heads",
        "get_ulysses_sequence_parallel_group",
        "slice_input_tensor",
    ],
)
config.add_import(
    "veomni.models.transformers.hunyuan_image_3.vae",
    names=["HunyuanImage3VAE"],
)
config.add_import("functools", names=["partial"])
config.add_import(
    "veomni.distributed.parallel_state",
    names=["get_parallel_state"],
)
config.add_import(
    "veomni.models.transformers.hunyuan_image_3.sequence_layout",
    names=["collate_hunyuan_image_3_metadata", "get_hunyuan_image_3_extra_collate_infos"],
)

# The eager expert loop is replaced below with an OpSlot-guarded forward, so the
# HuggingFace ``@use_experts_implementation`` decorator is dropped and its import
# would be unused.
config.drop_import_names("use_experts_implementation")

# OpSlot for the fused / expert-parallel MoE experts path, mirroring the
# qwen3_moe patch. Bound at model-build time by ``_bind_veomni_ops()`` in
# ``veomni/models/auto.py``; it stays unbound (eager fallback) unless
# ``moe_implementation`` selects a fused kernel. Expert parallelism is taken
# inside the fused kernel when the parallel state reports ``ep_enabled``.
config.add_post_import_block(
    """
    from veomni.ops.dispatch import OpSlot

    veomni_moe_experts_forward = OpSlot("moe_experts", "standard")
    """
)


# Lightweight definitions let patchgen inspect the replacement source without
# importing PyTorch. The generated file resolves ``nn`` and the pretrained base
# class from the Transformers 5.9 source module.
class _Module:
    pass


class _NN:
    Module = _Module


nn = _NN()


# ================================================================
# Patch: HunYuanMoEV1Attention
# 1. Preserve official group-interleaved fused QKV checkpoint layout.
# 2. Dispatch packed GCA metadata to the two-call varlen fast path (or, for the
#    correctness oracle, to the dense edge mask it encodes).
# ================================================================
@config.replace_class(
    "HunYuanMoEV1Attention",
    description="Preserve the official group-interleaved fused QKV parameter layout",
)
class PatchedHunYuanMoEV1Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_key_value_heads * self.head_dim
        self.qkv_proj = nn.Linear(config.hidden_size, q_width + 2 * kv_width, bias=config.attention_bias)
        self.o_proj = nn.Linear(q_width, config.hidden_size, bias=config.attention_bias)
        self.query_layernorm = HunYuanMoEV1RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = HunYuanMoEV1RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        # --- Patch.1 ---
        input_shape = hidden_states.shape[:-1]
        qkv_states = self.qkv_proj(hidden_states).reshape(
            *input_shape,
            self.num_key_value_heads,
            self.num_key_value_groups + 2,
            self.head_dim,
        )
        query_states, key_states, value_states = torch.split(
            qkv_states,
            [self.num_key_value_groups, 1, 1],
            dim=-2,
        )
        query_states = query_states.reshape(*input_shape, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.squeeze(-2).transpose(1, 2)
        value_states = value_states.squeeze(-2).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)
        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)
        # --- Patch.1 ---

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # --- Patch.2 ---
        gca_metadata = kwargs.pop("hy3_gca_metadata", None)
        if gca_metadata is not None:
            if past_key_values is not None:
                raise ValueError("single_gen_t2i_v1 GCA does not support KV cache.")
            dense_mask = gca_metadata.get("dense_attention_mask")
        else:
            dense_mask = None
        if dense_mask is not None:
            # Correctness oracle: run PyTorch SDPA against the dense edge mask
            # the two-call decomposition below encodes, built from the same
            # packed metadata. Test-only (O(T^2)); selected by
            # ``dense_reference_attention`` on the compiled metadata, never by
            # the collator. ``enable_gqa=True`` (torch>=2.5) expands the K/V
            # head groups internally so we do not materialise a repeated tensor.
            edge_mask = dense_mask if dense_mask.dim() == 4 else dense_mask.unsqueeze(1)
            attention_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=edge_mask,
                scale=self.scaling,
                enable_gqa=True,
            )
            attention_output = attention_output.transpose(1, 2).contiguous()
            attention_weights = None
        elif gca_metadata is not None:
            # Production fast path: two-call varlen GCA. When Ulysses SP is active
            # each rank arrives with a sequence shard and the full head set; the
            # all-to-all gathers the sequence and scatters heads so the two calls
            # run on the full packed sequence with a head shard, then the inverse
            # all-to-all restores the sequence shard. Both collapse to a no-op when
            # SP is disabled.
            #
            # Perf note: the packed layout is B==1 (asserted by _validate_packed_metadata),
            # so we SQUEEZE the batch dim before every A2A to keep scatter_dim/gather_dim
            # <= 1. That hits the fast contiguous-buffer ``dist.all_to_all_single`` path in
            # ``ulysses.all_to_all_tensor``; a 4D tensor with scatter_dim=2 falls into the
            # slow list-based ``dist.all_to_all`` (tensor_split + per-shard .contiguous() +
            # cat). Numerically identical either way.
            base_implementation = resolve_base_attention_implementation(self.config._attn_implementation)
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                base_implementation,
                eager_attention_forward,
            )
            # [1, H, T_shard, D] -> [T_shard, H, D] (batch known to be 1 for packed)
            query_seq_first = query_states.transpose(1, 2).squeeze(0)
            key_seq_first = key_states.transpose(1, 2).squeeze(0)
            value_seq_first = value_states.transpose(1, 2).squeeze(0)
            query_full = gather_seq_scatter_heads(query_seq_first, seq_dim=0, head_dim=1)
            key_full = gather_seq_scatter_heads(key_seq_first, seq_dim=0, head_dim=1)
            value_full = gather_seq_scatter_heads(value_seq_first, seq_dim=0, head_dim=1)
            # Restore [1, H_shard, T_full, D] for the GCA index_select on dim 2.
            attention_output, attention_weights = gca_varlen_attention_forward(
                self,
                attention_interface,
                query_full.unsqueeze(0).transpose(1, 2),
                key_full.unsqueeze(0).transpose(1, 2),
                value_full.unsqueeze(0).transpose(1, 2),
                gca_metadata,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
            )
            # GCA returns [1, T_full, H_shard, D]; squeeze for the fast inverse A2A,
            # then unsqueeze back.
            attention_output = gather_heads_scatter_seq(attention_output.squeeze(0), head_dim=1, seq_dim=0).unsqueeze(
                0
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation,
                eager_attention_forward,
            )
            attention_output, attention_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
        # --- Patch.2 ---
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attention_output), attention_weights


@config.replace_class(
    "HunYuanMoEV1MLP",
    description="Keep the official shared-MLP gate_and_up_proj key and [up, gate] half order",
)
class PatchedHunYuanMoEV1MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_and_up_proj = nn.Linear(
            self.hidden_size,
            2 * self.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        up_states, gate_states = self.gate_and_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(up_states * self.act_fn(gate_states))


# ================================================================
# Patch: HunYuanMoEV1Experts
# Drop the HuggingFace eager-only decorator and route through VeOmni's fused /
# expert-parallel MoE OpSlot when a fused kernel is bound. The 3D weight layout
# ([E, 2*moe_inter, hidden] / [E, hidden, moe_inter], gate-first) already matches
# the fused adapter and the checkpoint tensor converter, and parallel_plan.py
# shards both parameters on dim 0 for EP, so no relayout is needed.
# ================================================================
@config.replace_class(
    "HunYuanMoEV1Experts",
    description="Route MoE experts through the VeOmni fused/EP OpSlot with an eager fallback",
)
class PatchedHunYuanMoEV1Experts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        # ``num_experts`` is the GLOBAL expert count; the EP path divides it by the
        # EP size in ``preprocess``/``token_pre_all2all``.
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states,
        top_k_index,
        top_k_weights,
    ):
        final_hidden_states = torch.zeros_like(hidden_states)
        if veomni_moe_experts_forward.use_non_eager_impl:
            return veomni_moe_experts_forward(self, hidden_states, top_k_index, top_k_weights)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class HunYuanMoEV1PreTrainedModel:
    pass


class HunyuanImage3Config:
    pass


# ================================================================
# Patch: HunyuanImage3ForCausalMM
# 1. Construct only components selected by the model-local lifecycle policy.
# 2. Add the packed-varlen single_gen_t2i_v1 flow forward.
# ================================================================
@config.add_helper_after("HunYuanMoEV1ForCausalLM")
class HunyuanImage3ForCausalMM(HunYuanMoEV1PreTrainedModel):
    config_class = HunyuanImage3Config
    _no_split_modules = ["HunYuanMoEV1DecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        # --- Patch.1 ---
        self.component_policy = HunyuanImage3ComponentPolicy.from_dict(config.component_policy)
        self.model = HunYuanMoEV1Model(config)

        if self.component_policy.image_projector != "absent":
            self.patch_embed = UNetDown(
                patch_size=config.patch_size,
                in_channels=config.vae["latent_channels"],
                emb_channels=config.hidden_size,
                hidden_channels=config.patch_embed_hidden_dim,
                out_channels=config.hidden_size,
            )
        if self.component_policy.timestep_modules != "absent":
            self.timestep_emb = TimestepEmbedder(hidden_size=config.hidden_size)
            self.time_embed = TimestepEmbedder(hidden_size=config.hidden_size)
            self.time_embed_2 = TimestepEmbedder(hidden_size=config.hidden_size)
        if self.component_policy.image_head != "absent":
            self.final_layer = UNetUp(
                patch_size=config.patch_size,
                emb_channels=config.hidden_size,
                in_channels=config.hidden_size,
                hidden_channels=config.patch_embed_hidden_dim,
                out_channels=config.vae["latent_channels"],
                out_norm=True,
            )
        if self.component_policy.vae_encoder != "absent":
            self.vae = HunyuanImage3VAE(config.vae)

        self.pad_id = config.pad_id
        self.vocab_size = config.vocab_size
        # Flow-matching RNG state (posterior noise / timestep / diffusion noise
        # -- see veomni.schedulers.flow_matching_loss). Owned by the model so DCP model
        # load restores it via get/set_extra_state -- no trainer-side callback, no
        # per-batch identity plumbing. Lazily initialized on first forward call so
        # the (DP-rank-aware) seed and the (CUDA-resident) device are both known.
        self._flow_generator = None
        self.post_init()
        self.apply_component_policy()
        # --- Patch.1 ---

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return None

    def get_parallel_plan(self):
        return get_parallel_plan()

    def apply_pre_fsdp_dtype_policy(self) -> None:
        # MUTATES model dtype in place. ``parallelize_model_fsdp2`` calls this once,
        # unconditionally, right before the root ``fully_shard`` -- while params are
        # still on meta -- so the disk FP32 VAE weights load without a BF16 round-trip.
        # The set of ignored params is declared separately (and authoritatively) by
        # ``ParallelPlan.fsdp_ignored_param_fqn_patterns=["vae.*"]``; this hook only
        # owns the dtype cast that must precede the root shard.
        # Why FP32: BF16 perturbs the online latents by 4-6% (rel_max on latent-mean),
        # which drifts image quality. No-op when the component policy dropped the VAE.
        if hasattr(self, "vae"):
            self.vae.float()

    def get_position_id_func(self):
        # The single_gen_t2i_v1 transform emits no position_ids; the packed
        # compiler owns the 2D coordinates. Present so the VLM data-transform
        # builder (which calls this unconditionally) does not AttributeError.
        return None

    def get_extra_collate_infos(self):
        return get_hunyuan_image_3_extra_collate_infos()

    def get_metadata_collate_func(self):
        # Picklable hook (partial over a module-level fn) that finalizes the packed
        # hy3_sequence_metadata + component_inputs after the collator's pack/SP
        # stages. sp_size is bound here (main process) so the compiled padded length
        # matches the collator's SP-padded input_ids.
        return partial(collate_hunyuan_image_3_metadata, sp_size=get_parallel_state().sp_size)

    def apply_component_policy(self):
        self.model.layers.requires_grad_(self.component_policy.transformer == "trainable")
        # The official image path consumes raw decoder states and does not apply
        # the text final norm. Keep it loadable but frozen until a text-output
        # capability gives it a forward role.
        self.model.norm.requires_grad_(False)
        self.model.embed_tokens.requires_grad_(self.component_policy.text_embedding == "trainable")
        for component_name, policy_name in (
            ("patch_embed", "image_projector"),
            ("timestep_emb", "timestep_modules"),
            ("time_embed", "timestep_modules"),
            ("time_embed_2", "timestep_modules"),
            ("final_layer", "image_head"),
        ):
            component = getattr(self, component_name, None)
            if component is not None:
                component.requires_grad_(self.component_policy.state(policy_name) == "trainable")
        if hasattr(self, "vae"):
            self.vae.requires_grad_(False)
            # Frozen encoder runs FP32. On the FSDP path apply_pre_fsdp_dtype_policy
            # runs before root shard; cast here too so non-FSDP paths are FP32 as well.
            self.vae.float()
            self.vae.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        # Also the post-parallelize reapply point: FSDP2 keeps this override (it
        # inserts FSDPModule ahead of the model class in the MRO and defines no
        # ``train``), and the trainer calls ``model.train()`` right after wrapping,
        # so a policy reset by meta-init/``to_empty`` is restored before the
        # optimizer reads ``requires_grad``.
        self.apply_component_policy()
        return self

    def _ensure_flow_generator(self, device):
        # Lazy init: first forward call on this rank creates one CUDA generator
        # seeded from ``(config.flow["seed"], dp_rank)``. Two invariants underpin
        # correctness:
        #   1. Same DP replica (varying SP/EP rank) → same seed → same draw. Flow
        #      matching requires bit-identical noised_latents/flow_target across
        #      SP-shard peers or the gradient stops being flow-matching loss.
        #   2. Different DP replica → different seed → different draw, so DP
        #      variance averaging is preserved.
        # The base seed rides on the model config -- it is part of the flow recipe,
        # so it lands in the checkpoint's config.json and a re-run reproduces the
        # noise without depending on the launcher. Mirrors how
        # ``dit_trainer._build_condition_model`` passes its noise seed through the
        # config. Only ``dp_rank`` -- genuine topology -- comes from ParallelState.
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if self._flow_generator is not None:
            stored = self._flow_generator.device
            # ``torch.Generator(device="cuda")`` normalizes to ``device('cuda', 0)``
            # on most builds but not all; treat missing indices on either side as
            # matching the current CUDA device rather than tripping the guard.
            same = stored == device or (
                stored.type == device.type
                and stored.type == "cuda"
                and (stored.index is None or device.index is None or stored.index == device.index)
            )
            if not same:
                raise RuntimeError(f"Flow generator was initialized on {stored} but forward is running on {device}.")
            return self._flow_generator
        seed = derive_flow_seed(self.config.flow["seed"], get_parallel_state().dp_rank)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        self._flow_generator = generator
        return generator

    def get_extra_state(self):
        # Serialize the flow generator so DCP model load resumes the noise stream
        # where it stopped instead of replaying it from the top -- under frequent
        # preemption a restart-from-seed would over-sample the same sigma prefix.
        # Uninitialized generators (checkpointing before the first forward)
        # serialize to an empty payload and re-seed lazily on restore.
        if self._flow_generator is None:
            return {"flow_generator": None}
        return {
            "flow_generator": {
                "device_type": self._flow_generator.device.type,
                "state": self._flow_generator.get_state(),
            }
        }

    def set_extra_state(self, state):
        if not isinstance(state, dict):
            raise TypeError("HunyuanImage3ForCausalMM extra_state must be a dict.")
        payload = state.get("flow_generator")
        if payload is None:
            self._flow_generator = None
            return
        if not isinstance(payload, dict):
            raise TypeError("flow_generator extra_state entry must be a dict or None.")
        device = torch.device(payload["device_type"])
        generator = torch.Generator(device=device)
        generator.set_state(payload["state"])
        self._flow_generator = generator

    # --- Patch.2 ---
    def forward(
        self,
        input_ids=None,
        component_inputs=None,
        hy3_sequence_metadata=None,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        # The trainer calls ``model(**micro_batch, use_cache=False)``; ``**kwargs``
        # absorbs collated training-batch keys the flow forward does not consume
        # (labels, attention_mask, dummy position_ids, FA cu_seqlens, ...).
        if use_cache:
            raise ValueError("single_gen_t2i_v1 forward requires use_cache=False.")
        self._validate_reference_components()
        if not isinstance(hy3_sequence_metadata, dict) or hy3_sequence_metadata.get("layout") != "packed_varlen":
            raise ValueError("single_gen_t2i_v1 forward requires the packed_varlen compiled layout.")
        return self._forward_packed_varlen(
            input_ids=input_ids,
            component_inputs=component_inputs,
            hy3_sequence_metadata=hy3_sequence_metadata,
            return_dict=return_dict,
        )

    def _forward_packed_varlen(
        self,
        *,
        input_ids,
        component_inputs,
        hy3_sequence_metadata,
        return_dict,
    ):
        # Packed production path: two-call varlen GCA + optional Ulysses SP,
        # validated against the dense oracle. Samples are laid out contiguously in
        # one [1, T] row; the fixed-resolution image processor gives every packed
        # sample the same grid, so posteriors batch over num_samples.
        metadata = self._validate_packed_metadata(input_ids, hy3_sequence_metadata)
        num_samples = metadata["num_samples"]
        if metadata["sequence_length"] > self.config.max_position_embeddings:
            raise ValueError("Packed sequence length exceeds max_position_embeddings.")

        posterior_mean, posterior_logvar = self._get_latent_posterior(component_inputs, input_ids)
        if posterior_mean.shape[0] != num_samples:
            raise ValueError("Packed latent posterior must carry one entry per packed sample.")
        # Keep this draw here, at the top of forward and outside every
        # activation-checkpointing boundary. VeOmni applies AC per decoder layer,
        # so the backward recompute never re-enters this line; widening AC to wrap
        # the whole model would make recompute draw fresh noise against the
        # forward's targets and silently corrupt the gradient.
        flow_batch = prepare_reference_flow_batch(
            posterior_mean,
            posterior_logvar,
            vae_config=self.config.vae,
            flow_config=self.config.flow,
            generator=self._ensure_flow_generator(posterior_mean.device),
        )
        grid_height, grid_width = self._validate_reference_grid(metadata["grid_hw"], flow_batch["noised_latents"])

        timesteps = flow_batch["timesteps"]
        image_sequence, token_height, token_width = self.patch_embed(
            flow_batch["noised_latents"],
            self.time_embed(timesteps),
        )
        if (token_height, token_width) != (grid_height, grid_width):
            raise ValueError("Compiled image grid does not match the patch projection output.")

        hidden_size = self.config.hidden_size
        hidden_states = self.model.embed_tokens(input_ids)
        image_payload_indices = metadata["image_payload_indices"]
        image_tokens = image_sequence.reshape(1, num_samples * image_sequence.shape[1], hidden_size)
        if image_payload_indices.shape[1] != image_tokens.shape[1]:
            raise ValueError("Compiled image payload indices do not match the packed patch projection tokens.")
        hidden_states = hidden_states.scatter(
            1,
            image_payload_indices.unsqueeze(-1).expand(-1, -1, hidden_size),
            image_tokens,
        )
        timestep_embeddings = self.timestep_emb(timesteps)
        hidden_states = hidden_states.scatter(
            1,
            metadata["timestep_positions"].view(1, num_samples, 1).expand(1, num_samples, hidden_size),
            timestep_embeddings.unsqueeze(0),
        )

        rope_scaling = self.config.image_rope_scaling or {}
        if rope_scaling.get("type", "custom") != "custom":
            raise ValueError("single_gen_t2i_v1 packed forward requires custom 2D RoPE.")
        cos, sin = build_2d_rope(
            metadata["position_ids"],
            head_dim=self.config.head_dim,
            rope_theta=self.config.rope_parameters["rope_theta"],
            base_rescale_factor=float(rope_scaling.get("factor", 1.0)),
            dtype=hidden_states.dtype,
        )

        # Ulysses SP: everything above this line runs replicated on every rank;
        # slice the residual stream here so each decoder layer's attention
        # all-to-all works on a sequence shard. One Ulysses group drives both the
        # slice and the A2A, and both are no-ops when SP is disabled.
        #
        # Why the slice is model-side and not collator-side (``sp_slice=True``):
        # the single-stream packed layout addresses text/image/timestep tokens by
        # absolute full-sequence coordinates (``image_payload_indices`` /
        # ``timestep_positions``), and the 2D-conv ``final_layer`` needs the whole
        # image grid. Pre-slicing would force per-rank index remapping in every
        # scatter plus a gather before the head anyway — no compute saved. Hence
        # ``sp_slice=False`` in the extra collate infos (sequence_layout.py); the
        # collator only pads to a multiple of sp_size.
        #
        # The replicated prologue is cheap. The measured SP tax is NCCL launch
        # overhead from the ~256 A2A/step (docs/design/hunyuan_image_3_sp_toy_perf.md),
        # which ``async_ulysses_dit`` addresses — moving the slice would not.
        sp_group = get_ulysses_sequence_parallel_group()
        hidden_states = slice_input_tensor(hidden_states, dim=1, group=sp_group)
        cos = slice_input_tensor(cos, dim=1, group=sp_group)
        sin = slice_input_tensor(sin, dim=1, group=sp_group)
        gca_metadata = {key: metadata[key] for key in GCA_VARLEN_METADATA_KEYS}
        if metadata.get("dense_reference_attention"):
            # Correctness-oracle mode (tests): swap the two-call decomposition for
            # the dense edge mask it encodes, built from this same metadata. The
            # mask is indexed on logical positions, so it cannot describe an
            # SP-padded tail or a sequence shard.
            if sp_group is not None:
                raise ValueError("The dense GCA oracle does not support Ulysses sequence parallelism.")
            if metadata["padded_sequence_length"] != metadata["sequence_length"]:
                raise ValueError("The dense GCA oracle requires unpadded packed metadata.")
            gca_metadata = {"dense_attention_mask": build_packed_gca_dense_mask(metadata, device=input_ids.device)}
        for decoder_layer in self.model.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_embeddings=(cos, sin),
                past_key_values=None,
                use_cache=False,
                hy3_gca_metadata=gca_metadata,
            )

        # ``final_layer`` (UNetUp, 2D convs over ``token_height × token_width``)
        # cannot run on a 1D shard, so gather the full sequence and compute the head
        # + flow loss replicated on every SP rank. The loss stays SP-invariant
        # because ``image_output_mask`` rides through un-sliced (see the
        # ``mean_global_loss`` note in sequence_layout.py).
        hidden_states = gather_outputs(hidden_states, gather_dim=1, group=sp_group)
        image_hidden_states = hidden_states.gather(
            1,
            image_payload_indices.unsqueeze(-1).expand(-1, -1, hidden_size),
        )
        image_hidden_states = image_hidden_states.reshape(num_samples, token_height * token_width, hidden_size)
        prediction = self.final_layer(
            image_hidden_states,
            self.time_embed_2(timesteps),
            token_height,
            token_width,
        )
        flow_loss = flow_matching_loss(prediction, flow_batch["flow_target"])
        output = HunyuanImage3ReferenceOutput(
            loss={"image_decoder_loss": flow_loss},
            diffusion_prediction=prediction,
            flow_target=flow_batch["flow_target"],
            latents=flow_batch["latents"],
            noised_latents=flow_batch["noised_latents"],
            sigmas=flow_batch["sigmas"],
            timesteps=timesteps,
            transformer_hidden_states=hidden_states,
        )
        if not return_dict:
            return tuple(output.values())
        return output

    @staticmethod
    def _validate_packed_metadata(input_ids, metadata):
        # Only cross-boundary invariants are checked here. Structural properties of
        # ``metadata`` are postconditions of ``compile_single_gen_t2i_packed``; what
        # it cannot guarantee is agreement with the *collator* (which pads input_ids
        # independently) and the *trainer* (which moves tensors to the device).
        if input_ids is None or not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("Packed input_ids must have shape [1, sequence_length].")
        if input_ids.shape[0] != 1:
            raise ValueError("The packed varlen layout uses a single [1, T] packed row.")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must use an integer dtype.")
        if metadata.get("padded_sequence_length") != input_ids.shape[1]:
            raise ValueError("Packed input_ids length must match padded_sequence_length.")
        for name in ("position_ids", "timestep_positions", "image_payload_indices", *GCA_VARLEN_TENSOR_KEYS):
            if metadata[name].device != input_ids.device:
                raise ValueError(f"Packed metadata[{name!r}] must be on the input_ids device.")
        return metadata

    def _validate_reference_components(self):
        # Live guard, not a formality: ``component_policy`` may drop any of these
        # (a future understanding-only recipe would), and the packed T2I forward
        # dereferences all five unconditionally.
        required = ("patch_embed", "timestep_emb", "time_embed", "time_embed_2", "final_layer")
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise RuntimeError(f"single_gen_t2i_v1 reference components are absent: {missing}.")

    def _get_latent_posterior(self, component_inputs, input_ids):
        # One gen_image component carries EITHER a posterior (mean/logvar) or raw
        # pixel_values for the frozen online VAE; both feed the identical
        # prepare_reference_flow_batch path, so the two entry points give the same
        # x0/loss under the same flow RNG by construction. ``pixel_values`` is the
        # only form the data pipeline emits; ``latent_posterior`` is the
        # direct-injection form, used to feed latents to a model built without a VAE
        # encoder and to pin online/injected equivalence in tests.
        if not isinstance(component_inputs, dict):
            raise ValueError("component_inputs must be a dict with one gen_image reference input.")
        keys = set(component_inputs)
        if keys == {"latent_posterior"}:
            posterior = component_inputs["latent_posterior"]
            if not isinstance(posterior, dict) or set(posterior) != {"mean", "logvar"}:
                raise ValueError("latent_posterior must contain exactly mean and logvar tensors.")
            posterior_mean = self._stack_packed_samples(posterior["mean"], name="latent_posterior.mean")
            posterior_logvar = self._stack_packed_samples(posterior["logvar"], name="latent_posterior.logvar")
            if posterior_mean.shape != posterior_logvar.shape:
                raise ValueError("Cached posterior mean and logvar must share shape after stacking.")
        elif keys == {"pixel_values"}:
            posterior_mean, posterior_logvar = self._encode_pixel_values_to_posterior(component_inputs["pixel_values"])
        else:
            raise ValueError(
                "component_inputs must contain either cached 'latent_posterior' or online 'pixel_values'."
            )
        if posterior_mean.device != input_ids.device or posterior_logvar.device != input_ids.device:
            raise ValueError("Latent posterior tensors must be on the input_ids device.")
        return posterior_mean, posterior_logvar

    @staticmethod
    def _stack_packed_samples(value, *, name: str):
        # ``pack_mode="list"`` staging arrives as ``List[Tensor]``, one
        # ``[1, C, H, W]`` per packed sample; a directly-injected stacked tensor
        # passes through unchanged. Heterogeneous shapes are a data-pipeline
        # misconfiguration: the fixed-resolution processor CenterCrops every sample
        # to the same (H, W).
        if isinstance(value, torch.Tensor):
            return value
        if not isinstance(value, list) or not value:
            raise TypeError(f"{name} must be a non-empty List[Tensor] or a stacked tensor.")
        if not all(isinstance(entry, torch.Tensor) for entry in value):
            raise TypeError(f"{name} list entries must all be tensors.")
        shapes = {tuple(entry.shape[1:]) for entry in value}  # ignore per-sample batch dim (always 1)
        if len(shapes) != 1:
            raise ValueError(
                f"Heterogeneous latent shapes in {name} are not supported: the fixed CenterCrop "
                "resolution should give every packed sample the same (C, H, W). "
                "Got per-sample shapes: " + ", ".join(str(s) for s in sorted(shapes))
            )
        return torch.cat(value, dim=0)

    def _encode_pixel_values_to_posterior(self, pixel_values):
        if self.component_policy.vae_encoder != "frozen" or not hasattr(self, "vae"):
            raise RuntimeError("Online pixel_values latents require component_policy vae_encoder='frozen'.")
        pixel_values = self._stack_packed_samples(pixel_values, name="pixel_values")
        # The frozen encoder is kept FP32 (declared via ParallelPlan
        # fsdp_ignored_param_fqn_patterns=["vae.*"] and cast by
        # apply_pre_fsdp_dtype_policy); match its dtype AND device so the encode runs
        # FP32 end to end. The device co-cast is not cosmetic: ``pack_mode="list"``
        # staging survives the trainer's batch-to-device sweep on CPU (the sweep
        # recurses into dicts but not into ``List[Tensor]``), so without it
        # ``F.conv3d`` sees CPU input x CUDA weight and dispatches to the CPU-only
        # ``_slow_conv3d_forward`` with a misleading "CUDA backend" NotImplementedError.
        vae_param = next(self.vae.parameters())
        posterior = self.vae.encode(pixel_values.to(dtype=vae_param.dtype, device=vae_param.device))
        posterior_mean = posterior.mean
        posterior_logvar = posterior.logvar
        # ``encode`` promotes a single image to 5D ``[B, C, T_lat, H, W]`` with one
        # latent frame; drop the temporal axis so the online posterior matches the
        # injected ``[B, C, H, W]`` contract.
        if posterior_mean.ndim == 5:
            if posterior_mean.shape[2] != 1:
                raise ValueError("Online VAE posterior must reduce to a single latent frame.")
            posterior_mean = posterior_mean.squeeze(2)
            posterior_logvar = posterior_logvar.squeeze(2)
        return posterior_mean, posterior_logvar

    def _validate_reference_grid(self, grid_hw, noised_latents):
        # The posteriors are stacked into one [B, C, H, W] batch and the 2D-conv
        # head needs a single grid, so a packed batch must share one. The
        # fixed-resolution CenterCrop processor guarantees it; a mismatch means a
        # data-pipeline misconfiguration (or, later, a bucket-batching bug).
        if noised_latents.shape[1] != self.config.vae["latent_channels"]:
            raise ValueError("Cached posterior channel count does not match vae.latent_channels.")
        normalized_grids = {tuple(grid) for grid in grid_hw}
        if len(normalized_grids) != 1:
            raise ValueError("A packed batch requires one shared image grid.")
        latent_height, latent_width = noised_latents.shape[-2:]
        if latent_height % self.config.patch_size or latent_width % self.config.patch_size:
            raise ValueError("Cached latent dimensions must be divisible by patch_size.")
        expected_grid = (latent_height // self.config.patch_size, latent_width // self.config.patch_size)
        if normalized_grids != {expected_grid}:
            raise ValueError("Compiled image grid does not match the cached latent shape.")
        return expected_grid

    # --- Patch.2 ---


__all__ = ["HunyuanImage3ForCausalMM"]
