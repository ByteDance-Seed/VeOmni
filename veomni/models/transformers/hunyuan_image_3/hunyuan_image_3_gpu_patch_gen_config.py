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

This milestone adds a model-local, unpacked ``single_gen_t2i_v1`` reference
forward with dense GCA, generalized 2D RoPE, and the flow objective. Optimized
attention, sequence parallelism, and trainer integration remain separate.

Regen command:
patchgen veomni.models.transformers.hunyuan_image_3.hunyuan_image_3_gpu_patch_gen_config \
  -o veomni/models/transformers/hunyuan_image_3/generated --diff
"""

from patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.hunyuan_v1_moe.modeling_hunyuan_v1_moe",
    target_file="patched_modeling_hunyuan_image_3_gpu.py",
    description="HunyuanImage 3 official-layout import with single_gen_t2i_v1 dense reference forward",
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
    "veomni.schedulers.flow_matching",
    names=["flow_matching_loss", "prepare_reference_flow_batch"],
)
config.add_import(
    "veomni.utils.model_outputs",
    names=["HunyuanImage3ReferenceOutput"],
)
config.add_import(
    "veomni.models.transformers.hunyuan_image_3.patches.image_projection",
    names=["TimestepEmbedder", "UNetDown", "UNetUp"],
)
config.add_import(
    "veomni.models.transformers.hunyuan_image_3.reference_rope",
    names=["build_reference_2d_rope"],
)
config.add_import(
    "veomni.ops.kernels.attention.reference",
    names=["dense_gca_attention_forward"],
)
config.add_import(
    "veomni.models.transformers.hunyuan_image_3.patches.generalized_causal_attention",
    names=["GCA_VARLEN_METADATA_KEYS", "gca_varlen_attention_forward", "resolve_base_attention_implementation"],
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
    "veomni.models.transformers.hunyuan_image_3.metadata_collate",
    names=["collate_hunyuan_image_3_metadata", "get_hunyuan_image_3_extra_collate_infos"],
)

# The eager expert loop is replaced below with an OpSlot-guarded forward, so the
# HuggingFace ``@use_experts_implementation`` decorator is no longer applied and
# its import would be unused.
config.drop_import_names("use_experts_implementation")

# OpSlot for the fused / expert-parallel MoE experts path. Bound at model-build
# time by ``_bind_veomni_ops()`` in ``veomni/models/auto.py``; it stays unbound
# (eager fallback) unless ``moe_implementation`` selects a fused kernel. Expert
# parallelism is taken inside the fused kernel when the parallel state reports
# ``ep_enabled``. The declaration mirrors the qwen3_moe patch.
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
# 2. Dispatch the explicit reference path to model-local dense GCA.
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
        if kwargs.pop("hy3_reference_path", False):
            if past_key_values is not None:
                raise ValueError("single_gen_t2i_v1 reference attention does not support KV cache.")
            attention_output, attention_weights = dense_gca_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                scaling=self.scaling,
            )
        elif gca_metadata is not None:
            # Production fast path: two-call varlen GCA. When Ulysses SP is active
            # each rank arrives with a sequence shard and the full head set; the
            # all-to-all gathers the sequence and scatters heads so the two calls
            # run on the full packed sequence with a head shard, then the inverse
            # all-to-all restores the sequence shard. Both collapse to a no-op when
            # SP is disabled.
            #
            # Perf note: the packed layout is B==1 (asserted by _validate_packed_metadata).
            # We SQUEEZE the batch dim before every A2A so that scatter_dim/gather_dim
            # are both <= 1, which hits the fast dist.all_to_all_single (contiguous
            # buffer) path in ``ulysses.all_to_all_tensor``. Without the squeeze the
            # tensor is 4D with scatter_dim=2 and falls into the slow list-based
            # ``dist.all_to_all`` path (extra tensor_split + per-shard .contiguous()
            # + torch.cat). Numerically identical either way.
            if past_key_values is not None:
                raise ValueError("single_gen_t2i_v1 fast path does not support KV cache.")
            base_implementation = resolve_base_attention_implementation(self.config._attn_implementation)
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                base_implementation,
                eager_attention_forward,
            )
            # [1, H, T_shard, D] -> [T_shard, H, D] (batch known to be 1 for packed)
            query_seq_first = query_states.transpose(1, 2).squeeze(0)
            key_seq_first = key_states.transpose(1, 2).squeeze(0)
            value_seq_first = value_states.transpose(1, 2).squeeze(0)
            # Fast A2A: scatter head_dim=1, gather seq_dim=0 -> dist.all_to_all_single
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
            # GCA returns [1, T_full, H_shard, D]; squeeze for the fast inverse A2A
            # (scatter seq_dim=0, gather head_dim=1 -> both <= 1), then unsqueeze back.
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
# shards both parameters on dim 0 for EP. The eager loop is kept as the exact
# fallback for the unbound (moe_implementation="eager") case.
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
        # Modification: OpSlot guard — use the fused/EP MoE kernel when bound.
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
# 2. Add the unpacked cached-posterior single_gen_t2i_v1 reference forward.
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
        if self.component_policy.vae_encoder != "absent" or self.component_policy.vae_decoder != "absent":
            self.vae = HunyuanImage3VAE(
                config.vae,
                build_encoder=self.component_policy.vae_encoder != "absent",
                build_decoder=self.component_policy.vae_decoder != "absent",
            )

        self.pad_id = config.pad_id
        self.vocab_size = config.vocab_size
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

    def get_fsdp_ignored_params(self):
        # RFC §5.2: the frozen VAE encoder must stay replicated FP32 -- out of FSDP
        # sharding AND mixed precision -- so the online latents are not perturbed by
        # BF16 (measured bf16 latent-mean rel_max ~4-6%). This hook is called by
        # parallelize_model_fsdp2 right before the root fully_shard, while params are
        # still on meta (pre-to_empty/load): cast the VAE to FP32 here so the disk FP32
        # weights load without a BF16 round-trip, then hand its params to the ignored
        # set. Returns None when no VAE is built (e.g. posterior_cache), a no-op for
        # every other model (opt-in hook).
        if not hasattr(self, "vae"):
            return None
        self.vae.float()
        return set(self.vae.parameters())

    def get_position_id_func(self):
        # The single_gen_t2i_v1 transform emits no position_ids; the packed
        # compiler owns the 2D coordinates. Present so the VLM data-transform
        # builder (which calls this unconditionally) does not AttributeError.
        return None

    def get_extra_collate_infos(self):
        # Declare the per-key pack/pad/slice topology for the generation staging
        # tensors (RFC §7.3). input_ids/labels/image_output_mask stay unsliced;
        # the model performs the Ulysses slice internally on the full sequence.
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
            # Frozen encoder runs FP32 (RFC §5.2). On the FSDP path get_fsdp_ignored_params
            # keeps it out of sharding/MP; cast here too so non-FSDP paths are FP32 as well.
            self.vae.float()
            self.vae.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        self.apply_component_policy()
        return self

    # --- Patch.2 ---
    def forward(
        self,
        input_ids=None,
        component_inputs=None,
        hy3_sequence_metadata=None,
        flow_config=None,
        flow_step_context=None,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        # The trainer calls ``model(**micro_batch, use_cache=False)``; ``**kwargs``
        # absorbs collated training-batch keys the flow forward does not consume
        # (labels, attention_mask, dummy position_ids, FA cu_seqlens, ...).
        if use_cache:
            raise ValueError("single_gen_t2i_v1 reference forward requires use_cache=False.")
        self._validate_reference_components()
        if isinstance(hy3_sequence_metadata, dict) and hy3_sequence_metadata.get("layout") == "packed_varlen":
            return self._forward_packed_varlen(
                input_ids=input_ids,
                component_inputs=component_inputs,
                hy3_sequence_metadata=hy3_sequence_metadata,
                flow_config=flow_config,
                flow_step_context=flow_step_context,
                return_dict=return_dict,
            )
        metadata = self._validate_reference_metadata(input_ids, hy3_sequence_metadata)
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("Reference sequence length exceeds max_position_embeddings.")
        posterior_mean, posterior_logvar = self._get_latent_posterior(component_inputs, input_ids)
        if posterior_mean.shape[0] != input_ids.shape[0]:
            raise ValueError("Latent posterior batch size must match input_ids.")
        flow_batch = prepare_reference_flow_batch(
            posterior_mean,
            posterior_logvar,
            vae_config=self.config.vae,
            flow_config=flow_config,
            flow_step_context=flow_step_context,
        )

        batch_size = input_ids.shape[0]
        grid_height, grid_width = self._validate_reference_grid(
            metadata["grid_hw"],
            flow_batch["noised_latents"],
        )
        timesteps = flow_batch["timesteps"]
        time_embedding = self.time_embed(timesteps)
        image_sequence, token_height, token_width = self.patch_embed(
            flow_batch["noised_latents"],
            time_embedding,
        )
        if (token_height, token_width) != (grid_height, grid_width):
            raise ValueError("Compiled image grid does not match the patch projection output.")

        hidden_states = self.model.embed_tokens(input_ids)
        if image_sequence.shape[1] != grid_height * grid_width:
            raise ValueError("The patch projection token count does not match the compiled image grid.")
        image_payload_indices = metadata["image_payload_indices"]
        if image_payload_indices.shape[1] != image_sequence.shape[1]:
            raise ValueError("Compiled image payload indices do not match the patch projection token count.")
        hidden_states = hidden_states.scatter(
            1,
            image_payload_indices.unsqueeze(-1).expand(-1, -1, self.config.hidden_size),
            image_sequence,
        )

        timestep_embeddings = self.timestep_emb(timesteps)
        hidden_states = hidden_states.scatter(
            1,
            metadata["timestep_positions"].view(batch_size, 1, 1).expand(-1, 1, self.config.hidden_size),
            timestep_embeddings.unsqueeze(1),
        )

        rope_scaling = self.config.image_rope_scaling or {}
        if rope_scaling.get("type", "custom") != "custom":
            raise ValueError("single_gen_t2i_v1 reference forward requires custom 2D RoPE.")
        position_embeddings = build_reference_2d_rope(
            metadata["position_ids"],
            head_dim=self.config.head_dim,
            rope_theta=self.config.rope_parameters["rope_theta"],
            base_rescale_factor=float(rope_scaling.get("factor", 1.0)),
            dtype=hidden_states.dtype,
        )
        for decoder_layer in self.model.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=metadata["dense_attention_mask"],
                position_embeddings=position_embeddings,
                past_key_values=None,
                use_cache=False,
                hy3_reference_path=True,
            )

        image_hidden_states = hidden_states.gather(
            1,
            image_payload_indices.unsqueeze(-1).expand(-1, -1, self.config.hidden_size),
        )
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

    def _forward_packed_varlen(
        self,
        *,
        input_ids,
        component_inputs,
        hy3_sequence_metadata,
        flow_config,
        flow_step_context,
        return_dict,
    ):
        # Packed production path: two-call varlen GCA + optional Ulysses SP,
        # validated against the dense oracle. Samples are laid out contiguously in
        # one [1, T] row; a bucket-synchronized micro-step gives every packed
        # sample the same grid, so posteriors batch over num_samples.
        metadata = self._validate_packed_metadata(input_ids, hy3_sequence_metadata)
        num_samples = metadata["num_samples"]
        if metadata["sequence_length"] > self.config.max_position_embeddings:
            raise ValueError("Packed sequence length exceeds max_position_embeddings.")

        posterior_mean, posterior_logvar = self._get_latent_posterior(component_inputs, input_ids)
        if posterior_mean.shape[0] != num_samples:
            raise ValueError("Packed latent posterior must carry one entry per packed sample.")
        flow_batch = prepare_reference_flow_batch(
            posterior_mean,
            posterior_logvar,
            vae_config=self.config.vae,
            flow_config=flow_config,
            flow_step_context=flow_step_context,
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
        cos, sin = build_reference_2d_rope(
            metadata["position_ids"],
            head_dim=self.config.head_dim,
            rope_theta=self.config.rope_parameters["rope_theta"],
            base_rescale_factor=float(rope_scaling.get("factor", 1.0)),
            dtype=hidden_states.dtype,
        )

        # Ulysses SP: replicate the embedding/projection on every rank, then slice
        # the residual stream so each decoder layer's attention all-to-all runs on
        # a sequence shard. Both slices are no-ops when SP is disabled. The same
        # Ulysses group drives the sequence slice and the attention all-to-all.
        #
        # Why model-side slice (not collator-side ``sp_slice=True``): single-stream
        # packed layout mixes text/image/timestep tokens by absolute full-sequence
        # coordinates (``image_payload_indices`` / ``timestep_positions``), and the
        # 2D-conv ``final_layer`` needs the full image grid. Pre-slicing in the
        # collator would force per-rank index remapping in every scatter and a
        # gather before the head anyway — no compute saved, all glue added. So the
        # extra collate infos declare ``sp_slice=False`` (metadata_collate.py); the
        # collator only pads to a multiple of sp_size, and Ulysses lives here.
        #
        # Cost: everything above this line — ``embed_tokens``, ``patch_embed``,
        # ``time_embed``, both scatters, and 2D-RoPE build — runs replicated on
        # every SP rank, and ``hidden_states`` briefly holds the full ``[1, T,
        # hidden]`` residual before the slice. Cheap ops on the compute side; the
        # real SP tax measured on H20 is ~5pp MFU from 256 A2A/step NCCL launch
        # overhead (see docs/design/hunyuan_image_3_sp_toy_perf.md), addressed by
        # ``async_ulysses_dit`` — not by moving the slice.
        sp_group = get_ulysses_sequence_parallel_group()
        hidden_states = slice_input_tensor(hidden_states, dim=1, group=sp_group)
        cos = slice_input_tensor(cos, dim=1, group=sp_group)
        sin = slice_input_tensor(sin, dim=1, group=sp_group)
        gca_metadata = {key: metadata[key] for key in GCA_VARLEN_METADATA_KEYS}
        for decoder_layer in self.model.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_embeddings=(cos, sin),
                past_key_values=None,
                use_cache=False,
                hy3_gca_metadata=gca_metadata,
            )

        # The 2D-conv final head cannot run on a 1D shard, so gather the full
        # sequence and compute the flow loss replicated on every SP rank.
        # ``final_layer`` (UNetUp with 2D convs on ``token_height × token_width``)
        # is the hard reason model-side slice is the only viable layout: any SP
        # scheme has to gather before this head, so pre-slicing upstream buys
        # nothing. The replicated head cost is small (params + one image grid per
        # rank); the flow loss is naturally SP-invariant since ``image_output_mask``
        # rides through un-sliced (see metadata_collate.py's docstring on
        # ``mean_global_loss``).
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
        if input_ids is None or not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("Packed input_ids must have shape [1, sequence_length].")
        if input_ids.shape[0] != 1:
            raise ValueError("The packed varlen layout uses a single [1, T] packed row.")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must use an integer dtype.")
        if not isinstance(metadata, dict) or metadata.get("capability") != "single_gen_t2i_v1":
            raise ValueError("hy3_sequence_metadata must be compiled for single_gen_t2i_v1.")
        if metadata.get("layout") != "packed_varlen":
            raise ValueError("The packed forward requires the packed_varlen compiled layout.")
        num_samples = metadata.get("num_samples")
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("Packed metadata must record a positive num_samples.")
        if metadata.get("padded_sequence_length") != input_ids.shape[1]:
            raise ValueError("Packed input_ids length must match padded_sequence_length.")
        if not isinstance(metadata.get("sequence_length"), int):
            raise ValueError("Packed metadata must record the logical sequence_length.")
        required_tensors = (
            "position_ids",
            "timestep_positions",
            "image_payload_indices",
            *GCA_VARLEN_METADATA_KEYS[:6],
        )
        for name in required_tensors:
            tensor = metadata.get(name)
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Packed metadata[{name!r}] must be a tensor.")
            if tensor.device != input_ids.device:
                raise ValueError(f"Packed metadata[{name!r}] must be on the input_ids device.")
        grid_hw = metadata.get("grid_hw")
        if not isinstance(grid_hw, tuple) or len(grid_hw) != num_samples:
            raise ValueError("Packed grid_hw must contain one entry per packed sample.")
        return metadata

    def _validate_reference_components(self):
        required = ("patch_embed", "timestep_emb", "time_embed", "time_embed_2", "final_layer")
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise RuntimeError(f"single_gen_t2i_v1 reference components are absent: {missing}.")

    @staticmethod
    def _validate_reference_metadata(input_ids, metadata):
        if input_ids is None or not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence_length].")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must use an integer dtype.")
        if not isinstance(metadata, dict) or metadata.get("capability") != "single_gen_t2i_v1":
            raise ValueError("hy3_sequence_metadata must be compiled for single_gen_t2i_v1.")

        batch_size, sequence_length = input_ids.shape
        if batch_size == 0 or sequence_length == 0:
            raise ValueError("The unpacked reference batch and sequence must be non-empty.")
        if metadata.get("sequence_length") != sequence_length:
            raise ValueError("Compiled sequence_length must match input_ids.")
        expected_shapes = {
            "position_ids": (batch_size, 2, sequence_length),
            "dense_attention_mask": (batch_size, sequence_length, sequence_length),
            "timestep_sample_index": (batch_size, sequence_length),
            "timestep_positions": (batch_size,),
            "image_output_mask": (batch_size, sequence_length),
        }
        for name, shape in expected_shapes.items():
            tensor = metadata.get(name)
            if not isinstance(tensor, torch.Tensor) or tensor.shape != shape:
                raise ValueError(f"hy3_sequence_metadata[{name!r}] must have shape {shape}.")
            if tensor.device != input_ids.device:
                raise ValueError(f"hy3_sequence_metadata[{name!r}] must be on the input_ids device.")
        if metadata["position_ids"].dtype not in (torch.int32, torch.int64):
            raise TypeError("Compiled position_ids must use an integer dtype.")
        if metadata["dense_attention_mask"].dtype != torch.bool:
            raise TypeError("Compiled dense_attention_mask must be boolean.")
        if metadata["timestep_sample_index"].dtype not in (torch.int32, torch.int64):
            raise TypeError("Compiled timestep_sample_index must use an integer dtype.")
        if metadata["timestep_positions"].dtype not in (torch.int32, torch.int64):
            raise TypeError("Compiled timestep_positions must use an integer dtype.")
        if metadata["image_output_mask"].dtype != torch.bool:
            raise TypeError("Compiled image_output_mask must be boolean.")

        image_payload_indices = metadata.get("image_payload_indices")
        if (
            not isinstance(image_payload_indices, torch.Tensor)
            or image_payload_indices.ndim != 2
            or image_payload_indices.shape[0] != batch_size
        ):
            raise ValueError("hy3_sequence_metadata['image_payload_indices'] must have shape [batch, image_tokens].")
        if image_payload_indices.device != input_ids.device:
            raise ValueError("hy3_sequence_metadata['image_payload_indices'] must be on the input_ids device.")
        if image_payload_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("Compiled image_payload_indices must use an integer dtype.")

        grid_hw = metadata.get("grid_hw")
        if not isinstance(grid_hw, tuple) or len(grid_hw) != batch_size:
            raise ValueError("Compiled grid_hw must contain one entry per unpacked sample.")
        return metadata

    def _get_latent_posterior(self, component_inputs, input_ids):
        # One gen_image component carries EITHER a cached posterior (mean/logvar)
        # or raw pixel_values for the frozen online VAE. Both feed the identical
        # prepare_reference_flow_batch path, so online and cache produce the same
        # x0/loss under the same flow RNG by construction.
        #
        # P1a contract change: staging values are ``List[Tensor]`` (one entry per
        # packed sample, each ``[1, C, H_i, W_i]``) since the collator uses
        # ``pack_mode="list"`` for latent staging. Under mbs=1 that's a length-1
        # list; under mbs>1 + ``same_bucket_batching=True`` all entries share
        # ``(C, H, W)`` and are stacked into the batched fast path. The
        # heterogeneous mbs>1 case (``same_bucket_batching=False``) is rejected
        # here with a clear error — the downstream ``patch_embed`` / payload
        # scatter / ``final_layer`` haven't been per-sample'd yet (that's P1b in
        # plan_bucketing.md). Legacy stacked-tensor inputs are still accepted so
        # this function stays back-compatible for callers that never went through
        # the metadata hook.
        if not isinstance(component_inputs, dict):
            raise ValueError("component_inputs must be a dict with one gen_image reference input.")
        keys = set(component_inputs)
        if keys == {"latent_posterior"}:
            posterior = component_inputs["latent_posterior"]
            if not isinstance(posterior, dict) or set(posterior) != {"mean", "logvar"}:
                raise ValueError("latent_posterior must contain exactly mean and logvar tensors.")
            posterior_mean = self._stack_uniform_latent_list(posterior["mean"], name="latent_posterior.mean")
            posterior_logvar = self._stack_uniform_latent_list(posterior["logvar"], name="latent_posterior.logvar")
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
    def _stack_uniform_latent_list(value, *, name: str):
        # Accept either a legacy stacked tensor (back-compat with any non-list-mode
        # caller) or the P1a ``List[Tensor]`` staging. In the list case, stack the
        # per-sample entries into ``[num_samples, C, H, W]`` when shapes are
        # uniform; raise a P1a-scoped error otherwise so the misconfig points at
        # the right place (``same_bucket_batching=False`` + mixed resolutions is
        # deferred to P1b).
        if isinstance(value, torch.Tensor):
            return value
        if not isinstance(value, list) or not value:
            raise TypeError(f"{name} must be a non-empty List[Tensor] or a stacked tensor.")
        if not all(isinstance(entry, torch.Tensor) for entry in value):
            raise TypeError(f"{name} list entries must all be tensors.")
        shapes = {tuple(entry.shape[1:]) for entry in value}  # ignore per-sample batch dim (always 1)
        if len(shapes) != 1:
            raise NotImplementedError(
                f"Heterogeneous latent shapes in {name} are not supported yet (P1b). "
                "Use ``data.image_generation.same_bucket_batching=true`` (default) so every packed micro-batch "
                "shares one bucket. Got per-sample shapes: " + ", ".join(str(s) for s in sorted(shapes))
            )
        # Every entry already has a batch axis (1); cat along dim 0 yields [B, C, H, W].
        return torch.cat(value, dim=0)

    def _encode_pixel_values_to_posterior(self, pixel_values):
        if self.component_policy.vae_encoder != "frozen" or not hasattr(self, "vae"):
            raise RuntimeError("Online pixel_values latents require component_policy vae_encoder='frozen'.")
        # P1a: accept either the legacy stacked-tensor form or the ``List[Tensor]``
        # (one ``[1, C, H_i, W_i]`` per packed sample) produced by the ``pack_mode=
        # "list"`` collator. When shapes are uniform we stack and run a single
        # batched VAE encode (load-balanced fast path). Heterogeneous shapes are
        # rejected — P1b will lift that restriction with a per-sample encode loop
        # and per-sample downstream forward.
        pixel_values = self._stack_uniform_latent_list(pixel_values, name="pixel_values")
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError("Online pixel_values must be a tensor after stacking.")
        # The frozen encoder is kept FP32 (RFC §5.2, out of FSDP/MP via
        # get_fsdp_ignored_params); match the conv input dtype so the encode runs FP32
        # end to end. The FP32 posterior then feeds the same flow batch as the cached
        # FP32 mean/logvar path.
        vae_dtype = next(self.vae.parameters()).dtype
        distribution = self.vae.encode(pixel_values.to(dtype=vae_dtype))
        posterior_mean = distribution.mean
        posterior_logvar = distribution.logvar
        # ``encode`` promotes a single image to 5D ``[B, C, T_lat, H, W]`` with one
        # latent frame; drop the temporal axis so the online posterior matches the
        # cached ``[B, C, H, W]`` contract (the same squeeze ``sample_latents`` does).
        if posterior_mean.ndim == 5:
            if posterior_mean.shape[2] != 1:
                raise ValueError("Online VAE posterior must reduce to a single latent frame.")
            posterior_mean = posterior_mean.squeeze(2)
            posterior_logvar = posterior_logvar.squeeze(2)
        return posterior_mean, posterior_logvar

    def _validate_reference_grid(self, grid_hw, noised_latents):
        if noised_latents.shape[1] != self.config.vae["latent_channels"]:
            raise ValueError("Cached posterior channel count does not match vae.latent_channels.")
        normalized_grids = []
        for grid in grid_hw:
            if (
                not isinstance(grid, (tuple, list))
                or len(grid) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in grid)
            ):
                raise ValueError("Each compiled grid_hw entry must contain positive integer height and width.")
            normalized_grids.append(tuple(grid))
        if len(set(normalized_grids)) != 1:
            raise ValueError("The unpacked reference batch requires one shared image grid.")
        latent_height, latent_width = noised_latents.shape[-2:]
        if latent_height % self.config.patch_size or latent_width % self.config.patch_size:
            raise ValueError("Cached latent dimensions must be divisible by patch_size.")
        expected_grid = (latent_height // self.config.patch_size, latent_width // self.config.patch_size)
        if normalized_grids[0] != expected_grid:
            raise ValueError("Compiled image grid does not match the cached latent shape.")
        return expected_grid

    # --- Patch.2 ---


__all__ = ["HunyuanImage3ForCausalMM"]
