# Modified by the VeOmni contributors for native HunyuanVideo 1.5 integration.
# Changes include import/runtime adaptation and repository formatting; see veomni/models/diffusers/hunyuanvideo15/SOURCE.txt.

# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from transformers.modeling_layers import GradientCheckpointingLayer

from .byt5 import ByT5Mapper
from .modules.activation_layers import get_activation_layer
from .modules.attention import parallel_attention, validate_attention_mode
from .modules.embed_layers import PatchEmbed, TextProjection, TimestepEmbedder, VisionProjection
from .modules.mlp_layers import MLP, FinalLayer, LinearWarpforSingle, MLPEmbedder
from .modules.modulate_layers import ModulateDiT, apply_gate, modulate
from .modules.norm_layers import get_norm_layer
from .modules.posemb_layers import RotaryTables, apply_rotary_emb, get_nd_rotary_pos_embed
from .modules.token_refiner import SingleTokenRefiner


class MMDoubleStreamBlock(GradientCheckpointingLayer):
    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str = None,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.deterministic = False
        self.heads_num = heads_num
        self.attn_mode = validate_attention_mode(attn_mode)
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.img_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, **factory_kwargs)
        self.img_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-06, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-06, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, **factory_kwargs)
        self.img_mlp = MLP(
            hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs
        )
        self.txt_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, **factory_kwargs)
        self.txt_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-06, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-06, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, **factory_kwargs)
        self.txt_mlp = MLP(
            hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs
        )
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
        block_idx=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate = self.img_mod(
            vec
        ).chunk(6, dim=-1)
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = self.txt_mod(
            vec
        ).chunk(6, dim=-1)
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)
        img_q = self.img_attn_q(img_modulated)
        img_k = self.img_attn_k(img_modulated)
        img_v = self.img_attn_v(img_modulated)
        img_q = rearrange(img_q, "B L (H D) -> B L H D", H=self.heads_num)
        img_k = rearrange(img_k, "B L (H D) -> B L H D", H=self.heads_num)
        img_v = rearrange(img_v, "B L (H D) -> B L H D", H=self.heads_num)
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, (
                f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            )
            img_q, img_k = (img_qq, img_kk)
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
        txt_q = self.txt_attn_q(txt_modulated)
        txt_k = self.txt_attn_k(txt_modulated)
        txt_v = self.txt_attn_v(txt_modulated)
        txt_q = rearrange(txt_q, "B L (H D) -> B L H D", H=self.heads_num)
        txt_k = rearrange(txt_k, "B L (H D) -> B L H D", H=self.heads_num)
        txt_v = rearrange(txt_v, "B L (H D) -> B L H D", H=self.heads_num)
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)
        attn_mode = "flash" if is_flash else self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
        )
        img_attn, txt_attn = (attn[:, : img_q.shape[1]].contiguous(), attn[:, img_q.shape[1] :].contiguous())
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)), gate=img_mod2_gate
        )
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)), gate=txt_mod2_gate
        )
        return (img, txt)


class MMSingleStreamBlock(GradientCheckpointingLayer):
    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str = None,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.deterministic = False
        self.attn_mode = validate_attention_mode(attn_mode)
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** (-0.5)
        self.linear1_q = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_k = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_v = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_mlp = nn.Linear(hidden_size, mlp_hidden_dim, **factory_kwargs)
        self.linear2 = LinearWarpforSingle(hidden_size + mlp_hidden_dim, hidden_size, bias=True, **factory_kwargs)
        self.mlp_act = get_activation_layer(mlp_act_type)()
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-06, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-06, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, **factory_kwargs)
        self.modulation = ModulateDiT(hidden_size, factor=3, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
    ) -> torch.Tensor:
        """Forward pass for the single stream block."""
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        q = self.linear1_q(x_mod)
        k = self.linear1_k(x_mod)
        v = self.linear1_v(x_mod)
        q = rearrange(q, "B L (H D) -> B L H D", H=self.heads_num)
        k = rearrange(k, "B L (H D) -> B L H D", H=self.heads_num)
        v = rearrange(v, "B L (H D) -> B L H D", H=self.heads_num)
        mlp = self.linear1_mlp(x_mod)
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)
        img_q, txt_q = (q[:, :-txt_len, :, :], q[:, -txt_len:, :, :])
        img_k, txt_k = (k[:, :-txt_len, :, :], k[:, -txt_len:, :, :])
        img_v, txt_v = (v[:, :-txt_len, :, :], v[:, -txt_len:, :, :])
        img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
        assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, (
            f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
        )
        img_q, img_k = (img_qq, img_kk)
        if is_flash:
            attn_mode = "flash"
        else:
            attn_mode = self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
        )
        output = self.linear2(attn, self.mlp_act(mlp))
        return x + apply_gate(output, gate=mod_gate)


class HunyuanVideo_1_5_DiffusionTransformer(nn.Module):
    """
    HunyuanVideo Transformer backbone.

    Args:
        patch_size (list): The size of the patch.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        hidden_size (int): The hidden size of the transformer backbone.
        heads_num (int): The number of attention heads.
        mlp_width_ratio (float): Width ratio for the transformer MLPs.
        mlp_act_type (str): Activation type for the transformer MLPs.
        mm_double_blocks_depth (int): Number of double-stream transformer blocks.
        mm_single_blocks_depth (int): Number of single-stream transformer blocks.
        rope_dim_list (list): Rotary embedding dim for t, h, w.
        qkv_bias (bool): Use bias in qkv projection.
        qk_norm (bool): Whether to use qk norm.
        qk_norm_type (str): Type of qk norm.
        guidance_embed (bool): Use guidance embedding for distillation.
        text_projection (str): Text input projection. Default is "single_refiner".
        use_attention_mask (bool): If to use attention mask.
        text_states_dim (int): Text encoder output dim.
        text_states_dim_2 (int): Secondary text encoder output dim.
        text_pool_type (str): Type for text pooling.
        rope_theta (int): Rotary embedding theta parameter.
        attn_mode (str): Attention mode identifier.
        attn_param (dict): Attention parameter dictionary.
        glyph_byT5_v2 (bool): Use ByT5 glyph module.
        vision_projection (str): Vision condition embedding mode.
        vision_states_dim (int): Vision encoder states input dim.
        is_reshape_temporal_channels (bool): For video VAE adaptation.
        use_cond_type_embedding (bool): Use condition type embedding.
    """

    def __init__(
        self,
        patch_size: tuple = (1, 2, 2),
        in_channels: int = 4,
        concat_condition: bool = True,
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: tuple = (16, 56, 56),
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,
        use_meanflow: bool = False,
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        text_states_dim: int = 4096,
        text_states_dim_2: int = 768,
        text_pool_type: str = None,
        rope_theta: int = 256,
        attn_mode: str = "flash",
        attn_param: dict = None,
        glyph_byT5_v2: bool = False,
        vision_projection: str = "none",
        vision_states_dim: int = 1280,
        is_reshape_temporal_channels: bool = False,
        use_cond_type_embedding: bool = False,
        ideal_resolution: str = None,
        ideal_task: str = None,
    ):
        super().__init__()
        factory_kwargs = {}
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection
        self.attn_mode = validate_attention_mode(attn_mode)
        self.text_pool_type = text_pool_type
        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2
        self.vision_states_dim = vision_states_dim
        self.glyph_byT5_v2 = glyph_byT5_v2
        if self.glyph_byT5_v2:
            self.byt5_in = ByT5Mapper(
                in_dim=1472, out_dim=2048, hidden_dim=2048, out_dim1=hidden_size, use_residual=False
            )
        if hidden_size % heads_num != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}")
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(f"Got {rope_dim_list} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.img_in = PatchEmbed(
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            is_reshape_temporal_channels=is_reshape_temporal_channels,
            concat_condition=concat_condition,
            **factory_kwargs,
        )
        if vision_projection == "linear":
            self.vision_in = VisionProjection(input_dim=self.vision_states_dim, output_dim=self.hidden_size)
        else:
            self.vision_in = None
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                text_states_dim, self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(text_states_dim, hidden_size, heads_num, depth=2, **factory_kwargs)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")
        self.time_in = TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)
        self.vector_in = (
            MLPEmbedder(self.text_states_dim_2, self.hidden_size, **factory_kwargs)
            if self.text_pool_type is not None
            else None
        )
        self.guidance_in = (
            TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)
            if guidance_embed
            else None
        )
        self.time_r_in = (
            TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)
            if use_meanflow
            else None
        )
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )
        self.final_layer = FinalLayer(
            self.hidden_size, self.patch_size, self.out_channels, get_activation_layer("silu"), **factory_kwargs
        )
        if attn_param is None:
            self.attn_param = {
                "win_size": [[3, 3, 3]],
                "win_type": "fixed",
                "win_ratio": 10,
                "tile_size": [6, 8, 8],
                "ssta_topk": 64,
                "ssta_threshold": 0.0,
                "ssta_lambda": 0.7,
                "ssta_sampling_type": "importance",
                "ssta_adaptive_pool": None,
                "attn_sparse_type": "ssta",
                "attn_pad_type": "zero",
                "attn_use_text_mask": 1,
                "attn_mask_share_within_head": 0,
            }
        else:
            self.attn_param = attn_param
        if use_cond_type_embedding:
            self.cond_type_embedding = nn.Embedding(3, self.hidden_size)
            self.cond_type_embedding.weight.data.fill_(0)
            assert self.glyph_byT5_v2, "text type embedding is only used when glyph_byT5_v2 is True"
            assert vision_projection is not None, "text type embedding is only used when vision_projection is not None"
        else:
            self.cond_type_embedding = None

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def get_rotary_pos_embed(self, rope_sizes):
        target_ndim = 3
        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            tuple(rope_dim_list), rope_sizes, theta=self.rope_theta, use_real=True, theta_rescale_factor=1
        )
        return (freqs_cos, freqs_sin)

    def reorder_txt_token(self, byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=False, is_reorder=True):
        if is_reorder:
            reorder_txt = []
            reorder_mask = []
            for i in range(text_mask.shape[0]):
                byt5_text_mask_i = byt5_text_mask[i].bool()
                text_mask_i = text_mask[i].bool()
                byt5_txt_i = byt5_txt[i]
                txt_i = txt[i]
                if zero_feat:
                    pad_byt5 = torch.zeros_like(byt5_txt_i[~byt5_text_mask_i])
                    pad_text = torch.zeros_like(txt_i[~text_mask_i])
                    reorder_txt_i = torch.cat(
                        [byt5_txt_i[byt5_text_mask_i], txt_i[text_mask_i], pad_byt5, pad_text], dim=0
                    )
                else:
                    reorder_txt_i = torch.cat(
                        [
                            byt5_txt_i[byt5_text_mask_i],
                            txt_i[text_mask_i],
                            byt5_txt_i[~byt5_text_mask_i],
                            txt_i[~text_mask_i],
                        ],
                        dim=0,
                    )
                reorder_mask_i = torch.cat(
                    [
                        byt5_text_mask_i[byt5_text_mask_i],
                        text_mask_i[text_mask_i],
                        byt5_text_mask_i[~byt5_text_mask_i],
                        text_mask_i[~text_mask_i],
                    ],
                    dim=0,
                )
                reorder_txt.append(reorder_txt_i)
                reorder_mask.append(reorder_mask_i)
            reorder_txt = torch.stack(reorder_txt)
            reorder_mask = torch.stack(reorder_mask).to(dtype=torch.int64)
        else:
            reorder_txt = torch.concat([byt5_txt, txt], dim=1)
            reorder_mask = torch.concat([byt5_text_mask, text_mask], dim=1).to(dtype=torch.int64)
        return (reorder_txt, reorder_mask)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        text_states: torch.Tensor,
        text_states_2: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r=None,
        vision_states: torch.Tensor = None,
        output_features=False,
        output_features_stride=8,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        guidance=None,
        mask_type="t2v",
        extra_kwargs=None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if guidance is None:
            guidance = torch.tensor([6016.0], device=hidden_states.device, dtype=torch.bfloat16)
        img = x = hidden_states
        text_mask = encoder_attention_mask
        t = timestep
        txt = text_states
        bs, _, ot, oh, ow = x.shape
        tt, th, tw = (ot // self.patch_size[0], oh // self.patch_size[1], ow // self.patch_size[2])
        self.attn_param["thw"] = [tt, th, tw]
        if freqs_cos is None and freqs_sin is None:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))
        img = self.img_in(img)
        vec = self.time_in(t)
        if text_states_2 is not None:
            vec_2 = self.vector_in(text_states_2)
            vec = vec + vec_2
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(guidance)
        if timestep_r is not None:
            vec = vec + self.time_r_in(timestep_r)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")
        if self.cond_type_embedding is not None:
            cond_emb = self.cond_type_embedding(
                torch.zeros_like(txt[:, :, 0], device=text_mask.device, dtype=torch.long)
            )
            txt = txt + cond_emb
        if self.glyph_byT5_v2:
            byt5_text_states = extra_kwargs["byt5_text_states"]
            byt5_text_mask = extra_kwargs["byt5_text_mask"]
            byt5_txt = self.byt5_in(byt5_text_states)
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    torch.ones_like(byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long)
                )
                byt5_txt = byt5_txt + cond_emb
            txt, text_mask = self.reorder_txt_token(byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=True)
        if self.vision_in is not None and vision_states is not None:
            extra_encoder_hidden_states = self.vision_in(vision_states)
            if mask_type == "t2v" and torch.all(vision_states == 0):
                extra_attention_mask = torch.zeros(
                    (bs, extra_encoder_hidden_states.shape[1]), dtype=text_mask.dtype, device=text_mask.device
                )
                extra_encoder_hidden_states = extra_encoder_hidden_states * 0.0
            else:
                extra_attention_mask = torch.ones(
                    (bs, extra_encoder_hidden_states.shape[1]), dtype=text_mask.dtype, device=text_mask.device
                )
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    2
                    * torch.ones_like(
                        extra_encoder_hidden_states[:, :, 0],
                        dtype=torch.long,
                        device=extra_encoder_hidden_states.device,
                    )
                )
                extra_encoder_hidden_states = extra_encoder_hidden_states + cond_emb
            txt, text_mask = self.reorder_txt_token(extra_encoder_hidden_states, txt, extra_attention_mask, text_mask)
        freqs_cis = (
            RotaryTables(
                freqs_cos.to(device=hidden_states.device, dtype=torch.float32),
                freqs_sin.to(device=hidden_states.device, dtype=torch.float32),
            )
            if freqs_cos is not None
            else None
        )
        for index, block in enumerate(self.double_blocks):
            force_full_attn = (
                self.attn_mode in ["flex-block-attn"]
                and self.attn_param["win_type"] == "hybrid"
                and (self.attn_param["win_ratio"] > 0)
                and ((index + 1) % self.attn_param["win_ratio"] == 0 or index + 1 == len(self.double_blocks))
            )
            self.attn_param["layer-name"] = f"double_block_{index + 1}"
            # Reentrant checkpointing only tracks gradient-carrying positional inputs.
            # Keep img/txt/vec positional so the checkpoint backward reaches every block.
            img, txt = block(
                img,
                txt,
                vec,
                freqs_cis=freqs_cis,
                text_mask=text_mask,
                attn_param=self.attn_param,
                is_flash=force_full_attn,
                block_idx=index,
            )
        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]
        x = torch.cat((img, txt), 1)
        features_list = [] if output_features else None
        if len(self.single_blocks) > 0:
            for index, block in enumerate(self.single_blocks):
                force_full_attn = (
                    self.attn_mode in ["flex-block-attn"]
                    and self.attn_param["win_type"] == "hybrid"
                    and (self.attn_param["win_ratio"] > 0)
                    and ((index + 1) % self.attn_param["win_ratio"] == 0 or index + 1 == len(self.single_blocks))
                )
                self.attn_param["layer-name"] = f"single_block_{index + 1}"
                x = block(
                    x,
                    vec,
                    txt_len=txt_seq_len,
                    freqs_cis=freqs_cis,
                    text_mask=text_mask,
                    attn_param=self.attn_param,
                    is_flash=force_full_attn,
                )
                if output_features and index % output_features_stride == 0:
                    features_list.append(x[:, :img_seq_len, ...])
        img = x[:, :img_seq_len, ...]
        img = self.final_layer(img, vec)
        img = self.unpatchify(img, tt, th, tw)
        assert return_dict is False, "return_dict is not supported."
        if output_features:
            features_list = torch.stack(features_list, dim=0)
        else:
            features_list = None
        return (img, features_list)

    def unpatchify(self, x, t, h, w):
        """
        Unpatchify a tensorized input back to frame format.

        Args:
            x (Tensor): Input tensor of shape (N, T, patch_size**2 * C)
            t (int): Number of time steps
            h (int): Height in patch units
            w (int): Width in patch units

        Returns:
            Tensor: Output tensor of shape (N, C, t * pt, h * ph, w * pw)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def set_attn_mode(self, attn_mode: str):
        attn_mode = validate_attention_mode(attn_mode)
        self.attn_mode = validate_attention_mode(attn_mode)
        for block in self.double_blocks:
            block.attn_mode = attn_mode
        for block in self.single_blocks:
            block.attn_mode = attn_mode
