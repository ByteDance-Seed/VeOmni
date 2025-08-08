# Copyright 2024-2025 The Black-forest-labs Authors. All rights reserved.
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


import math
from contextlib import contextmanager

import torch
from einops import rearrange, repeat
from transformers import T5EncoderModel


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
    computation_device=None,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0,
        end=half_dim,
        dtype=torch.float32,
        device=timesteps.device if computation_device is None else computation_device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.device)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbeddings(torch.nn.Module):
    def __init__(self, dim_in, dim_out, computation_device=None):
        super().__init__()
        self.time_proj = TemporalTimesteps(
            num_channels=dim_in, flip_sin_to_cos=True, downscale_freq_shift=0, computation_device=computation_device
        )
        self.timestep_embedder = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out), torch.nn.SiLU(), torch.nn.Linear(dim_out, dim_out)
        )

    def forward(self, timestep, dtype):
        time_emb = self.time_proj(timestep).to(dtype)
        time_emb = self.timestep_embedder(time_emb)
        return time_emb


class TemporalTimesteps(torch.nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, computation_device=None):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.computation_device = computation_device

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            computation_device=self.computation_device,
        )
        return t_emb


class TileWorker:
    def __init__(self):
        pass

    def mask(self, height, width, border_width):
        # Create a mask with shape (height, width).
        # The centre area is filled with 1, and the border line is filled with values in range (0, 1].
        x = torch.arange(height).repeat(width, 1).T
        y = torch.arange(width).repeat(height, 1)
        mask = torch.stack([x + 1, height - x, y + 1, width - y]).min(dim=0).values
        mask = (mask / border_width).clip(0, 1)
        return mask

    def tile(self, model_input, tile_size, tile_stride, tile_device, tile_dtype):
        # Convert a tensor (b, c, h, w) to (b, c, tile_size, tile_size, tile_num)
        batch_size, channel, _, _ = model_input.shape
        model_input = model_input.to(device=tile_device, dtype=tile_dtype)
        unfold_operator = torch.nn.Unfold(kernel_size=(tile_size, tile_size), stride=(tile_stride, tile_stride))
        model_input = unfold_operator(model_input)
        model_input = model_input.view((batch_size, channel, tile_size, tile_size, -1))

        return model_input

    def tiled_inference(
        self, forward_fn, model_input, tile_batch_size, inference_device, inference_dtype, tile_device, tile_dtype
    ):
        # Call y=forward_fn(x) for each tile
        tile_num = model_input.shape[-1]
        model_output_stack = []

        for tile_id in range(0, tile_num, tile_batch_size):
            # process input
            tile_id_ = min(tile_id + tile_batch_size, tile_num)
            x = model_input[:, :, :, :, tile_id:tile_id_]
            x = x.to(device=inference_device, dtype=inference_dtype)
            x = rearrange(x, "b c h w n -> (n b) c h w")

            # process output
            y = forward_fn(x)
            y = rearrange(y, "(n b) c h w -> b c h w n", n=tile_id_ - tile_id)
            y = y.to(device=tile_device, dtype=tile_dtype)
            model_output_stack.append(y)

        model_output = torch.concat(model_output_stack, dim=-1)
        return model_output

    def io_scale(self, model_output, tile_size):
        # Determine the size modification happened in forward_fn
        # We only consider the same scale on height and width.
        io_scale = model_output.shape[2] / tile_size
        return io_scale

    def untile(self, model_output, height, width, tile_size, tile_stride, border_width, tile_device, tile_dtype):
        # The reversed function of tile
        mask = self.mask(tile_size, tile_size, border_width)
        mask = mask.to(device=tile_device, dtype=tile_dtype)
        mask = rearrange(mask, "h w -> 1 1 h w 1")
        model_output = model_output * mask

        fold_operator = torch.nn.Fold(
            output_size=(height, width), kernel_size=(tile_size, tile_size), stride=(tile_stride, tile_stride)
        )
        mask = repeat(mask[0, 0, :, :, 0], "h w -> 1 (h w) n", n=model_output.shape[-1])
        model_output = rearrange(model_output, "b c h w n -> b (c h w) n")
        model_output = fold_operator(model_output) / fold_operator(mask)

        return model_output

    def tiled_forward(
        self,
        forward_fn,
        model_input,
        tile_size,
        tile_stride,
        tile_batch_size=1,
        tile_device="cpu",
        tile_dtype=torch.float32,
        border_width=None,
    ):
        # Prepare
        inference_device, inference_dtype = model_input.device, model_input.dtype
        height, width = model_input.shape[2], model_input.shape[3]
        border_width = int(tile_stride * 0.5) if border_width is None else border_width

        # tile
        model_input = self.tile(model_input, tile_size, tile_stride, tile_device, tile_dtype)

        # inference
        model_output = self.tiled_inference(
            forward_fn, model_input, tile_batch_size, inference_device, inference_dtype, tile_device, tile_dtype
        )

        # resize
        io_scale = self.io_scale(model_output, tile_size)
        height, width = int(height * io_scale), int(width * io_scale)
        tile_size, tile_stride = int(tile_size * io_scale), int(tile_stride * io_scale)
        border_width = int(border_width * io_scale)

        # untile
        model_output = self.untile(
            model_output, height, width, tile_size, tile_stride, border_width, tile_device, tile_dtype
        )

        # Done!
        model_output = model_output.to(device=inference_device, dtype=inference_dtype)
        return model_output


@contextmanager
def init_weights_on_device(device=torch.device("meta"), include_buffers: bool = False):
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def low_version_attention(query, key, value, attn_bias=None):
    scale = 1 / query.shape[-1] ** 0.5
    query = query * scale
    attn = torch.matmul(query, key.transpose(-2, -1))
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = attn.softmax(-1)
    return attn @ value


class FluxTextEncoder2(T5EncoderModel):
    def __init__(self, config):
        super().__init__(config)
        self.eval()

    def forward(self, input_ids):
        outputs = super().forward(input_ids=input_ids)
        prompt_emb = outputs.last_hidden_state
        return prompt_emb

    @staticmethod
    def state_dict_converter():
        return FluxTextEncoder2StateDictConverter()


class FluxTextEncoder2StateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        state_dict_ = state_dict
        return state_dict_

    def from_civitai(self, state_dict):
        return self.from_diffusers(state_dict)


class Attention(torch.nn.Module):
    def __init__(self, q_dim, num_heads, head_dim, kv_dim=None, bias_q=False, bias_kv=False, bias_out=False):
        super().__init__()
        dim_inner = head_dim * num_heads
        kv_dim = kv_dim if kv_dim is not None else q_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = torch.nn.Linear(q_dim, dim_inner, bias=bias_q)
        self.to_k = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_v = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_out = torch.nn.Linear(dim_inner, q_dim, bias=bias_out)

    def interact_with_ipadapter(self, hidden_states, q, ip_k, ip_v, scale=1.0):
        batch_size = q.shape[0]
        ip_k = ip_k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        ip_v = ip_v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        ip_hidden_states = torch.nn.functional.scaled_dot_product_attention(q, ip_k, ip_v)
        hidden_states = hidden_states + scale * ip_hidden_states
        return hidden_states

    def torch_forward(
        self, hidden_states, encoder_hidden_states=None, attn_mask=None, ipadapter_kwargs=None, qkv_preprocessor=None
    ):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch_size = encoder_hidden_states.shape[0]

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if qkv_preprocessor is not None:
            q, k, v = qkv_preprocessor(q, k, v)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        if ipadapter_kwargs is not None:
            hidden_states = self.interact_with_ipadapter(hidden_states, q, **ipadapter_kwargs)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)

        hidden_states = self.to_out(hidden_states)

        return hidden_states

    def xformers_forward(self, hidden_states, encoder_hidden_states=None, attn_mask=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        q = rearrange(q, "b f (n d) -> (b n) f d", n=self.num_heads)
        k = rearrange(k, "b f (n d) -> (b n) f d", n=self.num_heads)
        v = rearrange(v, "b f (n d) -> (b n) f d", n=self.num_heads)

        if attn_mask is not None:
            hidden_states = low_version_attention(q, k, v, attn_bias=attn_mask)
        else:
            import xformers.ops as xops

            hidden_states = xops.memory_efficient_attention(q, k, v)
        hidden_states = rearrange(hidden_states, "(b n) f d -> b f (n d)", n=self.num_heads)

        hidden_states = hidden_states.to(q.dtype)
        hidden_states = self.to_out(hidden_states)

        return hidden_states

    def forward(
        self, hidden_states, encoder_hidden_states=None, attn_mask=None, ipadapter_kwargs=None, qkv_preprocessor=None
    ):
        return self.torch_forward(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attn_mask=attn_mask,
            ipadapter_kwargs=ipadapter_kwargs,
            qkv_preprocessor=qkv_preprocessor,
        )


class CLIPEncoderLayer(torch.nn.Module):
    def __init__(self, embed_dim, intermediate_size, num_heads=12, head_dim=64, use_quick_gelu=True):
        super().__init__()
        self.attn = Attention(
            q_dim=embed_dim, num_heads=num_heads, head_dim=head_dim, bias_q=True, bias_kv=True, bias_out=True
        )
        self.layer_norm1 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm2 = torch.nn.LayerNorm(embed_dim)
        self.fc1 = torch.nn.Linear(embed_dim, intermediate_size)
        self.fc2 = torch.nn.Linear(intermediate_size, embed_dim)

        self.use_quick_gelu = use_quick_gelu

    def quickGELU(self, x):
        return x * torch.sigmoid(1.702 * x)

    def forward(self, hidden_states, attn_mask=None):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.attn(hidden_states, attn_mask=attn_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.fc1(hidden_states)
        if self.use_quick_gelu:
            hidden_states = self.quickGELU(hidden_states)
        else:
            hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SDTextEncoder(torch.nn.Module):
    def __init__(
        self,
        embed_dim=768,
        vocab_size=49408,
        max_position_embeddings=77,
        num_encoder_layers=12,
        encoder_intermediate_size=3072,
    ):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.position_embeds = torch.nn.Parameter(torch.zeros(1, max_position_embeddings, embed_dim))
        self.encoders = torch.nn.ModuleList(
            [CLIPEncoderLayer(embed_dim, encoder_intermediate_size) for _ in range(num_encoder_layers)]
        )
        self.attn_mask = self.attention_mask(max_position_embeddings)
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)

    def attention_mask(self, length):
        mask = torch.empty(length, length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, input_ids, clip_skip=1):
        embeds = self.token_embedding(input_ids) + self.position_embeds
        attn_mask = self.attn_mask.to(device=embeds.device, dtype=embeds.dtype)
        for encoder_id, encoder in enumerate(self.encoders):
            embeds = encoder(embeds, attn_mask=attn_mask)
            if encoder_id + clip_skip == len(self.encoders):
                break
        embeds = self.final_layer_norm(embeds)
        return embeds


class SD3TextEncoder1(SDTextEncoder):
    def __init__(self, vocab_size=49408):
        super().__init__(vocab_size=vocab_size)

    def forward(self, input_ids, clip_skip=2, extra_mask=None):
        embeds = self.token_embedding(input_ids)
        embeds = embeds + self.position_embeds.to(dtype=embeds.dtype, device=input_ids.device)
        attn_mask = self.attn_mask.to(device=embeds.device, dtype=embeds.dtype)
        if extra_mask is not None:
            attn_mask[:, extra_mask[0] == 0] = float("-inf")
        for encoder_id, encoder in enumerate(self.encoders):
            embeds = encoder(embeds, attn_mask=attn_mask)
            if encoder_id + clip_skip == len(self.encoders):
                hidden_states = embeds
        embeds = self.final_layer_norm(embeds)
        pooled_embeds = embeds[torch.arange(embeds.shape[0]), input_ids.to(dtype=torch.int).argmax(dim=-1)]
        return pooled_embeds, hidden_states


class SD3VAEEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scaling_factor = 1.5305  # Different from SD 1.x
        self.shift_factor = 0.0609  # Different from SD 1.x
        self.conv_in = torch.nn.Conv2d(3, 128, kernel_size=3, padding=1)

        self.blocks = torch.nn.ModuleList(
            [
                # DownEncoderBlock2D
                ResnetBlock(128, 128, eps=1e-6),
                ResnetBlock(128, 128, eps=1e-6),
                DownSampler(128, padding=0, extra_padding=True),
                # DownEncoderBlock2D
                ResnetBlock(128, 256, eps=1e-6),
                ResnetBlock(256, 256, eps=1e-6),
                DownSampler(256, padding=0, extra_padding=True),
                # DownEncoderBlock2D
                ResnetBlock(256, 512, eps=1e-6),
                ResnetBlock(512, 512, eps=1e-6),
                DownSampler(512, padding=0, extra_padding=True),
                # DownEncoderBlock2D
                ResnetBlock(512, 512, eps=1e-6),
                ResnetBlock(512, 512, eps=1e-6),
                # UNetMidBlock2D
                ResnetBlock(512, 512, eps=1e-6),
                VAEAttentionBlock(1, 512, 512, 1, eps=1e-6),
                ResnetBlock(512, 512, eps=1e-6),
            ]
        )

        self.conv_norm_out = torch.nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(512, 32, kernel_size=3, padding=1)

    def tiled_forward(self, sample, tile_size=64, tile_stride=32):
        hidden_states = TileWorker().tiled_forward(
            lambda x: self.forward(x),
            sample,
            tile_size,
            tile_stride,
            tile_device=sample.device,
            tile_dtype=sample.dtype,
        )
        return hidden_states

    def forward(self, sample, tiled=False, tile_size=64, tile_stride=32, **kwargs):
        # For VAE Decoder, we do not need to apply the tiler on each layer.
        if tiled:
            return self.tiled_forward(sample, tile_size=tile_size, tile_stride=tile_stride)

        # 1. pre-process
        hidden_states = self.conv_in(sample)
        time_emb = None
        text_emb = None
        res_stack = None

        # 2. blocks
        for i, block in enumerate(self.blocks):
            hidden_states, time_emb, text_emb, res_stack = block(hidden_states, time_emb, text_emb, res_stack)

        # 3. output
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        hidden_states = hidden_states[:, :16]
        hidden_states = (hidden_states - self.shift_factor) * self.scaling_factor

        return hidden_states

    def encode_video(self, sample, batch_size=8):
        B = sample.shape[0]
        hidden_states = []

        for i in range(0, sample.shape[2], batch_size):
            j = min(i + batch_size, sample.shape[2])
            sample_batch = rearrange(sample[:, :, i:j], "B C T H W -> (B T) C H W")

            hidden_states_batch = self(sample_batch)
            hidden_states_batch = rearrange(hidden_states_batch, "(B T) C H W -> B C T H W", B=B)

            hidden_states.append(hidden_states_batch)

        hidden_states = torch.concat(hidden_states, dim=2)
        return hidden_states

    @staticmethod
    def state_dict_converter():
        return SDVAEEncoderStateDictConverter()


class ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=None, groups=32, eps=1e-5):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels is not None:
            self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = torch.nn.SiLU()
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True
            )

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        x = hidden_states
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        if time_emb is not None:
            emb = self.nonlinearity(time_emb)
            emb = self.time_emb_proj(emb)[:, :, None, None]
            x = x + emb
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        if self.conv_shortcut is not None:
            hidden_states = self.conv_shortcut(hidden_states)
        hidden_states = hidden_states + x
        return hidden_states, time_emb, text_emb, res_stack


class DownSampler(torch.nn.Module):
    def __init__(self, channels, padding=1, extra_padding=False):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, stride=2, padding=padding)
        self.extra_padding = extra_padding

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        if self.extra_padding:
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        hidden_states = self.conv(hidden_states)
        return hidden_states, time_emb, text_emb, res_stack


class VAEAttentionBlock(torch.nn.Module):
    def __init__(
        self, num_attention_heads, attention_head_dim, in_channels, num_layers=1, norm_num_groups=32, eps=1e-5
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=eps, affine=True)

        self.transformer_blocks = torch.nn.ModuleList(
            [
                Attention(inner_dim, num_attention_heads, attention_head_dim, bias_q=True, bias_kv=True, bias_out=True)
                for d in range(num_layers)
            ]
        )

    def forward(self, hidden_states, time_emb, text_emb, res_stack):
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)

        hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        hidden_states = hidden_states + residual

        return hidden_states, time_emb, text_emb, res_stack


class FluxVAEEncoder(SD3VAEEncoder):
    def __init__(self):
        super().__init__()
        self.scaling_factor = 0.3611
        self.shift_factor = 0.1159

    @staticmethod
    def state_dict_converter():
        return FluxVAEEncoderStateDictConverter()


class SDVAEEncoderStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        # architecture
        block_types = [
            "ResnetBlock",
            "ResnetBlock",
            "DownSampler",
            "ResnetBlock",
            "ResnetBlock",
            "DownSampler",
            "ResnetBlock",
            "ResnetBlock",
            "DownSampler",
            "ResnetBlock",
            "ResnetBlock",
            "ResnetBlock",
            "VAEAttentionBlock",
            "ResnetBlock",
        ]

        # Rename each parameter
        local_rename_dict = {
            "quant_conv": "quant_conv",
            "encoder.conv_in": "conv_in",
            "encoder.mid_block.attentions.0.group_norm": "blocks.12.norm",
            "encoder.mid_block.attentions.0.to_q": "blocks.12.transformer_blocks.0.to_q",
            "encoder.mid_block.attentions.0.to_k": "blocks.12.transformer_blocks.0.to_k",
            "encoder.mid_block.attentions.0.to_v": "blocks.12.transformer_blocks.0.to_v",
            "encoder.mid_block.attentions.0.to_out.0": "blocks.12.transformer_blocks.0.to_out",
            "encoder.mid_block.resnets.0.norm1": "blocks.11.norm1",
            "encoder.mid_block.resnets.0.conv1": "blocks.11.conv1",
            "encoder.mid_block.resnets.0.norm2": "blocks.11.norm2",
            "encoder.mid_block.resnets.0.conv2": "blocks.11.conv2",
            "encoder.mid_block.resnets.1.norm1": "blocks.13.norm1",
            "encoder.mid_block.resnets.1.conv1": "blocks.13.conv1",
            "encoder.mid_block.resnets.1.norm2": "blocks.13.norm2",
            "encoder.mid_block.resnets.1.conv2": "blocks.13.conv2",
            "encoder.conv_norm_out": "conv_norm_out",
            "encoder.conv_out": "conv_out",
        }
        name_list = sorted(state_dict)
        rename_dict = {}
        block_id = {"ResnetBlock": -1, "DownSampler": -1, "UpSampler": -1}
        last_block_type_with_id = {"ResnetBlock": "", "DownSampler": "", "UpSampler": ""}
        for name in name_list:
            names = name.split(".")
            name_prefix = ".".join(names[:-1])
            if name_prefix in local_rename_dict:
                rename_dict[name] = local_rename_dict[name_prefix] + "." + names[-1]
            elif name.startswith("encoder.down_blocks"):
                block_type = {"resnets": "ResnetBlock", "downsamplers": "DownSampler", "upsamplers": "UpSampler"}[
                    names[3]
                ]
                block_type_with_id = ".".join(names[:5])
                if block_type_with_id != last_block_type_with_id[block_type]:
                    block_id[block_type] += 1
                last_block_type_with_id[block_type] = block_type_with_id
                while block_id[block_type] < len(block_types) and block_types[block_id[block_type]] != block_type:
                    block_id[block_type] += 1
                block_type_with_id = ".".join(names[:5])
                names = ["blocks", str(block_id[block_type])] + names[5:]
                rename_dict[name] = ".".join(names)

        # Convert state_dict
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
        return state_dict_

    def from_civitai(self, state_dict):
        rename_dict = {
            "first_stage_model.encoder.conv_in.bias": "conv_in.bias",
            "first_stage_model.encoder.conv_in.weight": "conv_in.weight",
            "first_stage_model.encoder.conv_out.bias": "conv_out.bias",
            "first_stage_model.encoder.conv_out.weight": "conv_out.weight",
            "first_stage_model.encoder.down.0.block.0.conv1.bias": "blocks.0.conv1.bias",
            "first_stage_model.encoder.down.0.block.0.conv1.weight": "blocks.0.conv1.weight",
            "first_stage_model.encoder.down.0.block.0.conv2.bias": "blocks.0.conv2.bias",
            "first_stage_model.encoder.down.0.block.0.conv2.weight": "blocks.0.conv2.weight",
            "first_stage_model.encoder.down.0.block.0.norm1.bias": "blocks.0.norm1.bias",
            "first_stage_model.encoder.down.0.block.0.norm1.weight": "blocks.0.norm1.weight",
            "first_stage_model.encoder.down.0.block.0.norm2.bias": "blocks.0.norm2.bias",
            "first_stage_model.encoder.down.0.block.0.norm2.weight": "blocks.0.norm2.weight",
            "first_stage_model.encoder.down.0.block.1.conv1.bias": "blocks.1.conv1.bias",
            "first_stage_model.encoder.down.0.block.1.conv1.weight": "blocks.1.conv1.weight",
            "first_stage_model.encoder.down.0.block.1.conv2.bias": "blocks.1.conv2.bias",
            "first_stage_model.encoder.down.0.block.1.conv2.weight": "blocks.1.conv2.weight",
            "first_stage_model.encoder.down.0.block.1.norm1.bias": "blocks.1.norm1.bias",
            "first_stage_model.encoder.down.0.block.1.norm1.weight": "blocks.1.norm1.weight",
            "first_stage_model.encoder.down.0.block.1.norm2.bias": "blocks.1.norm2.bias",
            "first_stage_model.encoder.down.0.block.1.norm2.weight": "blocks.1.norm2.weight",
            "first_stage_model.encoder.down.0.downsample.conv.bias": "blocks.2.conv.bias",
            "first_stage_model.encoder.down.0.downsample.conv.weight": "blocks.2.conv.weight",
            "first_stage_model.encoder.down.1.block.0.conv1.bias": "blocks.3.conv1.bias",
            "first_stage_model.encoder.down.1.block.0.conv1.weight": "blocks.3.conv1.weight",
            "first_stage_model.encoder.down.1.block.0.conv2.bias": "blocks.3.conv2.bias",
            "first_stage_model.encoder.down.1.block.0.conv2.weight": "blocks.3.conv2.weight",
            "first_stage_model.encoder.down.1.block.0.nin_shortcut.bias": "blocks.3.conv_shortcut.bias",
            "first_stage_model.encoder.down.1.block.0.nin_shortcut.weight": "blocks.3.conv_shortcut.weight",
            "first_stage_model.encoder.down.1.block.0.norm1.bias": "blocks.3.norm1.bias",
            "first_stage_model.encoder.down.1.block.0.norm1.weight": "blocks.3.norm1.weight",
            "first_stage_model.encoder.down.1.block.0.norm2.bias": "blocks.3.norm2.bias",
            "first_stage_model.encoder.down.1.block.0.norm2.weight": "blocks.3.norm2.weight",
            "first_stage_model.encoder.down.1.block.1.conv1.bias": "blocks.4.conv1.bias",
            "first_stage_model.encoder.down.1.block.1.conv1.weight": "blocks.4.conv1.weight",
            "first_stage_model.encoder.down.1.block.1.conv2.bias": "blocks.4.conv2.bias",
            "first_stage_model.encoder.down.1.block.1.conv2.weight": "blocks.4.conv2.weight",
            "first_stage_model.encoder.down.1.block.1.norm1.bias": "blocks.4.norm1.bias",
            "first_stage_model.encoder.down.1.block.1.norm1.weight": "blocks.4.norm1.weight",
            "first_stage_model.encoder.down.1.block.1.norm2.bias": "blocks.4.norm2.bias",
            "first_stage_model.encoder.down.1.block.1.norm2.weight": "blocks.4.norm2.weight",
            "first_stage_model.encoder.down.1.downsample.conv.bias": "blocks.5.conv.bias",
            "first_stage_model.encoder.down.1.downsample.conv.weight": "blocks.5.conv.weight",
            "first_stage_model.encoder.down.2.block.0.conv1.bias": "blocks.6.conv1.bias",
            "first_stage_model.encoder.down.2.block.0.conv1.weight": "blocks.6.conv1.weight",
            "first_stage_model.encoder.down.2.block.0.conv2.bias": "blocks.6.conv2.bias",
            "first_stage_model.encoder.down.2.block.0.conv2.weight": "blocks.6.conv2.weight",
            "first_stage_model.encoder.down.2.block.0.nin_shortcut.bias": "blocks.6.conv_shortcut.bias",
            "first_stage_model.encoder.down.2.block.0.nin_shortcut.weight": "blocks.6.conv_shortcut.weight",
            "first_stage_model.encoder.down.2.block.0.norm1.bias": "blocks.6.norm1.bias",
            "first_stage_model.encoder.down.2.block.0.norm1.weight": "blocks.6.norm1.weight",
            "first_stage_model.encoder.down.2.block.0.norm2.bias": "blocks.6.norm2.bias",
            "first_stage_model.encoder.down.2.block.0.norm2.weight": "blocks.6.norm2.weight",
            "first_stage_model.encoder.down.2.block.1.conv1.bias": "blocks.7.conv1.bias",
            "first_stage_model.encoder.down.2.block.1.conv1.weight": "blocks.7.conv1.weight",
            "first_stage_model.encoder.down.2.block.1.conv2.bias": "blocks.7.conv2.bias",
            "first_stage_model.encoder.down.2.block.1.conv2.weight": "blocks.7.conv2.weight",
            "first_stage_model.encoder.down.2.block.1.norm1.bias": "blocks.7.norm1.bias",
            "first_stage_model.encoder.down.2.block.1.norm1.weight": "blocks.7.norm1.weight",
            "first_stage_model.encoder.down.2.block.1.norm2.bias": "blocks.7.norm2.bias",
            "first_stage_model.encoder.down.2.block.1.norm2.weight": "blocks.7.norm2.weight",
            "first_stage_model.encoder.down.2.downsample.conv.bias": "blocks.8.conv.bias",
            "first_stage_model.encoder.down.2.downsample.conv.weight": "blocks.8.conv.weight",
            "first_stage_model.encoder.down.3.block.0.conv1.bias": "blocks.9.conv1.bias",
            "first_stage_model.encoder.down.3.block.0.conv1.weight": "blocks.9.conv1.weight",
            "first_stage_model.encoder.down.3.block.0.conv2.bias": "blocks.9.conv2.bias",
            "first_stage_model.encoder.down.3.block.0.conv2.weight": "blocks.9.conv2.weight",
            "first_stage_model.encoder.down.3.block.0.norm1.bias": "blocks.9.norm1.bias",
            "first_stage_model.encoder.down.3.block.0.norm1.weight": "blocks.9.norm1.weight",
            "first_stage_model.encoder.down.3.block.0.norm2.bias": "blocks.9.norm2.bias",
            "first_stage_model.encoder.down.3.block.0.norm2.weight": "blocks.9.norm2.weight",
            "first_stage_model.encoder.down.3.block.1.conv1.bias": "blocks.10.conv1.bias",
            "first_stage_model.encoder.down.3.block.1.conv1.weight": "blocks.10.conv1.weight",
            "first_stage_model.encoder.down.3.block.1.conv2.bias": "blocks.10.conv2.bias",
            "first_stage_model.encoder.down.3.block.1.conv2.weight": "blocks.10.conv2.weight",
            "first_stage_model.encoder.down.3.block.1.norm1.bias": "blocks.10.norm1.bias",
            "first_stage_model.encoder.down.3.block.1.norm1.weight": "blocks.10.norm1.weight",
            "first_stage_model.encoder.down.3.block.1.norm2.bias": "blocks.10.norm2.bias",
            "first_stage_model.encoder.down.3.block.1.norm2.weight": "blocks.10.norm2.weight",
            "first_stage_model.encoder.mid.attn_1.k.bias": "blocks.12.transformer_blocks.0.to_k.bias",
            "first_stage_model.encoder.mid.attn_1.k.weight": "blocks.12.transformer_blocks.0.to_k.weight",
            "first_stage_model.encoder.mid.attn_1.norm.bias": "blocks.12.norm.bias",
            "first_stage_model.encoder.mid.attn_1.norm.weight": "blocks.12.norm.weight",
            "first_stage_model.encoder.mid.attn_1.proj_out.bias": "blocks.12.transformer_blocks.0.to_out.bias",
            "first_stage_model.encoder.mid.attn_1.proj_out.weight": "blocks.12.transformer_blocks.0.to_out.weight",
            "first_stage_model.encoder.mid.attn_1.q.bias": "blocks.12.transformer_blocks.0.to_q.bias",
            "first_stage_model.encoder.mid.attn_1.q.weight": "blocks.12.transformer_blocks.0.to_q.weight",
            "first_stage_model.encoder.mid.attn_1.v.bias": "blocks.12.transformer_blocks.0.to_v.bias",
            "first_stage_model.encoder.mid.attn_1.v.weight": "blocks.12.transformer_blocks.0.to_v.weight",
            "first_stage_model.encoder.mid.block_1.conv1.bias": "blocks.11.conv1.bias",
            "first_stage_model.encoder.mid.block_1.conv1.weight": "blocks.11.conv1.weight",
            "first_stage_model.encoder.mid.block_1.conv2.bias": "blocks.11.conv2.bias",
            "first_stage_model.encoder.mid.block_1.conv2.weight": "blocks.11.conv2.weight",
            "first_stage_model.encoder.mid.block_1.norm1.bias": "blocks.11.norm1.bias",
            "first_stage_model.encoder.mid.block_1.norm1.weight": "blocks.11.norm1.weight",
            "first_stage_model.encoder.mid.block_1.norm2.bias": "blocks.11.norm2.bias",
            "first_stage_model.encoder.mid.block_1.norm2.weight": "blocks.11.norm2.weight",
            "first_stage_model.encoder.mid.block_2.conv1.bias": "blocks.13.conv1.bias",
            "first_stage_model.encoder.mid.block_2.conv1.weight": "blocks.13.conv1.weight",
            "first_stage_model.encoder.mid.block_2.conv2.bias": "blocks.13.conv2.bias",
            "first_stage_model.encoder.mid.block_2.conv2.weight": "blocks.13.conv2.weight",
            "first_stage_model.encoder.mid.block_2.norm1.bias": "blocks.13.norm1.bias",
            "first_stage_model.encoder.mid.block_2.norm1.weight": "blocks.13.norm1.weight",
            "first_stage_model.encoder.mid.block_2.norm2.bias": "blocks.13.norm2.bias",
            "first_stage_model.encoder.mid.block_2.norm2.weight": "blocks.13.norm2.weight",
            "first_stage_model.encoder.norm_out.bias": "conv_norm_out.bias",
            "first_stage_model.encoder.norm_out.weight": "conv_norm_out.weight",
            "first_stage_model.quant_conv.bias": "quant_conv.bias",
            "first_stage_model.quant_conv.weight": "quant_conv.weight",
        }
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if "transformer_blocks" in rename_dict[name]:
                    param = param.squeeze()
                state_dict_[rename_dict[name]] = param
        return state_dict_


class FluxVAEEncoderStateDictConverter(SDVAEEncoderStateDictConverter):
    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        rename_dict = {
            "encoder.conv_in.bias": "conv_in.bias",
            "encoder.conv_in.weight": "conv_in.weight",
            "encoder.conv_out.bias": "conv_out.bias",
            "encoder.conv_out.weight": "conv_out.weight",
            "encoder.down.0.block.0.conv1.bias": "blocks.0.conv1.bias",
            "encoder.down.0.block.0.conv1.weight": "blocks.0.conv1.weight",
            "encoder.down.0.block.0.conv2.bias": "blocks.0.conv2.bias",
            "encoder.down.0.block.0.conv2.weight": "blocks.0.conv2.weight",
            "encoder.down.0.block.0.norm1.bias": "blocks.0.norm1.bias",
            "encoder.down.0.block.0.norm1.weight": "blocks.0.norm1.weight",
            "encoder.down.0.block.0.norm2.bias": "blocks.0.norm2.bias",
            "encoder.down.0.block.0.norm2.weight": "blocks.0.norm2.weight",
            "encoder.down.0.block.1.conv1.bias": "blocks.1.conv1.bias",
            "encoder.down.0.block.1.conv1.weight": "blocks.1.conv1.weight",
            "encoder.down.0.block.1.conv2.bias": "blocks.1.conv2.bias",
            "encoder.down.0.block.1.conv2.weight": "blocks.1.conv2.weight",
            "encoder.down.0.block.1.norm1.bias": "blocks.1.norm1.bias",
            "encoder.down.0.block.1.norm1.weight": "blocks.1.norm1.weight",
            "encoder.down.0.block.1.norm2.bias": "blocks.1.norm2.bias",
            "encoder.down.0.block.1.norm2.weight": "blocks.1.norm2.weight",
            "encoder.down.0.downsample.conv.bias": "blocks.2.conv.bias",
            "encoder.down.0.downsample.conv.weight": "blocks.2.conv.weight",
            "encoder.down.1.block.0.conv1.bias": "blocks.3.conv1.bias",
            "encoder.down.1.block.0.conv1.weight": "blocks.3.conv1.weight",
            "encoder.down.1.block.0.conv2.bias": "blocks.3.conv2.bias",
            "encoder.down.1.block.0.conv2.weight": "blocks.3.conv2.weight",
            "encoder.down.1.block.0.nin_shortcut.bias": "blocks.3.conv_shortcut.bias",
            "encoder.down.1.block.0.nin_shortcut.weight": "blocks.3.conv_shortcut.weight",
            "encoder.down.1.block.0.norm1.bias": "blocks.3.norm1.bias",
            "encoder.down.1.block.0.norm1.weight": "blocks.3.norm1.weight",
            "encoder.down.1.block.0.norm2.bias": "blocks.3.norm2.bias",
            "encoder.down.1.block.0.norm2.weight": "blocks.3.norm2.weight",
            "encoder.down.1.block.1.conv1.bias": "blocks.4.conv1.bias",
            "encoder.down.1.block.1.conv1.weight": "blocks.4.conv1.weight",
            "encoder.down.1.block.1.conv2.bias": "blocks.4.conv2.bias",
            "encoder.down.1.block.1.conv2.weight": "blocks.4.conv2.weight",
            "encoder.down.1.block.1.norm1.bias": "blocks.4.norm1.bias",
            "encoder.down.1.block.1.norm1.weight": "blocks.4.norm1.weight",
            "encoder.down.1.block.1.norm2.bias": "blocks.4.norm2.bias",
            "encoder.down.1.block.1.norm2.weight": "blocks.4.norm2.weight",
            "encoder.down.1.downsample.conv.bias": "blocks.5.conv.bias",
            "encoder.down.1.downsample.conv.weight": "blocks.5.conv.weight",
            "encoder.down.2.block.0.conv1.bias": "blocks.6.conv1.bias",
            "encoder.down.2.block.0.conv1.weight": "blocks.6.conv1.weight",
            "encoder.down.2.block.0.conv2.bias": "blocks.6.conv2.bias",
            "encoder.down.2.block.0.conv2.weight": "blocks.6.conv2.weight",
            "encoder.down.2.block.0.nin_shortcut.bias": "blocks.6.conv_shortcut.bias",
            "encoder.down.2.block.0.nin_shortcut.weight": "blocks.6.conv_shortcut.weight",
            "encoder.down.2.block.0.norm1.bias": "blocks.6.norm1.bias",
            "encoder.down.2.block.0.norm1.weight": "blocks.6.norm1.weight",
            "encoder.down.2.block.0.norm2.bias": "blocks.6.norm2.bias",
            "encoder.down.2.block.0.norm2.weight": "blocks.6.norm2.weight",
            "encoder.down.2.block.1.conv1.bias": "blocks.7.conv1.bias",
            "encoder.down.2.block.1.conv1.weight": "blocks.7.conv1.weight",
            "encoder.down.2.block.1.conv2.bias": "blocks.7.conv2.bias",
            "encoder.down.2.block.1.conv2.weight": "blocks.7.conv2.weight",
            "encoder.down.2.block.1.norm1.bias": "blocks.7.norm1.bias",
            "encoder.down.2.block.1.norm1.weight": "blocks.7.norm1.weight",
            "encoder.down.2.block.1.norm2.bias": "blocks.7.norm2.bias",
            "encoder.down.2.block.1.norm2.weight": "blocks.7.norm2.weight",
            "encoder.down.2.downsample.conv.bias": "blocks.8.conv.bias",
            "encoder.down.2.downsample.conv.weight": "blocks.8.conv.weight",
            "encoder.down.3.block.0.conv1.bias": "blocks.9.conv1.bias",
            "encoder.down.3.block.0.conv1.weight": "blocks.9.conv1.weight",
            "encoder.down.3.block.0.conv2.bias": "blocks.9.conv2.bias",
            "encoder.down.3.block.0.conv2.weight": "blocks.9.conv2.weight",
            "encoder.down.3.block.0.norm1.bias": "blocks.9.norm1.bias",
            "encoder.down.3.block.0.norm1.weight": "blocks.9.norm1.weight",
            "encoder.down.3.block.0.norm2.bias": "blocks.9.norm2.bias",
            "encoder.down.3.block.0.norm2.weight": "blocks.9.norm2.weight",
            "encoder.down.3.block.1.conv1.bias": "blocks.10.conv1.bias",
            "encoder.down.3.block.1.conv1.weight": "blocks.10.conv1.weight",
            "encoder.down.3.block.1.conv2.bias": "blocks.10.conv2.bias",
            "encoder.down.3.block.1.conv2.weight": "blocks.10.conv2.weight",
            "encoder.down.3.block.1.norm1.bias": "blocks.10.norm1.bias",
            "encoder.down.3.block.1.norm1.weight": "blocks.10.norm1.weight",
            "encoder.down.3.block.1.norm2.bias": "blocks.10.norm2.bias",
            "encoder.down.3.block.1.norm2.weight": "blocks.10.norm2.weight",
            "encoder.mid.attn_1.k.bias": "blocks.12.transformer_blocks.0.to_k.bias",
            "encoder.mid.attn_1.k.weight": "blocks.12.transformer_blocks.0.to_k.weight",
            "encoder.mid.attn_1.norm.bias": "blocks.12.norm.bias",
            "encoder.mid.attn_1.norm.weight": "blocks.12.norm.weight",
            "encoder.mid.attn_1.proj_out.bias": "blocks.12.transformer_blocks.0.to_out.bias",
            "encoder.mid.attn_1.proj_out.weight": "blocks.12.transformer_blocks.0.to_out.weight",
            "encoder.mid.attn_1.q.bias": "blocks.12.transformer_blocks.0.to_q.bias",
            "encoder.mid.attn_1.q.weight": "blocks.12.transformer_blocks.0.to_q.weight",
            "encoder.mid.attn_1.v.bias": "blocks.12.transformer_blocks.0.to_v.bias",
            "encoder.mid.attn_1.v.weight": "blocks.12.transformer_blocks.0.to_v.weight",
            "encoder.mid.block_1.conv1.bias": "blocks.11.conv1.bias",
            "encoder.mid.block_1.conv1.weight": "blocks.11.conv1.weight",
            "encoder.mid.block_1.conv2.bias": "blocks.11.conv2.bias",
            "encoder.mid.block_1.conv2.weight": "blocks.11.conv2.weight",
            "encoder.mid.block_1.norm1.bias": "blocks.11.norm1.bias",
            "encoder.mid.block_1.norm1.weight": "blocks.11.norm1.weight",
            "encoder.mid.block_1.norm2.bias": "blocks.11.norm2.bias",
            "encoder.mid.block_1.norm2.weight": "blocks.11.norm2.weight",
            "encoder.mid.block_2.conv1.bias": "blocks.13.conv1.bias",
            "encoder.mid.block_2.conv1.weight": "blocks.13.conv1.weight",
            "encoder.mid.block_2.conv2.bias": "blocks.13.conv2.bias",
            "encoder.mid.block_2.conv2.weight": "blocks.13.conv2.weight",
            "encoder.mid.block_2.norm1.bias": "blocks.13.norm1.bias",
            "encoder.mid.block_2.norm1.weight": "blocks.13.norm1.weight",
            "encoder.mid.block_2.norm2.bias": "blocks.13.norm2.bias",
            "encoder.mid.block_2.norm2.weight": "blocks.13.norm2.weight",
            "encoder.norm_out.bias": "conv_norm_out.bias",
            "encoder.norm_out.weight": "conv_norm_out.weight",
        }
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if "transformer_blocks" in rename_dict[name]:
                    param = param.squeeze()
                state_dict_[rename_dict[name]] = param
        return state_dict_


class FluxDiTStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        global_rename_dict = {
            "context_embedder": "context_embedder",
            "x_embedder": "x_embedder",
            "time_text_embed.timestep_embedder.linear_1": "time_embedder.timestep_embedder.0",
            "time_text_embed.timestep_embedder.linear_2": "time_embedder.timestep_embedder.2",
            "time_text_embed.guidance_embedder.linear_1": "guidance_embedder.timestep_embedder.0",
            "time_text_embed.guidance_embedder.linear_2": "guidance_embedder.timestep_embedder.2",
            "time_text_embed.text_embedder.linear_1": "pooled_text_embedder.0",
            "time_text_embed.text_embedder.linear_2": "pooled_text_embedder.2",
            "norm_out.linear": "final_norm_out.linear",
            "proj_out": "final_proj_out",
        }
        rename_dict = {
            "proj_out": "proj_out",
            "norm1.linear": "norm1_a.linear",
            "norm1_context.linear": "norm1_b.linear",
            "attn.to_q": "attn.a_to_q",
            "attn.to_k": "attn.a_to_k",
            "attn.to_v": "attn.a_to_v",
            "attn.to_out.0": "attn.a_to_out",
            "attn.add_q_proj": "attn.b_to_q",
            "attn.add_k_proj": "attn.b_to_k",
            "attn.add_v_proj": "attn.b_to_v",
            "attn.to_add_out": "attn.b_to_out",
            "ff.net.0.proj": "ff_a.0",
            "ff.net.2": "ff_a.2",
            "ff_context.net.0.proj": "ff_b.0",
            "ff_context.net.2": "ff_b.2",
            "attn.norm_q": "attn.norm_q_a",
            "attn.norm_k": "attn.norm_k_a",
            "attn.norm_added_q": "attn.norm_q_b",
            "attn.norm_added_k": "attn.norm_k_b",
        }
        rename_dict_single = {
            "attn.to_q": "a_to_q",
            "attn.to_k": "a_to_k",
            "attn.to_v": "a_to_v",
            "attn.norm_q": "norm_q_a",
            "attn.norm_k": "norm_k_a",
            "norm.linear": "norm.linear",
            "proj_mlp": "proj_in_besides_attn",
            "proj_out": "proj_out",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name.endswith(".weight") or name.endswith(".bias"):
                suffix = ".weight" if name.endswith(".weight") else ".bias"
                prefix = name[: -len(suffix)]
                if prefix in global_rename_dict:
                    state_dict_[global_rename_dict[prefix] + suffix] = param
                elif prefix.startswith("transformer_blocks."):
                    names = prefix.split(".")
                    names[0] = "blocks"
                    middle = ".".join(names[2:])
                    if middle in rename_dict:
                        name_ = ".".join(names[:2] + [rename_dict[middle]] + [suffix[1:]])
                        state_dict_[name_] = param
                elif prefix.startswith("single_transformer_blocks."):
                    names = prefix.split(".")
                    names[0] = "single_blocks"
                    middle = ".".join(names[2:])
                    if middle in rename_dict_single:
                        name_ = ".".join(names[:2] + [rename_dict_single[middle]] + [suffix[1:]])
                        state_dict_[name_] = param
                    else:
                        pass
                else:
                    pass
        for name in list(state_dict_.keys()):
            if "single_blocks." in name and ".a_to_q." in name:
                mlp = state_dict_.get(name.replace(".a_to_q.", ".proj_in_besides_attn."), None)
                if mlp is None:
                    mlp = torch.zeros(
                        4 * state_dict_[name].shape[0], *state_dict_[name].shape[1:], dtype=state_dict_[name].dtype
                    )
                else:
                    state_dict_.pop(name.replace(".a_to_q.", ".proj_in_besides_attn."))
                param = torch.concat(
                    [
                        state_dict_.pop(name),
                        state_dict_.pop(name.replace(".a_to_q.", ".a_to_k.")),
                        state_dict_.pop(name.replace(".a_to_q.", ".a_to_v.")),
                        mlp,
                    ],
                    dim=0,
                )
                name_ = name.replace(".a_to_q.", ".to_qkv_mlp.")
                state_dict_[name_] = param
        for name in list(state_dict_.keys()):
            for component in ["a", "b"]:
                if f".{component}_to_q." in name:
                    name_ = name.replace(f".{component}_to_q.", f".{component}_to_qkv.")
                    param = torch.concat(
                        [
                            state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_q.")],
                            state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_k.")],
                            state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_v.")],
                        ],
                        dim=0,
                    )
                    state_dict_[name_] = param
                    state_dict_.pop(name.replace(f".{component}_to_q.", f".{component}_to_q."))
                    state_dict_.pop(name.replace(f".{component}_to_q.", f".{component}_to_k."))
                    state_dict_.pop(name.replace(f".{component}_to_q.", f".{component}_to_v."))
        return state_dict_

    def from_civitai(self, state_dict):
        rename_dict = {
            "time_in.in_layer.bias": "time_embedder.timestep_embedder.0.bias",
            "time_in.in_layer.weight": "time_embedder.timestep_embedder.0.weight",
            "time_in.out_layer.bias": "time_embedder.timestep_embedder.2.bias",
            "time_in.out_layer.weight": "time_embedder.timestep_embedder.2.weight",
            "txt_in.bias": "context_embedder.bias",
            "txt_in.weight": "context_embedder.weight",
            "vector_in.in_layer.bias": "pooled_text_embedder.0.bias",
            "vector_in.in_layer.weight": "pooled_text_embedder.0.weight",
            "vector_in.out_layer.bias": "pooled_text_embedder.2.bias",
            "vector_in.out_layer.weight": "pooled_text_embedder.2.weight",
            "final_layer.linear.bias": "final_proj_out.bias",
            "final_layer.linear.weight": "final_proj_out.weight",
            "guidance_in.in_layer.bias": "guidance_embedder.timestep_embedder.0.bias",
            "guidance_in.in_layer.weight": "guidance_embedder.timestep_embedder.0.weight",
            "guidance_in.out_layer.bias": "guidance_embedder.timestep_embedder.2.bias",
            "guidance_in.out_layer.weight": "guidance_embedder.timestep_embedder.2.weight",
            "img_in.bias": "x_embedder.bias",
            "img_in.weight": "x_embedder.weight",
            "final_layer.adaLN_modulation.1.weight": "final_norm_out.linear.weight",
            "final_layer.adaLN_modulation.1.bias": "final_norm_out.linear.bias",
        }
        suffix_rename_dict = {
            "img_attn.norm.key_norm.scale": "attn.norm_k_a.weight",
            "img_attn.norm.query_norm.scale": "attn.norm_q_a.weight",
            "img_attn.proj.bias": "attn.a_to_out.bias",
            "img_attn.proj.weight": "attn.a_to_out.weight",
            "img_attn.qkv.bias": "attn.a_to_qkv.bias",
            "img_attn.qkv.weight": "attn.a_to_qkv.weight",
            "img_mlp.0.bias": "ff_a.0.bias",
            "img_mlp.0.weight": "ff_a.0.weight",
            "img_mlp.2.bias": "ff_a.2.bias",
            "img_mlp.2.weight": "ff_a.2.weight",
            "img_mod.lin.bias": "norm1_a.linear.bias",
            "img_mod.lin.weight": "norm1_a.linear.weight",
            "txt_attn.norm.key_norm.scale": "attn.norm_k_b.weight",
            "txt_attn.norm.query_norm.scale": "attn.norm_q_b.weight",
            "txt_attn.proj.bias": "attn.b_to_out.bias",
            "txt_attn.proj.weight": "attn.b_to_out.weight",
            "txt_attn.qkv.bias": "attn.b_to_qkv.bias",
            "txt_attn.qkv.weight": "attn.b_to_qkv.weight",
            "txt_mlp.0.bias": "ff_b.0.bias",
            "txt_mlp.0.weight": "ff_b.0.weight",
            "txt_mlp.2.bias": "ff_b.2.bias",
            "txt_mlp.2.weight": "ff_b.2.weight",
            "txt_mod.lin.bias": "norm1_b.linear.bias",
            "txt_mod.lin.weight": "norm1_b.linear.weight",
            "linear1.bias": "to_qkv_mlp.bias",
            "linear1.weight": "to_qkv_mlp.weight",
            "linear2.bias": "proj_out.bias",
            "linear2.weight": "proj_out.weight",
            "modulation.lin.bias": "norm.linear.bias",
            "modulation.lin.weight": "norm.linear.weight",
            "norm.key_norm.scale": "norm_k_a.weight",
            "norm.query_norm.scale": "norm_q_a.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name.startswith("model.diffusion_model."):
                name = name[len("model.diffusion_model.") :]
            names = name.split(".")
            if name in rename_dict:
                rename = rename_dict[name]
                if name.startswith("final_layer.adaLN_modulation.1."):
                    param = torch.concat([param[3072:], param[:3072]], dim=0)
                state_dict_[rename] = param
            elif names[0] == "double_blocks":
                rename = f"blocks.{names[1]}." + suffix_rename_dict[".".join(names[2:])]
                state_dict_[rename] = param
            elif names[0] == "single_blocks":
                if ".".join(names[2:]) in suffix_rename_dict:
                    rename = f"single_blocks.{names[1]}." + suffix_rename_dict[".".join(names[2:])]
                    state_dict_[rename] = param
            else:
                pass
        if "guidance_embedder.timestep_embedder.0.weight" not in state_dict_:
            return state_dict_, {"disable_guidance_embedder": True}
        elif "blocks.8.attn.norm_k_a.weight" not in state_dict_:
            return state_dict_, {"input_dim": 196, "num_blocks": 8}
        else:
            return state_dict_
