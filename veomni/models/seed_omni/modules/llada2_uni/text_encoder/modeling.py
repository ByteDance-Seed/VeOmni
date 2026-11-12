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

"""LLaDA2.0-Uni word-token embedding + untied LM head + block-diffusion FSM."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import torch

from veomni.models.loss_utils.dllm import dllm_masked_cross_entropy
from veomni.utils.constants import IGNORE_INDEX

from ....graphs.generation_graph import FSM_SIGNAL_KEY
from ....utils.conversation import ConversationItem
from ...text_encoder.modeling import TextEncoder
from .configuration import LladaUniTextEncoderConfig
from .processing import LladaUniTextEncoderPreprocessor


SIGNAL_TEXT_DONE = "text_done"
SIGNAL_VQ_DONE = "vq_done"
SIGNAL_THINK_DONE = "think_done"


class LladaUniTextEncoder(TextEncoder):
    """Unified-vocab embedding + untied LM head + block-diffusion generation for LLaDA2.0-Uni."""

    config_class = LladaUniTextEncoderConfig
    preprocessor_class = LladaUniTextEncoderPreprocessor
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    UNDERSTAND_SYSTEM = "You are a multimodal understanding assistant."
    T2I_SYSTEM = "You are a text-to-image generation assistant."
    THINK_SYSTEM = "You are a text-to-image generation assistant with a thinking process."
    EDIT_SYSTEM = "You are an image editing assistant."
    SIGNAL_VQ_DONE = SIGNAL_VQ_DONE
    SIGNAL_THINK_DONE = SIGNAL_THINK_DONE

    def __init__(self, config: LladaUniTextEncoderConfig):
        super().__init__(config)
        self._eos_token_id: Optional[int] = None
        self._reset_diffusion_state()

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._eos_token_id = (
            int(tokenizer.eos_token_id) if getattr(tokenizer, "eos_token_id", None) is not None else None
        )

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def decode(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_weight: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        loss: Optional[torch.Tensor] = None
        logits: Optional[torch.Tensor] = None
        if labels is not None and loss_weight is not None:
            selected = labels != IGNORE_INDEX
            if bool(selected.any()):
                sel_logits = self._project(hidden_states[selected])
                weighted_sum = dllm_masked_cross_entropy(
                    sel_logits,
                    labels[selected],
                    loss_weight[selected],
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
            else:
                weighted_sum = self._project(hidden_states[:1]).sum() * 0.0
            loss = weighted_sum / max(int(num_samples), 1)
        else:
            logits = self._project(hidden_states)
        return {"loss": loss, "logits": logits}

    # ── Block-diffusion inference state & FSM ──────────────────────────────────

    def _reset_diffusion_state(self) -> None:
        self._diff_x: Optional[torch.Tensor] = None
        self._diff_prompt_len: int = 0
        self._diff_gen_len: int = 0
        self._diff_block_length: int = 0
        self._diff_num_blocks: int = 0
        self._diff_steps_per_block: int = 0
        self._diff_block_idx: int = 0
        self._diff_step_in_block: int = 0
        self._diff_total_length: int = 0
        self._diff_num_transfer: List[int] = []
        self._diff_attn: Optional[torch.Tensor] = None
        self._diff_done: bool = False
        self._t2i_conv: Any = None
        self._t2i_uncond: Optional[torch.Tensor] = None
        self._t2i_pad_len: int = 0
        self._t2i_pos: Optional[torch.Tensor] = None
        self._t2i_cfg_scale: float = 4.0
        self._t2i_cfg_rescale: float = 0.7
        self._t2i_grid: tuple = (0, 0)
        self._t2i_n: int = 0
        self._edit_ut: Optional[torch.Tensor] = None
        self._edit_ui: Optional[torch.Tensor] = None
        self._edit_pad_t: int = 0
        self._edit_pad_i: int = 0
        self._edit_cfg_text: float = 4.0
        self._edit_cfg_image: float = 0.0
        self._think_active: bool = False
        self._think_prefix: Optional[torch.Tensor] = None
        self._think_text: Optional[str] = None
        self._think_grid: Optional[tuple] = None

    def reset_local_inference_state(self) -> None:
        super().reset_local_inference_state()
        self._reset_diffusion_state()

    @staticmethod
    def _make_num_transfer_schedule(gen_len: int, steps: int) -> List[int]:
        base, rem = divmod(gen_len, steps)
        return [base + 1 if i < rem else base for i in range(steps)]

    def _build_block_causal_mask(self, total_length: int, block_length: int) -> torch.Tensor:
        device = self.device
        blk = torch.arange(total_length, device=device) // block_length
        allow = blk.unsqueeze(0) <= blk.unsqueeze(1)
        return allow.unsqueeze(0).unsqueeze(0)

    def _i2t_emit(self) -> Dict[str, Any]:
        bl = self._diff_block_length
        cwe = (self._diff_block_idx + 1) * bl
        embed_device = self.get_input_embeddings().weight.device
        embeds = self._embed_tokens(self._diff_x[:cwe].to(embed_device).unsqueeze(0)).squeeze(0)
        return {"llada_inputs_embeds": embeds, "llada_attention_mask": self._diff_attn[:, :, :cwe, :cwe]}

    def _build_prompt_ids(self, conversation_list: Optional[List[ConversationItem]]) -> torch.Tensor:
        tok = self._tokenizer
        device = self.device
        conv = conversation_list or []
        image_item = next((it for it in conv if it.type == "image" and not it.is_dummy), None)
        question = " ".join(
            str(it.value) for it in conv if it.type == "text" and it.role in ("user", "human", None)
        ).strip()

        def t(s: str) -> List[int]:
            return tok(s, add_special_tokens=False)["input_ids"]

        ids: List[int] = t(f"<role>SYSTEM</role> {self.UNDERSTAND_SYSTEM} <role>HUMAN</role>")
        if image_item is not None:
            vq = image_item.meta.get("llada2_uni_vq_ids")
            grid = image_item.meta.get("llada2_uni_grid")
            if vq is None or grid is None:
                raise RuntimeError(
                    "LladaUniTextEncoder: image item missing VQ ids — the image_tokenizer must run first."
                )
            h, w = int(grid[0]), int(grid[1])
            ids += t("<|image|>") + t(f"<|reserved_token_{h}|>") + t(f"<|reserved_token_{w}|>") + t("<boi>")
            ids += vq.to(torch.long).reshape(-1).tolist()
            ids += t("<|/image|>")
        ids += t("\n")
        if question:
            ids += t(question)
        ids += t("<role>ASSISTANT</role>")
        return torch.tensor(ids, dtype=torch.long, device=device)

    def _diffusion_init(
        self,
        conversation_list: Optional[List[ConversationItem]],
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        gk = generation_kwargs or {}
        block_length = int(gk.get("block_length", 32))
        gen_length = int(gk.get("gen_length", 256))
        steps = int(gk.get("diffusion_steps", gk.get("steps", 32)))

        prompt_ids = self._build_prompt_ids(conversation_list).to(self.device)
        prompt_len = int(prompt_ids.shape[0])
        mask_id = int(self.config.mask_token_id)
        num_blocks = (prompt_len + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length
        x = torch.full((total_length,), mask_id, dtype=torch.long, device=self.device)
        x[:prompt_len] = prompt_ids

        self._diff_x = x
        self._diff_prompt_len = prompt_len
        self._diff_gen_len = total_length - prompt_len
        self._diff_block_length = block_length
        self._diff_num_blocks = num_blocks
        self._diff_total_length = total_length
        self._diff_steps_per_block = max(1, steps)
        self._diff_block_idx = prompt_len // block_length
        self._diff_step_in_block = 0
        self._diff_num_transfer = self._make_num_transfer_schedule(block_length, max(1, steps))
        self._diff_attn = self._build_block_causal_mask(total_length, block_length)
        self._diff_done = False
        return self._i2t_emit()

    def _diffusion_step(
        self,
        llada_hidden: Optional[torch.Tensor],
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if llada_hidden is None:
            raise RuntimeError("llada2_uni_text_encoder.generate expects `llada_hidden` from the backbone after init.")
        hidden = llada_hidden.squeeze(0) if llada_hidden.dim() == 3 else llada_hidden

        bl = self._diff_block_length
        b = self._diff_block_idx
        start, end = b * bl, (b + 1) * bl
        logits = self._project(hidden[-bl:]).float()
        dev = logits.device
        mask_id = int(self.config.mask_token_id)
        xb = self._diff_x[start:end].to(dev)
        active = xb == mask_id

        x0 = logits.argmax(dim=-1)
        probs = torch.softmax(logits, dim=-1)
        x0_p = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        neg_inf = torch.finfo(x0_p.dtype).min
        confidence = torch.where(active, x0_p, torch.full_like(x0_p, neg_inf))

        num_to_transfer = int(self._diff_num_transfer[self._diff_step_in_block])
        threshold = float((generation_kwargs or {}).get("threshold", 0.95))
        high_conf = (confidence > threshold) & active
        if num_to_transfer > 0 and int(high_conf.sum().item()) >= num_to_transfer:
            transfer = high_conf
        else:
            transfer = torch.zeros_like(active)
            kk = min(num_to_transfer, int(active.sum().item()))
            if kk > 0:
                idx = torch.topk(confidence, kk).indices
                transfer[idx] = True
        if bool(transfer.any()):
            xb = xb.clone()
            xb[transfer] = x0[transfer].to(xb.dtype)
            self._diff_x[start:end] = xb.to(self._diff_x.device)

        self._diff_step_in_block += 1
        block_done = (
            self._diff_step_in_block >= self._diff_steps_per_block
            or int((self._diff_x[start:end] == mask_id).sum().item()) == 0
        )
        if block_done:
            self._diff_block_idx += 1
            self._diff_step_in_block = 0

        eos_id = self._eos_token_id if self._eos_token_id is not None else 156892
        gen = self._diff_x[self._diff_prompt_len :]
        eos_pos = (gen == eos_id).nonzero(as_tuple=True)[0]
        finished = self._diff_block_idx >= self._diff_num_blocks
        if len(eos_pos) > 0 and bool((gen[: eos_pos[0]] != mask_id).all()):
            finished = True
        if self._think_active and not finished:
            boi_id = int(self.config.boi_token_id)
            boi_pos = (gen == boi_id).nonzero(as_tuple=True)[0]
            if len(boi_pos) > 0 and bool((gen[: boi_pos[0]] != mask_id).all()):
                finished = True
        if finished:
            if self._think_active:
                return self._think_finish()
            return self._finish_diffusion()
        return self._i2t_emit()

    def _finish_diffusion(self) -> Dict[str, Any]:
        eos_id = self._eos_token_id if self._eos_token_id is not None else 156892
        gen_ids = self._diff_x[self._diff_prompt_len :].tolist()
        if eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id)]
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        self._diff_done = True
        return {
            FSM_SIGNAL_KEY: SIGNAL_TEXT_DONE,
            "generated": {"type": "text", "value": text, "meta": {"token_ids": gen_ids}},
        }

    def finalize(self, *, ctx: Dict[str, Any]) -> Dict[str, Any]:
        del ctx
        if self._diff_x is None or self._diff_done:
            return {}
        return {"generated": self._finish_diffusion()["generated"]}

    # ── Thinking-mode T2I & CFG VQ generation (T2I / Edit) ─────────────────────

    def _think_prompt_ids(self, conversation_list, gk) -> torch.Tensor:
        tok = self._tokenizer
        prompt = (gk or {}).get("prompt")
        if prompt is None:
            prompt = " ".join(
                str(it.value)
                for it in (conversation_list or [])
                if it.type == "text" and it.role in ("user", "human", None)
            ).strip()

        def t(s: str) -> List[int]:
            return tok(s, add_special_tokens=False)["input_ids"]

        ids = (
            t(f"<role>SYSTEM</role> {self.THINK_SYSTEM} <role>HUMAN</role>") + t(prompt) + t("<role>ASSISTANT</role>")
        )
        return torch.tensor(ids, dtype=torch.long)

    def _think_init(self, conversation_list, generation_kwargs) -> Dict[str, Any]:
        gk = generation_kwargs or {}
        block_length = int(gk.get("block_length", 32))
        gen_length = int(gk.get("thinking_gen_length", 4096))
        steps = int(gk.get("thinking_steps", gk.get("diffusion_steps", 32)))
        mask_id = int(self.config.mask_token_id)

        prompt_ids = self._think_prompt_ids(conversation_list, gk).to(self.device)
        prompt_len = int(prompt_ids.shape[0])
        num_blocks = (prompt_len + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length
        x = torch.full((total_length,), mask_id, dtype=torch.long, device=self.device)
        x[:prompt_len] = prompt_ids

        self._diff_x = x
        self._diff_prompt_len = prompt_len
        self._diff_gen_len = total_length - prompt_len
        self._diff_block_length = block_length
        self._diff_num_blocks = num_blocks
        self._diff_total_length = total_length
        self._diff_steps_per_block = max(1, steps)
        self._diff_block_idx = prompt_len // block_length
        self._diff_step_in_block = 0
        self._diff_num_transfer = self._make_num_transfer_schedule(block_length, max(1, steps))
        self._diff_attn = self._build_block_causal_mask(total_length, block_length)
        self._diff_done = False
        self._think_active = True
        return self._i2t_emit()

    def _parse_header_grid(self, seq: torch.Tensor, boi_pos: int):
        if boi_pos < 2:
            return None

        def _to_int(tid: int):
            m = re.search(r"reserved_token_(\d+)", self._tokenizer.decode([int(tid)]))
            return int(m.group(1)) if m else None

        gh = _to_int(seq[boi_pos - 2].item())
        gw = _to_int(seq[boi_pos - 1].item())
        return (gh, gw) if (gh and gw) else None

    def _think_step(self, llada_hidden, generation_kwargs) -> Dict[str, Any]:
        return self._diffusion_step(llada_hidden, generation_kwargs)

    def _think_finish(self) -> Dict[str, Any]:
        boi_id = int(self.config.boi_token_id)
        seq = self._diff_x
        boi_pos_t = (seq == boi_id).nonzero(as_tuple=True)[0]
        if len(boi_pos_t) == 0:
            raise RuntimeError(
                "llada2_uni thinking-T2I: the thinking trace produced no <boi> token — "
                "increase thinking_gen_length or adjust the prompt."
            )
        boi_pos = int(boi_pos_t[0].item())
        self._think_prefix = seq[: boi_pos + 1].detach().clone()
        thinking_ids = seq[self._diff_prompt_len : boi_pos].tolist()
        self._think_text = self._tokenizer.decode(thinking_ids, skip_special_tokens=True)
        self._think_active = False
        self._think_grid = self._parse_header_grid(seq, boi_pos)
        self._diff_x = None
        return {
            FSM_SIGNAL_KEY: self.SIGNAL_THINK_DONE,
            "generated": {"type": "text", "value": self._think_text},
        }

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        llada_hidden: Optional[torch.Tensor] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        gk = generation_kwargs or {}
        task = gk.get("task")
        infer_type = gk.get("infer_type")
        if task is None:
            if infer_type in ("infer_t2i", "infer_gen"):
                task = "t2i"
            elif infer_type == "infer_t2i_thinking":
                task = "t2i"
                gk = {**gk, "mode": "thinking"}
            elif infer_type == "infer_edit":
                task = "edit"

        if task == "t2i":
            if gk.get("mode") == "thinking" and self._think_prefix is None:
                if self._diff_x is None:
                    self._t2i_conv = conversation_list
                    return self._think_init(conversation_list, gk)
                return self._think_step(llada_hidden, gk)
            if self._diff_x is None:
                self._t2i_conv = conversation_list
                return self._t2i_init(conversation_list, gk)
            return self._t2i_step(llada_hidden, gk)
        if task == "edit":
            if self._diff_x is None:
                self._t2i_conv = conversation_list
                return self._edit_init(conversation_list, gk)
            return self._edit_step(llada_hidden, gk)

        if self._diff_x is None:
            return self._diffusion_init(conversation_list, gk)
        return self._diffusion_step(llada_hidden, gk)

    def _t2i_prompts(self, conversation_list, gk):
        tok = self._tokenizer
        gk = gk or {}
        prompt = gk.get("prompt")
        if prompt is None:
            prompt = " ".join(
                str(it.value)
                for it in (conversation_list or [])
                if it.type == "text" and it.role in ("user", "human", None)
            ).strip()
        gh = (int(gk.get("image_h", 1024)) // 2) // 16
        gw = (int(gk.get("image_w", 1024)) // 2) // 16
        n = gh * gw

        def t(s: str) -> List[int]:
            return tok(s, add_special_tokens=False)["input_ids"]

        header = t("<|image|>") + t(f"<|reserved_token_{gh}|>") + t(f"<|reserved_token_{gw}|>") + t("<boi>")
        if self._think_prefix is not None:
            if self._think_grid is not None:
                gh, gw = self._think_grid
                n = gh * gw
                header = t("<|image|>") + t(f"<|reserved_token_{gh}|>") + t(f"<|reserved_token_{gw}|>") + t("<boi>")
            sys_ids = t(f"<role>SYSTEM</role> {self.THINK_SYSTEM} <role>HUMAN</role>")
            asst_ids = t("<role>ASSISTANT</role>")
            cond = self._think_prefix.tolist()
            unc = sys_ids + t("<uncondition>") + asst_ids + header
            return cond, unc, gh, gw, n
        sys_ids = t(f"<role>SYSTEM</role> {self.T2I_SYSTEM} <role>HUMAN</role>")
        asst_ids = t("<role>ASSISTANT</role>")
        cond = sys_ids + t(prompt) + asst_ids + header
        unc = sys_ids + t("<uncondition>") + asst_ids + header
        return cond, unc, gh, gw, n

    def _t2i_init(self, conversation_list, generation_kwargs):
        gk = generation_kwargs or {}
        block_length = int(gk.get("block_length", 32))
        steps = int(gk.get("diffusion_steps", gk.get("steps", 32)))
        device = self.device
        mask_id = int(self.config.mask_token_id)

        cond_l, unc_l, gh, gw, n = self._t2i_prompts(conversation_list, gk)
        cond = torch.tensor(cond_l, dtype=torch.long, device=device)
        unc = torch.tensor(unc_l, dtype=torch.long, device=device)
        prompt_len = int(cond.shape[0])
        num_blocks = (prompt_len + n + block_length - 1) // block_length
        total = num_blocks * block_length

        x = torch.full((total,), mask_id, dtype=torch.long, device=device)
        x[:prompt_len] = cond
        pad_len = max(0, prompt_len - int(unc.shape[0]))
        unc_padded = torch.full((prompt_len,), mask_id, dtype=torch.long, device=device)
        if pad_len >= 0 and unc.shape[0] <= prompt_len:
            unc_padded[prompt_len - unc.shape[0] :] = unc
        else:
            unc_padded[:] = unc[-prompt_len:]
            pad_len = 0

        self._diff_x = x
        self._diff_prompt_len = prompt_len
        self._diff_gen_len = n
        self._diff_block_length = block_length
        self._diff_num_blocks = num_blocks
        self._diff_total_length = total
        self._diff_block_idx = prompt_len // block_length
        self._diff_step_in_block = 0
        self._diff_steps_per_block = max(1, steps)
        self._diff_num_transfer = self._make_num_transfer_schedule(block_length, max(1, steps))
        self._t2i_uncond = unc_padded
        self._t2i_pad_len = pad_len
        self._t2i_cfg_scale = float(gk.get("cfg_scale", 4.0))
        self._t2i_cfg_rescale = float(gk.get("cfg_rescale", 0.7))
        self._t2i_grid = (gh, gw)
        self._t2i_n = n
        self._diff_attn = self._t2i_build_masks(total, block_length, pad_len)
        self._t2i_pos = self._t2i_build_pos(total, pad_len)
        self._diff_done = False
        return self._t2i_emit()

    def _t2i_build_masks(self, total: int, block_length: int, pad_len: int) -> torch.Tensor:
        blk = torch.arange(total, device=self.device) // block_length
        allow = blk.unsqueeze(0) <= blk.unsqueeze(1)
        cond_m = allow.clone()
        unc_m = allow.clone()
        if pad_len > 0:
            unc_m[:, :pad_len] = False
        return torch.stack([cond_m, unc_m]).unsqueeze(1)

    def _t2i_build_pos(self, total: int, pad_len: int) -> torch.Tensor:
        cond_pos = torch.arange(total, device=self.device)
        unc_pos = torch.cat(
            [
                torch.zeros(pad_len, dtype=torch.long, device=self.device),
                torch.arange(total - pad_len, device=self.device),
            ]
        )
        return torch.stack([cond_pos, unc_pos])

    def _t2i_emit(self) -> Dict[str, Any]:
        bl = self._diff_block_length
        cwe = (self._diff_block_idx + 1) * bl
        embed_device = self.get_input_embeddings().weight.device
        cond_x = self._diff_x[:cwe]
        uncond_x = cond_x.clone()
        uncond_x[: self._diff_prompt_len] = self._t2i_uncond
        batch = torch.stack([cond_x, uncond_x]).to(embed_device)
        embeds = self._embed_tokens(batch)
        return {
            "llada_inputs_embeds": embeds,
            "llada_attention_mask": self._diff_attn[:, :, :cwe, :cwe],
            "llada_position_ids": self._t2i_pos[:, :cwe],
        }

    def _vq_image_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits[:, : int(self.config.image_token_offset)] = float("-inf")
        return logits

    def _vq_unmask(self, active: torch.Tensor, generation_kwargs) -> bool:
        bl = self._diff_block_length
        b = self._diff_block_idx
        start, end = b * bl, (b + 1) * bl
        active = active.to(self._diff_x.device)
        mask_id = int(self.config.mask_token_id)
        xb = self._diff_x[start:end]
        active_mask = xb == mask_id
        x0 = active.argmax(dim=-1)
        probs = torch.softmax(active, dim=-1)
        x0_p = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        neg_inf = torch.finfo(x0_p.dtype).min
        confidence = torch.where(active_mask, x0_p, torch.full_like(x0_p, neg_inf))

        num_to_transfer = int(self._diff_num_transfer[self._diff_step_in_block])
        threshold = float((generation_kwargs or {}).get("threshold", 0.95))
        high = (confidence > threshold) & active_mask
        if num_to_transfer > 0 and int(high.sum().item()) >= num_to_transfer:
            transfer = high
        else:
            transfer = torch.zeros_like(active_mask)
            kk = min(num_to_transfer, int(active_mask.sum().item()))
            if kk > 0:
                transfer[torch.topk(confidence, kk).indices] = True
        if bool(transfer.any()):
            xb = xb.clone()
            xb[transfer] = x0[transfer].to(xb.dtype)
            self._diff_x[start:end] = xb

        self._diff_step_in_block += 1
        block_done = (
            self._diff_step_in_block >= self._diff_steps_per_block
            or int((self._diff_x[start:end] == mask_id).sum().item()) == 0
        )
        if block_done:
            self._diff_block_idx += 1
            self._diff_step_in_block = 0

        img = self._diff_x[self._diff_prompt_len : self._diff_prompt_len + self._t2i_n]
        return int((img == mask_id).sum().item()) == 0 or self._diff_block_idx >= self._diff_num_blocks

    def _t2i_step(self, llada_hidden, generation_kwargs):
        if llada_hidden is None or llada_hidden.dim() != 3 or llada_hidden.size(0) != 2:
            raise RuntimeError("llada2_uni T2I expects batch-2 (cond/uncond) hidden states from the backbone.")
        bl = self._diff_block_length
        logits = self._project(llada_hidden[:, -bl:, :]).float()
        cond_l, unc_l = logits[0], logits[1]

        active = unc_l + self._t2i_cfg_scale * (cond_l - unc_l)
        if self._t2i_cfg_rescale > 0:
            std_cond = cond_l.std(dim=-1, keepdim=True)
            std_cfg = active.std(dim=-1, keepdim=True)
            rescaled = active * (std_cond / (std_cfg + 1e-6))
            active = self._t2i_cfg_rescale * rescaled + (1.0 - self._t2i_cfg_rescale) * active

        active = self._vq_image_logits(active)
        if self._vq_unmask(active, generation_kwargs):
            return self._t2i_finish()
        return self._t2i_emit()

    def _t2i_finish(self) -> Dict[str, Any]:
        img = self._diff_x[self._diff_prompt_len : self._diff_prompt_len + self._t2i_n].detach().to("cpu")
        self._diff_done = True
        conv = self._t2i_conv
        if conv is not None:
            conv.append(
                ConversationItem(
                    type="image",
                    value=None,
                    role="assistant",
                    meta={"llada2_uni_gen_vq_ids": img, "llada2_uni_gen_grid": self._t2i_grid},
                )
            )
        return {FSM_SIGNAL_KEY: self.SIGNAL_VQ_DONE, "conversation_list": conv}

    def _edit_prompts(self, conversation_list, gk):
        tok = self._tokenizer
        gk = gk or {}
        instruction = gk.get("prompt")
        if instruction is None:
            instruction = " ".join(
                str(it.value)
                for it in (conversation_list or [])
                if it.type == "text" and it.role in ("user", "human", None)
            ).strip()
        image_item = next((it for it in (conversation_list or []) if it.type == "image" and not it.is_dummy), None)
        if (
            image_item is None
            or not isinstance(image_item.meta, dict)
            or image_item.meta.get("llada2_uni_vq_ids") is None
        ):
            raise RuntimeError(
                "llada2_uni edit: source image VQ ids missing — the image_tokenizer must run first "
                "(expected meta['llada2_uni_vq_ids'] + meta['llada2_uni_grid'])."
            )
        src_vq = image_item.meta["llada2_uni_vq_ids"].to(torch.long).reshape(-1).tolist()
        grid = image_item.meta["llada2_uni_grid"]
        gh, gw = int(grid[0]), int(grid[1])
        n = gh * gw

        def t(s: str) -> List[int]:
            return tok(s, add_special_tokens=False)["input_ids"]

        header = t("<|image|>") + t(f"<|reserved_token_{gh}|>") + t(f"<|reserved_token_{gw}|>") + t("<boi>")
        soi = t("<|image|>")
        eoi = t("<|/image|>")
        sys_ids = t(f"<role>SYSTEM</role> {self.EDIT_SYSTEM} <role>HUMAN</role>")
        asst_ids = t("<role>ASSISTANT</role>")
        instr_ids = t(instruction)
        src_image = header + src_vq + eoi
        cond = sys_ids + src_image + instr_ids + asst_ids + header
        uncond_text = sys_ids + src_image + t("<uncondition>") + asst_ids + header
        uncond_img = sys_ids + soi + instr_ids + asst_ids + header
        return cond, uncond_text, uncond_img, gh, gw, n

    def _front_pad(self, token_list, prompt_len, mask_id):
        device = self.device
        t = torch.tensor(token_list, dtype=torch.long, device=device)
        length = int(t.shape[0])
        if length <= prompt_len:
            out = torch.full((prompt_len,), mask_id, dtype=torch.long, device=device)
            out[prompt_len - length :] = t
            return out, prompt_len - length
        return t[-prompt_len:].clone(), 0

    def _edit_init(self, conversation_list, generation_kwargs):
        gk = generation_kwargs or {}
        block_length = int(gk.get("block_length", 32))
        steps = int(gk.get("diffusion_steps", gk.get("steps", 8)))
        device = self.device
        mask_id = int(self.config.mask_token_id)

        cond_l, ut_l, ui_l, gh, gw, n = self._edit_prompts(conversation_list, gk)
        cond = torch.tensor(cond_l, dtype=torch.long, device=device)
        prompt_len = int(cond.shape[0])
        num_blocks = (prompt_len + n + block_length - 1) // block_length
        total = num_blocks * block_length

        x = torch.full((total,), mask_id, dtype=torch.long, device=device)
        x[:prompt_len] = cond
        ut_padded, pad_t = self._front_pad(ut_l, prompt_len, mask_id)
        ui_padded, pad_i = self._front_pad(ui_l, prompt_len, mask_id)

        self._diff_x = x
        self._diff_prompt_len = prompt_len
        self._diff_gen_len = n
        self._diff_block_length = block_length
        self._diff_num_blocks = num_blocks
        self._diff_total_length = total
        self._diff_block_idx = prompt_len // block_length
        self._diff_step_in_block = 0
        self._diff_steps_per_block = max(1, steps)
        self._diff_num_transfer = self._make_num_transfer_schedule(block_length, max(1, steps))
        self._edit_ut = ut_padded
        self._edit_ui = ui_padded
        self._edit_pad_t = pad_t
        self._edit_pad_i = pad_i
        self._edit_cfg_text = float(gk.get("cfg_text_scale", 4.0))
        self._edit_cfg_image = float(gk.get("cfg_image_scale", 0.0))
        self._t2i_cfg_rescale = float(gk.get("cfg_rescale", 0.7))
        self._t2i_grid = (gh, gw)
        self._t2i_n = n
        self._diff_attn = self._edit_build_masks(total, block_length, pad_t, pad_i)
        self._t2i_pos = self._edit_build_pos(total, pad_t, pad_i)
        self._diff_done = False
        return self._edit_emit()

    def _edit_build_masks(self, total: int, block_length: int, pad_t: int, pad_i: int) -> torch.Tensor:
        blk = torch.arange(total, device=self.device) // block_length
        allow = blk.unsqueeze(0) <= blk.unsqueeze(1)
        cond_m = allow.clone()
        ut_m = allow.clone()
        if pad_t > 0:
            ut_m[:, :pad_t] = False
        ui_m = allow.clone()
        if pad_i > 0:
            ui_m[:, :pad_i] = False
        return torch.stack([cond_m, ut_m, ui_m]).unsqueeze(1)

    def _edit_build_pos(self, total: int, pad_t: int, pad_i: int) -> torch.Tensor:
        def pp(pad: int) -> torch.Tensor:
            return torch.cat(
                [
                    torch.zeros(pad, dtype=torch.long, device=self.device),
                    torch.arange(total - pad, device=self.device),
                ]
            )

        return torch.stack([torch.arange(total, device=self.device), pp(pad_t), pp(pad_i)])

    def _edit_emit(self) -> Dict[str, Any]:
        bl = self._diff_block_length
        cwe = (self._diff_block_idx + 1) * bl
        embed_device = self.get_input_embeddings().weight.device
        cond_x = self._diff_x[:cwe]
        ut_x = cond_x.clone()
        ut_x[: self._diff_prompt_len] = self._edit_ut
        ui_x = cond_x.clone()
        ui_x[: self._diff_prompt_len] = self._edit_ui
        batch = torch.stack([cond_x, ut_x, ui_x]).to(embed_device)
        embeds = self._embed_tokens(batch)
        return {
            "llada_inputs_embeds": embeds,
            "llada_attention_mask": self._diff_attn[:, :, :cwe, :cwe],
            "llada_position_ids": self._t2i_pos[:, :cwe],
        }

    def _edit_step(self, llada_hidden, generation_kwargs):
        if llada_hidden is None or llada_hidden.dim() != 3 or llada_hidden.size(0) != 3:
            raise RuntimeError(
                "llada2_uni edit expects batch-3 (full/no_text/no_img) hidden states from the backbone."
            )
        bl = self._diff_block_length
        logits = self._project(llada_hidden[:, -bl:, :]).float()
        full_l, no_text_l, no_img_l = logits[0], logits[1], logits[2]

        active = no_text_l + self._edit_cfg_text * (full_l - no_text_l) + self._edit_cfg_image * (no_text_l - no_img_l)
        if self._t2i_cfg_rescale > 0:
            std_cond = full_l.std(dim=-1, keepdim=True)
            std_cfg = active.std(dim=-1, keepdim=True)
            rescaled = active * (std_cond / (std_cfg + 1e-6))
            active = self._t2i_cfg_rescale * rescaled + (1.0 - self._t2i_cfg_rescale) * active

        active = self._vq_image_logits(active)
        if self._vq_unmask(active, generation_kwargs):
            return self._t2i_finish()
        return self._edit_emit()


__all__ = [
    "LladaUniTextEncoder",
    "SIGNAL_TEXT_DONE",
    "SIGNAL_THINK_DONE",
    "SIGNAL_VQ_DONE",
]
