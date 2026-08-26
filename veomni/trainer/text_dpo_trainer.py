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

import copy
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..arguments import ModelRuntimeArguments, VeOmniArguments
from ..data import build_data_transform
from ..data.data_collator import PostCollator
from ..distributed.parallel_state import get_parallel_state, use_parallel_state
from ..distributed.sequence_parallel import gather_outputs
from ..distributed.torch_compile import mark_compile_step_begin
from ..models.model_runtime import VeOmniModelRuntime
from ..ops.batch_invariant_ops import set_batch_invariant_mode
from ..utils import helper, logging
from ..utils.constants import IGNORE_INDEX
from ..utils.device import synchronize
from .base import BaseTrainer, VeOmniIter


logger = logging.get_logger(__name__)

_NON_MODEL_KEYS = set()


def _build_dpo_labels_list(
    all_labels: torch.Tensor,
    seq_lens: List[int],
    sp_enabled: bool,
) -> List[torch.Tensor]:
    """Split the packed label tensor into per-segment mask targets.

    DPO packs each preference pair as two adjacent segments
    ``[chosen | rejected]``; multiple pairs are packed together. Aligning
    the mask to per-token log-probs depends on whether SP is active:

    * SP off — ``SequenceParallelCollator`` does not run; labels are
      *unshifted*. Apply the causal shift per segment and pad the trailing
      slot with ``IGNORE_INDEX``.
    * SP on — ``SequenceParallelCollator`` already applied a single global
      shift across the packed sequence, so naive slicing leaves each
      segment's tail label holding the *next* segment's head token
      (chosen tail = rejected head). That position would leak the wrong
      target into ``loss_mask`` and pull cross-segment content into the
      chosen / rejected log-prob sums. Force the trailing slot of every
      segment to ``IGNORE_INDEX`` to mask the boundary.

    Args:
        all_labels: flat ``[sum(seq_lens)]`` label tensor, already gathered
            back from SP ranks by the caller when ``sp_enabled``.
        seq_lens: per-segment token counts, in order.
        sp_enabled: whether the SP path applied a global shift upstream.

    Returns:
        A list of per-segment label tensors, each of length ``seq_lens[i]``,
        with the segment boundary masked to ``IGNORE_INDEX``.
    """
    labels_list: List[torch.Tensor] = []
    offset = 0
    if sp_enabled:
        for sl in seq_lens:
            seg = all_labels[offset : offset + sl].clone()
            seg[-1] = IGNORE_INDEX
            labels_list.append(seg)
            offset += sl
    else:
        for sl in seq_lens:
            seq_labels = all_labels[offset : offset + sl]
            labels_list.append(F.pad(seq_labels[1:], (0, 1), value=IGNORE_INDEX))
            offset += sl
    return labels_list


# ================================ DPO Arguments ======================================


@dataclass
class DPOConfig:
    """dpo.* — Direct Preference Optimization hyperparameters."""

    beta: float = field(
        default=0.1,
        metadata={"help": "Temperature parameter for the DPO loss. Controls deviation from the reference model."},
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "Label smoothing for DPO loss. Non-zero values assume noisy preference labels."},
    )
    reference_free: bool = field(
        default=False,
        metadata={"help": "If True, ignore the reference model and use an implicit uniform reference."},
    )
    loss_type: Literal["sigmoid", "ipo"] = field(
        default="sigmoid",
        metadata={"help": "DPO loss variant: 'sigmoid' for standard DPO, 'ipo' for Identity Preference Optimization."},
    )
    average_log_prob: bool = field(
        default=False,
        metadata={"help": "If True, average log probs per token instead of summing."},
    )
    refer_model_precision: Literal["float32", "bfloat16"] = field(
        default="bfloat16",
        metadata={"help": "Precision of the reference model."},
    )


@dataclass
class VeOmniDPOArguments(VeOmniArguments):
    """Root config for DPO training — extends VeOmniArguments with DPO hyperparameters."""

    dpo_config: DPOConfig = field(default_factory=DPOConfig)
    reference_model: Optional[ModelRuntimeArguments] = field(
        default=None,
        metadata={
            "help": (
                "Model-level args for the frozen DPO reference. "
                "Omit to reuse `model`. This is a full config, not a partial overlay."
            )
        },
    )


class DPOReferenceModelRuntime(VeOmniModelRuntime):
    """Frozen DPO reference: same model build as the policy, then eval.

    Construction takes this model's *own* arguments, so the reference can
    load a different checkpoint or accelerator than the policy. Frozen-eval
    knobs (no LoRA / AMP / recompute / compile) are applied here, not by
    rewriting the caller's config.

    Init only builds the module. Optimizer, assets, and checkpoint stay
    uncalled — callbacks only ever fan out to the policy.
    """

    def __init__(
        self,
        args: ModelRuntimeArguments,
        model_name: str = "reference",
        *,
        train,
        torch_dtype: str = "bfloat16",
    ):
        args = copy.deepcopy(args)
        args.lora_config = {}
        args.accelerator.fsdp_config.mixed_precision.enable = False
        args.accelerator.gradient_checkpointing.enable = False
        args.accelerator.torch_compile.enable = False
        self.args = args
        self.model_name = model_name
        self.train = train
        self.chat_template_name = None
        self._torch_dtype = torch_dtype
        self.setup()
        with use_parallel_state(self.model_name):
            self.build_model()
            self.model.requires_grad_(False)
            self.build_parallelized_model()
            self.model.eval()

    @property
    def skip_hf_weight_load(self) -> bool:
        """A policy resume does not carry reference weights; always materialize HF."""
        return False

    def build_model(self) -> None:
        from ..models.auto import build_foundation_model

        args = self.args
        logger.info_rank0("Build DPO reference model")
        self.model = build_foundation_model(
            config_path=args.config_path,
            weights_path=args.model_path,
            torch_dtype=self._torch_dtype,
            init_device=args.accelerator.init_device,
            ops_implementation=args.ops_implementation,
            config_kwargs=args.model_config,
        )
        self.model_config = self.model.config


class TextDPOTrainer:
    """Text DPO trainer that composes BaseTrainer with DPO-specific logic."""

    base: BaseTrainer
    policy_model: VeOmniModelRuntime
    reference_model: DPOReferenceModelRuntime

    def __init__(self, args: VeOmniDPOArguments):
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        self.base.device = self.base.setup_distributed(args)  # registers ParallelState("base") before seed
        self.policy_model = self.build_policy_model_runtime()

        self._build_data_transform()

        self.base._build_dataset()
        self.base._build_collate_fn()
        self.base._build_dataloader()
        self._build_postforward()
        self.policy_model.build_lr_scheduler(args.train_steps * args.train.num_train_epochs)
        self.base._build_training_context()
        self.base._init_callbacks(self)

        self.reference_model = self.build_reference_model_runtime()

    @property
    def model(self):
        """Callbacks bind to this trainer and read ``.model`` — that is the policy."""
        return self.policy_model

    def __getattr__(self, name):
        return getattr(self.base, name)

    def load(self) -> None:
        self.policy_model.load()

    def save_dcp(self, state) -> None:
        self.policy_model.save_dcp(state)

    def save_hf_or_lora(self, state, stage: str = "step_end") -> None:
        self.policy_model.save_hf_or_lora(state, stage=stage)

    # ── Trainer build functions ────────────────────────────────

    def _build_data_transform(self):
        args: VeOmniDPOArguments = self.base.args
        self.base.data_transform = build_data_transform(
            "dpo",
            tokenizer=self.policy_model.tokenizer,
            chat_template=self.policy_model.chat_template,
            max_seq_len=args.data.max_seq_len,
        )

    def _build_postforward(self):
        self.post_forward = PostCollator()

    def build_policy_model_runtime(self) -> VeOmniModelRuntime:
        """Build the trainable policy under ParallelState ``"policy"``."""
        return self.base.build_model_runtime(model_name="policy")

    def build_reference_model_runtime(self) -> DPOReferenceModelRuntime:
        """Build the frozen reference as its own runtime.

        ``reference_model`` is a full model-level config when set; otherwise
        the policy's ``model`` is reused.
        """
        args: VeOmniDPOArguments = self.base.args
        return DPOReferenceModelRuntime(
            args.reference_model or args.model,
            "reference",
            train=args.train,
            torch_dtype=args.dpo_config.refer_model_precision,
        )

    def on_step_begin(self, micro_batches=None):
        # Each DPO preference pair is packed as two consecutive causal-LM
        # segments (chosen, rejected) but carries one source metadata entry.
        self.base.on_step_begin(micro_batches=micro_batches, source_repeat=2)

    # ── Trainer train step functions ────────────────────────────────

    @staticmethod
    def dpo_loss(
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        beta: float,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
        reference_free: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the DPO/IPO loss for a batch of policy and reference model log probabilities.

        Returns:
            (losses, chosen_rewards, rejected_rewards) -- each of shape (batch_size,).
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        if loss_type == "ipo":
            losses = (logits - 1 / (2 * beta)) ** 2
        else:
            losses = (
                -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
            )

        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(self, model: nn.Module, micro_batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single forward pass on the packed batch containing chosen+rejected pairs.

        Each DPO sample contributes two consecutive sequences (chosen
        then rejected) to the packed sequence. Activates VeOmni's
        chunked CE log-probs path by passing ``return_log_probs=True``
        to the model — the wrapper installed by
        ``build_foundation_model`` promotes per-token NLL into
        ``output.fused_linear_aux.log_probs`` (actual log-probabilities,
        non-positive) without materializing the ``[B, L, V]`` logits
        tensor that the previous gather-on-logits path OOMed on at long
        context.
        This is the same entry point external integrators (verl) and
        the future PPO trainer use. Even-indexed sequences are
        chosen; odd are rejected.

        Caller must enter ``use_parallel_state`` for *this* model's
        mesh so forward and SP gathers resolve the correct groups.

        Returns:
            (chosen_logps, rejected_logps) each of shape ``(B,)``.
        """
        model_inputs = {k: v for k, v in micro_batch.items() if k not in _NON_MODEL_KEYS}
        outputs = model(**model_inputs, return_log_probs=True, use_cache=False)

        # ``outputs.fused_linear_aux.log_probs`` is shape [1, packed_L]
        # (actual log-probabilities; sign already flipped). PostCollator
        # only knows about ``outputs.logits``, so we replicate its
        # SP-gather + per-seq split inline against the log_probs field.
        # Caller must enter this model's ``use_parallel_state`` so forward and
        # SP gathers resolve the correct groups.
        log_probs_packed = outputs.fused_linear_aux.log_probs.squeeze(0)  # [packed_L]
        seq_lens = self.post_forward.compute_seqlens_func(micro_batch)
        sp_enabled = get_parallel_state().sp_enabled
        if sp_enabled:
            sp_group = get_parallel_state().sp_group
            log_probs_packed = gather_outputs(log_probs_packed, gather_dim=0, group=sp_group)
            log_probs_packed = log_probs_packed[: sum(seq_lens)]
        log_probs_list = list(log_probs_packed.split(seq_lens, dim=0))

        # Build per-segment label targets aligned to the kernel's per-token
        # log-probs. ``_build_dpo_labels_list`` handles both SP-on (segment
        # boundary masking against the global shift) and SP-off (per-segment
        # causal shift with IGNORE_INDEX trailing pad) — see helper docstring.
        if sp_enabled:
            all_labels = gather_outputs(micro_batch["labels"], gather_dim=-1, group=sp_group)
            all_labels = all_labels.view(-1)[: sum(seq_lens)]
        else:
            all_labels = micro_batch["labels"].view(-1)
        labels_list = _build_dpo_labels_list(all_labels, seq_lens, sp_enabled)

        average_log_prob = getattr(self.base.args, "dpo_config", None) and self.base.args.dpo_config.average_log_prob
        all_logps: List[torch.Tensor] = []
        for seq_log_probs, seq_labels in zip(log_probs_list, labels_list):
            loss_mask = seq_labels != IGNORE_INDEX
            per_token_logps = seq_log_probs.float()  # already true log p; no negation
            if average_log_prob:
                logp = (per_token_logps * loss_mask).sum() / loss_mask.sum().clamp(min=1)
            else:
                logp = (per_token_logps * loss_mask).sum()
            all_logps.append(logp)

        all_logps_t = torch.stack(all_logps)
        return all_logps_t[0::2], all_logps_t[1::2]

    def forward_backward_step(
        self, micro_batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        channel_loss_callback = getattr(self, "channel_loss_callback", None)
        micro_step_context = (
            channel_loss_callback.micro_step_context(self.state, micro_batch)
            if channel_loss_callback is not None
            else nullcontext()
        )
        with micro_step_context:
            args: VeOmniDPOArguments = self.base.args
            dpo_config = args.dpo_config

            micro_batch = self.base.preforward(micro_batch)
            if channel_loss_callback is not None:
                channel_loss_callback.strip_model_inputs(micro_batch)

            reference_name = getattr(self.reference_model, "model_name", "reference")
            with torch.no_grad(), use_parallel_state(reference_name):
                ref_chosen_logps, ref_rejected_logps = self.concatenated_forward(self.reference_model, micro_batch)

            channel_forward_context = (
                channel_loss_callback.model_forward_context() if channel_loss_callback is not None else nullcontext()
            )
            policy_name = getattr(self.policy_model, "model_name", "policy")
            with (
                use_parallel_state(policy_name),
                self.base.model_fwd_context,
                set_batch_invariant_mode(args.train.enable_batch_invariant_mode),
                channel_forward_context,
            ):
                policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy_model, micro_batch)

            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=dpo_config.beta,
                label_smoothing=dpo_config.label_smoothing,
                loss_type=dpo_config.loss_type,
                reference_free=dpo_config.reference_free,
            )

            loss = losses.mean()

            reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()
            loss_dict: Dict[str, torch.Tensor] = {
                "dpo_loss": loss.detach(),
                "chosen_rewards": chosen_rewards.mean().detach(),
                "rejected_rewards": rejected_rewards.mean().detach(),
                "reward_accuracy": reward_accuracies.detach(),
                "reward_margin": (chosen_rewards - rejected_rewards).mean().detach(),
            }

            with (
                use_parallel_state(policy_name),
                self.base.model_bwd_context,
                set_batch_invariant_mode(args.train.enable_batch_invariant_mode),
            ):
                loss.backward()

            del micro_batch
            return loss, loss_dict

    def train_step(self, data_iterator: Any) -> Dict[str, float]:
        self.state.global_step += 1

        micro_batches: List[Dict[str, Any]] = next(data_iterator)

        self.on_step_begin(micro_batches=micro_batches)

        self.base.sync_before_train_step()

        total_loss = 0.0
        total_loss_dict: Dict[str, float] = defaultdict(float)

        num_micro_steps = len(micro_batches)
        for micro_step, micro_batch in enumerate(micro_batches):
            mark_compile_step_begin(getattr(self.policy_model, "_veomni_compile_uses_cuda_graphs", False))
            self.base.model_reshard(micro_step, num_micro_steps, self.policy_model)
            self.base._configure_hsdp_allreduce(micro_step, num_micro_steps, self.policy_model)
            loss, loss_dict = self.forward_backward_step(micro_batch)

            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_loss_dict[k] += v.item()

        grad_norm = self.policy_model.clip_grad_norm()

        self.policy_model.optimizer.step()
        self.policy_model.lr_scheduler.step()
        self.policy_model.optimizer.zero_grad()

        self.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm)

    # ── Trainer train loop ────────────────────────────────

    def train(self):
        args: VeOmniDPOArguments = self.base.args
        self.on_train_begin()
        logger.info(
            f"Rank{args.train.local_rank} Start DPO training. "
            f"Start step: {self.start_step}. "
            f"Train steps: {args.train_steps}. "
            f"Start epoch: {self.start_epoch}. "
            f"Train epochs: {args.train.num_train_epochs}."
        )

        for epoch in range(self.start_epoch, args.train.num_train_epochs):
            if hasattr(self.base.train_dataloader, "set_epoch"):
                self.base.train_dataloader.set_epoch(epoch)
            self.state.epoch = epoch

            self.on_epoch_begin()

            # Create a batch generator
            self.base.data_iterator = VeOmniIter(
                self.base.train_dataloader, use_background_prefetcher=args.data.dataloader.use_background_prefetcher
            )

            for _ in range(self.start_step, args.train_steps):
                try:
                    self.train_step(self.base.data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.dataloader.drop_last}")
                    break

            self.on_epoch_end()

            self.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
            if args.data.dataloader.use_background_prefetcher:
                self.base.data_iterator.stop()

        self.on_train_end()

        if args.data.dataloader.use_background_prefetcher:
            self.base.data_iterator.stop()

        synchronize()

        self.base.destroy_distributed()
