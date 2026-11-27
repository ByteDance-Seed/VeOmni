"""Accelerated wrapper for the Qwen3-Omni audio codec.

The codec is never trained: Qwen3-Omni fits the talker to a fixed codec, and
the talker-training round uses this module the way Janus uses its VQVAE — as a
frozen tokenizer that turns the assistant's reference speech into the code
targets the talker is scored against.

So the one training hook here is ``encode``, and it carries no loss and no FSDP
gradient anchor. A dummy forward would be pointless for a module with nothing
to anchor, which is why a batch holding no assistant audio simply does nothing.
"""

from typing import Any, Dict, List, Optional

import torch
from torch.distributed.tensor import DTensor

from .......data.seed_omni.utils.media_metadata import AUDIO_METADATA_KEY
from .....mixins.base_mixin import BaseMixin
from .....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from .....utils.conversation import ConversationItem, iter_desired_items
from ..modeling import TALKER_CODE_LABELS, Qwen3OmniCodec


class TrainingMixin(TrainingModuleMixin):
    """Training-graph hook: assistant speech -> RVQ code targets on the carrier."""

    config: Any
    device: torch.device

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: Any = None
        # None until the first encode walks the tree; `()` is a real answer.
        self._unsharded_mimi_units: Optional[tuple] = None

    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[List[List[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        waveforms, rates = [], []
        for item in self._target_items(conversation_list):
            waveforms.append(item.value)
            rates.append(self._rate_of(item))
        return {"waveforms": waveforms, "sampling_rates": rates}

    def encode(self, **kwargs: Any) -> Dict[str, Any]:
        """All-gather the units inside Mimi that nothing else will, encode, give the memory back.

        Done here rather than in ``encode_pre`` because a ``pre_forward`` hook
        runs *outside* the FSDP-wrapped call: at that point not even this
        module's own root unit has unsharded, so Mimi's convolutional encoder —
        in no ``_no_split_modules`` entry, hence part of that root unit — is
        still sharded. Inside the endpoint body both halves are readable, which
        is the same reason the audio tower casts its mel features in ``forward``
        rather than in its pre-hook.
        """
        self._unshard_mimi_units()
        try:
            return super().encode(**kwargs)
        finally:
            self._reshard_mimi_units()

    def _unshard_mimi_units(self) -> None:
        """All-gather every FSDP2 unit under Mimi, once per encode.

        ``MimiResidualVectorQuantizer`` is one of this codec's
        ``_no_split_modules``, so each of the two instances becomes its own
        FSDP2 unit — three in all under ``self.mimi``, counting the enclosing
        ``MimiSplitResidualVectorQuantizer`` that HF ``post_init`` unions in
        from upstream's own list, which is one more reason the walk below is
        deliberately a superset. Every path to one goes through a *method*, never
        ``__call__``: ``MimiModel._encode_frame`` calls
        ``self.quantizer.encode(...)``, which calls
        ``self.semantic_residual_vector_quantizer.encode(...)`` and the acoustic
        one. FSDP2 installs its pre-forward hooks on ``__call__`` only, so those
        parameters stayed sharded and the quantizer's ``input_proj`` conv
        reported "mixed torch.Tensor and DTensor" from inside upstream code,
        naming neither this module nor the cause. Handling it here keeps the
        codec shardable under either ``fsdp_scope``, with no ``fsdp_mode: ddp``
        pin and no fp32 replica per rank.

        **Why not ``register_fsdp_forward_method``**, which is upstream's
        sanctioned answer for a unit entered by method: ``Qwen3OmniCodec.encode``
        loops over the batch's assistant clips, one ``mimi.encode`` each, so
        registering makes each RVQ unit all-gather and reshard *per clip* — a
        collective count of ``2 x clips_on_this_rank``. That is a rank-dependent
        number of collectives on a module that runs as a graph root, before the
        talker's cooperative agreement all-reduce can report a mismatch, so an
        uneven batch would hang every peer until the NCCL watchdog instead of
        raising. One window around the whole loop is two collectives whatever
        the data does, which is also fewer.

        A superset on purpose: the transformer stacks under Mimi are entered
        through ``__call__`` (``self.encoder_transformer(...)``) and would
        unshard themselves. Telling them apart would mean encoding upstream's
        call graph here, and holding a few sub-megabyte layers gathered for one
        encode buys nothing — whereas a unit missed by a narrower walk is a
        crash. Every rank walks the same tree and gets the same units in the
        same order, so the all-gathers cannot diverge.
        """
        from torch.distributed.fsdp import FSDPModule

        if self._unsharded_mimi_units is None:
            # Cached including the empty result: under DDP and eager inference
            # there is no FSDPModule under self.mimi to find.
            self._unsharded_mimi_units = tuple(m for m in self.mimi.modules() if isinstance(m, FSDPModule))
        for unit in self._unsharded_mimi_units:
            # Keep every unit unsharded across the whole per-clip encode loop:
            # MimiTransformerLayer is invoked via __call__ once per clip, so with
            # reshard_after_forward=True clip 2 would trigger a new all-gather and
            # deadlock when ranks hold different numbers of assistant clips.
            unit.set_reshard_after_forward(False)
            unit.unshard()

        sharded = next((name for name, param in self.mimi.named_parameters() if isinstance(param, DTensor)), None)
        if sharded is not None:
            raise RuntimeError(
                f"qwen3omni_codec: {sharded} is still a DTensor after unsharding every FSDP2 unit under "
                f"`self.mimi`, so Mimi holds sharded parameters belonging to no unit this walk can reach. "
                f"Encoding would fail inside upstream code with a bare 'mixed torch.Tensor and DTensor'. "
                f"This raises identically on every rank (it reads the module tree, not the batch), so it "
                f"cannot desync them."
            )

    def _reshard_mimi_units(self) -> None:
        """Give the memory back. ``reshard()`` issues no collective, so the
        ``finally`` path is safe even when ``encode`` raises on one rank only."""
        for unit in self._unsharded_mimi_units or ():
            unit.reshard()
            unit.set_reshard_after_forward(True)

    @post_forward("encode")
    def encode_post(self, **outputs: Any) -> Dict[str, Any]:
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        items = self._target_items(conversation)
        for item, codes in zip(items, outputs["codes"], strict=True):
            item.meta[TALKER_CODE_LABELS] = codes
        return {"conversation_list": conversation}

    @staticmethod
    def _target_items(conversation_list: Any) -> List[ConversationItem]:
        """The assistant's reference speech — the only audio this module tokenizes.

        Role is what separates the two audio streams in an omni conversation and
        it also keeps their sample rates apart: the user's speech is an *input*,
        already turned into 16kHz mel features by the audio tower's
        preprocessor, while the assistant's is a *target* the codec reads as a
        24kHz waveform. Selecting by role means neither ever reaches the wrong
        consumer at the wrong rate.
        """
        if conversation_list is None:
            return []
        return [
            item
            for item in iter_desired_items(conversation_list, types=["audio"], roles=["assistant"])
            if not item.is_dummy
        ]

    def _rate_of(self, item: ConversationItem) -> int:
        """The clip's declared rate, required rather than assumed.

        ``encode_wav`` rejects anything but 24kHz, so the data layer has to load
        assistant audio at the codec's rate. Reading a declared rate here (and
        refusing to invent one) is what turns a corpus loaded at 16 or 22.05kHz
        into an error at step 0 instead of a talker quietly trained against
        targets whose timeline runs fast.
        """
        rate = getattr(item.meta.get(AUDIO_METADATA_KEY), "sampling_rate", None)
        if rate is None:
            raise ValueError(
                f"qwen3omni_codec: an assistant audio item states no meta['{AUDIO_METADATA_KEY}'] rate. The data "
                f"layer must state it — this codec only accepts {self.config.sampling_rate} Hz and cannot check "
                f"silently."
            )
        return int(rate)


class VeOmniMixin(BaseMixin, TrainingMixin):
    """``generate()`` already lives on the native :class:`Qwen3OmniCodec`."""


class Qwen3OmniCodecAccelerated(VeOmniMixin, Qwen3OmniCodec):
    pass


__all__ = ["Qwen3OmniCodecAccelerated", "TrainingMixin", "VeOmniMixin"]
