from typing import Dict

import torch
from transformers.modeling_outputs import ModelOutput

from ..distributed.parallel_state import get_parallel_state
from ..distributed.sequence_parallel import gather_outputs
from ..utils.seqlen_pos_transform_utils import culen2len, pos2culen


class Postforward:
    def __init__(self, rmpad_with_pos_ids: bool = False):
        self.postforward_pipeline = []

        self.compute_seqlens_func = SeqlensComputePostForward()
        if rmpad_with_pos_ids:
            self.postforward_pipeline.append(PackingPostForward())
        else:
            self.postforward_pipeline.append(PaddingPostForward())

    def __call__(self, outputs: ModelOutput, micro_batch: Dict[str, torch.Tensor]):
        seq_lens = self.compute_seqlens_func(micro_batch)
        for postforward_func in self.postforward_pipeline:
            outputs = postforward_func(outputs, seq_lens)
        return outputs


class SeqlensComputePostForward:
    def __call__(self, micro_batch: Dict[str, torch.Tensor]):
        seq_lens = culen2len(pos2culen(micro_batch["position_ids"])).tolist()
        # seq_lens = 1 is sp_pad
        seq_lens = [s for s in seq_lens if s != 1]
        return seq_lens


class PackingPostForward:
    def __call__(self, outputs: ModelOutput, seq_lens):
        logits = outputs.logits
        if get_parallel_state().sp_enabled:
            logits = gather_outputs(logits, gather_dim=0, group=get_parallel_state().sp_group)
            logits = logits[: sum(seq_lens)]  # remove sp padding
        logits_list = logits.split(seq_lens, dim=0)
        outputs.logits = logits_list
        return outputs


class PaddingPostForward:
    def __call__(self, outputs: ModelOutput, seq_lens):
        logits = outputs.logits
        if logits.dim() != 3:  # logits is calculated in ce, so it is flattened_seqlen, dim
            dim = logits.shape[-1]
            bs = len(seq_lens)
            logits = logits.view(bs, -1, dim)

        if get_parallel_state().sp_enabled:
            logits = gather_outputs(logits, gather_dim=1, group=get_parallel_state().sp_group)

        logits_list = logits.unbind(dim=0)
        logits_list = [item[:seq_len] for item, seq_len in zip(logits_list, seq_lens)]
        outputs.logits = logits_list
        return outputs
