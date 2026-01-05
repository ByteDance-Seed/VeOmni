from transformers.modeling_outputs import ModelOutput

from ..distributed.parallel_state import get_parallel_state
from ..distributed.sequence_parallel import gather_outputs


class Postforward:
    def __init__(self, rmpad_with_pos_ids: bool = False):
        self.postforward_pipeline = []

        if rmpad_with_pos_ids:
            self.postforward_pipeline.append(PackingPostForward())
        else:
            self.postforward_pipeline.append(PaddingPostForward())

        if get_parallel_state().sp_enabled:
            self.postforward_pipeline.append(SequenceParallelPostForward())

    def __call__(self, outputs: ModelOutput, seq_lens):
        for postforward_func in self.postforward_pipeline:
            outputs = postforward_func(outputs, seq_lens)
        return outputs


class PackingPostForward:
    def __call__(self, outputs: ModelOutput, *args, **kwargs):
        return outputs


class PaddingPostForward:
    def __call__(self, outputs: ModelOutput, seq_lens):
        logits = outputs.logits
        if logits.dim() == 3:  # bs, seqlen, dim
            return outputs
        else:  # flattened_seqlen, dim
            dim = logits.shape[-1]
            bs = len(seq_lens)
            logits = logits.view(bs, -1, dim)
            outputs.logits = logits
            return outputs


class SequenceParallelPostForward:
    def __call__(self, outputs: ModelOutput, seq_lens):
        logits_list = []
        logits = outputs.logits
        if logits.dim() == 3:  # padding logits
            logits = gather_outputs(logits, gather_dim=1, group=get_parallel_state().sp_group)
            logits_list = logits.unbind(dim=0)
            logits_list = [item[:seq_len] for item, seq_len in zip(logits_list, seq_lens)]
        else:  # packing logits
            logits = gather_outputs(logits, gather_dim=0, group=get_parallel_state().sp_group)
            logits = logits[: sum(seq_lens)]
            logits_list = logits.split(seq_lens, dim=0)
        outputs.logits = logits_list
        return outputs
