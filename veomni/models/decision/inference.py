"""Prediction using the same processor and decoder as decision training."""

from contextlib import nullcontext

import torch

from veomni.data.data_collator import MainCollator
from veomni.data.decision.collator import DecisionCollator, encode_request
from veomni.distributed.parallel_state import use_parallel_state

from .base import get_decision_model


@torch.no_grad()
def predict_request(model, preprocessor, request, *, decision_model="clef", max_request_tokens=16384, collate_fn=None):
    sample = encode_request(request, preprocessor=preprocessor, max_request_tokens=max_request_tokens)
    was_training = model.training
    model.eval()
    state = getattr(model, "parallel_state", None)
    try:
        with use_parallel_state(state) if state is not None else nullcontext():
            if collate_fn is None:
                collate_fn = DecisionCollator(
                    feature_collator=MainCollator(
                        metadata_collate_func=getattr(model, "get_metadata_collate_func", lambda: None)(),
                        data_collate_info=getattr(model, "get_extra_collate_infos", lambda: {})(),
                    )
                )
            batch = collate_fn([sample])
            output = model(batch)
        return get_decision_model(decision_model).decode(output, batch)
    finally:
        model.train(was_training)
