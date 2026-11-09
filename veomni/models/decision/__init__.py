"""Decision-model contracts and registered architecture implementations."""

from .base import DECISION_MODELS, ScoringOutput, decode_decisions, get_decision_model
from .models.clef import CLEF_SPEC
from .models.clef.modeling import ClefDecisionModel, load_local_model
from .models.clef.release import save_clef_release, split_clef_release_state_dict
from .processing import DecisionPreprocessor


DECISION_MODELS[CLEF_SPEC.name] = CLEF_SPEC


def predict_request(*args, **kwargs):
    # Keep data -> registry imports independent of the inference helper.
    from .inference import predict_request as predict

    return predict(*args, **kwargs)


__all__ = [
    "CLEF_SPEC",
    "ClefDecisionModel",
    "DecisionPreprocessor",
    "ScoringOutput",
    "decode_decisions",
    "get_decision_model",
    "load_local_model",
    "predict_request",
    "save_clef_release",
    "split_clef_release_state_dict",
]
