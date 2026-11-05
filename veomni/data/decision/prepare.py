"""Select the typed-decisions train/evaluation partitions."""

import random
from dataclasses import dataclass, field
from typing import Any

from . import typed_decisions


@dataclass
class PreparedDecisionData:
    train_cases: list
    validation_cases: list
    padding_case: Any
    _options: Any = field(repr=False)

    def evaluation_cases(self, split):
        if split == "validation":
            cases = self.validation_cases
        elif split == "test":
            if self._options.mode != "evaluate":
                raise ValueError("Official test is allowed only in explicit evaluation mode")
            cases = typed_decisions.load_cases(self._options, split="test")
        else:
            raise ValueError("Unknown evaluation split")
        return cases[: self._options.limit_eval_cases] if self._options.limit_eval_cases else cases


def prepare_decision_data(options, *, seed) -> PreparedDecisionData:
    cases = typed_decisions.load_cases(options)
    order = list(cases)
    random.Random(seed).shuffle(order)
    validation_size = (
        max(1, int(len(order) * options.validation_fraction)) if options.validation_fraction and len(order) > 1 else 0
    )
    validation_cases = order[:validation_size]
    train_cases = order[validation_size:]
    if options.limit_train_cases:
        train_cases = train_cases[: options.limit_train_cases]
    if not train_cases:
        raise ValueError("Training partition is empty")
    return PreparedDecisionData(
        train_cases=train_cases,
        validation_cases=validation_cases,
        padding_case=train_cases[0],
        _options=options,
    )
