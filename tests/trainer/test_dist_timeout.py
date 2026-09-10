"""Unit tests for `TrainingArguments.dist_timeout`.

The value reaches `init_process_group` as the default group's collective timeout.
Torch treats a collective that outlives it as a hang and the NCCL watchdog aborts
the process, so an operation that legitimately blocks longer -- a checkpoint
written over a slow network filesystem -- needs this raised to survive.
"""

from datetime import timedelta

import pytest

from veomni.arguments import TrainingArguments


class TestDistTimeout:
    def test_unset_defers_to_torch(self):
        """None is not a value we can substitute: torch's default differs per backend."""
        assert TrainingArguments().dist_timeout is None

    def test_seconds_become_a_timedelta(self):
        assert TrainingArguments(dist_timeout_seconds=1800).dist_timeout == timedelta(seconds=1800)

    @pytest.mark.parametrize("seconds", [0, -1])
    def test_non_positive_is_rejected_at_parse_time(self, seconds):
        """Torch reads a non-positive timeout as "expire immediately", which aborts the
        first collective. Rejecting it here fails the run before it reserves any GPU."""
        with pytest.raises(ValueError, match="must be positive"):
            TrainingArguments(dist_timeout_seconds=seconds)
