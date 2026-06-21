from pathlib import Path

from tests.seed_omni.parity_suite.pytest_entrypoint import make_parity_test


test_bagel_parity_case = make_parity_test(Path(__file__).parent)
