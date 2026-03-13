import subprocess

import pytest
from e2e_test_helpers import parse_training_log
from exec_scripts import E2E_TEST_SCRIPT


test_cases = [
    pytest.param("qwen3_0p6b_base_tulu_sft_no_reshard"),
]


@pytest.mark.parametrize("task_name", test_cases)
def test_e2e_training(task_name):
    exec_script = E2E_TEST_SCRIPT[task_name]
    e2e_test_res = subprocess.run(exec_script, shell=True, check=True, capture_output=True, text=True)
    print(e2e_test_res.stdout)
    exec_log_df = parse_training_log(e2e_test_res.stdout)
    print(exec_log_df)
