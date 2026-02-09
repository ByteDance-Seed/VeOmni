"""Utilities for saving and loading training results for distributed tests."""

import json
import os
from typing import Any


def save_training_results(
    results: dict[str, Any],
    output_dir: str,
    run_name: str,
    config: dict[str, Any],
) -> str:
    """Save training results to disk.

    Args:
        results: Training results (loss, grad_norm, etc.)
        output_dir: Directory to save results
        run_name: Run name (e.g., "baseline", "fsdp2_rmpad")
        config: Test configuration

    Returns:
        Path to the saved results file
    """
    os.makedirs(output_dir, exist_ok=True)

    run_data = {
        "results": results,
        "config": config,
        "run_name": run_name,
    }

    results_path = os.path.join(output_dir, f"{run_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(run_data, f, indent=2)

    print(f"[{run_name}] Results saved to: {results_path}")
    return results_path


def load_training_results(results_dir: str, run_name: str) -> dict[str, Any]:
    """Load training results from disk.

    Args:
        results_dir: Directory containing results
        run_name: Run name to load

    Returns:
        Dictionary with training results and config

    Raises:
        FileNotFoundError: If results file doesn't exist
        ValueError: If results are invalid
    """
    results_path = os.path.join(results_dir, f"{run_name}_results.json")

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results not found: {results_path}")

    with open(results_path) as f:
        run_data = json.load(f)

    if "results" not in run_data or "config" not in run_data:
        raise ValueError(f"Invalid results file: {results_path}")

    print(f"[Load] Loaded results from: {results_path}")
    print(f"[Load] Config: {run_data['config']}")

    return run_data


def verify_config_compatibility(
    baseline_config: dict[str, Any],
    test_config: dict[str, Any],
    strict_keys: list = None,
) -> bool:
    """Verify test config is compatible with baseline config.

    Args:
        baseline_config: Baseline configuration
        test_config: Test configuration
        strict_keys: Keys that must match exactly

    Returns:
        True if compatible

    Raises:
        ValueError: If incompatible
    """
    if strict_keys is None:
        strict_keys = [
            "model_name",
            "global_batch_size",
            "max_seq_len",
            "num_train_steps",
        ]

    mismatches = []
    for key in strict_keys:
        if key in baseline_config and key in test_config:
            if baseline_config[key] != test_config[key]:
                mismatches.append(f"  {key}: baseline={baseline_config[key]}, test={test_config[key]}")

    if mismatches:
        error_msg = "Config mismatch between baseline and test:\n" + "\n".join(mismatches)
        raise ValueError(error_msg)

    return True
