import argparse
from dataclasses import dataclass


@dataclass
class TestArguments:
    """Arguments for FSDP2 integration validation tests."""

    # Mode selection
    mode: str

    # Directory and model configuration
    baseline_dir: str = "./baseline_outputs"
    model_name: str = "Qwen/Qwen3-0.6B"

    # Data configuration
    data_path: str = None
    data_format: str = "rmpad_with_pos_ids"

    # Training hyperparameters
    global_batch_size: int = None
    micro_batch_size: int = None
    max_seq_len: int = None
    num_train_steps: int = None

    # Model configuration
    attn_implementation: str = "flash_attention_2"

    # Parallelism configuration
    ulysses_sp_size: int = 1

    # Tolerance settings (must be explicitly set by each test)
    rtol_loss: float = None
    rtol_grad_norm: float = None


def parse_test_arguments() -> TestArguments:
    """Parse command-line arguments into TestArguments dataclass.

    Returns:
        TestArguments instance with parsed values
    """
    parser = argparse.ArgumentParser(
        description="FSDP2 integration test for VeOmni",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["generate_baseline", "test"],
        help="Mode: generate_baseline or test",
    )

    # Directory and model configuration
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default="./baseline_outputs",
        help="Directory to save/load baseline results",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name or path",
    )

    # Data configuration
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset (supports parquet, jsonl, arrow, csv)",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="rmpad_with_pos_ids",
        choices=["padded_bsh", "rmpad_with_pos_ids"],
        help="Data format for FSDP2 test",
    )

    # Training hyperparameters
    parser.add_argument(
        "--global_batch_size",
        type=int,
        required=True,
        help="Global batch size across all GPUs",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        required=True,
        help="Micro batch size per GPU",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        required=True,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--num_train_steps",
        type=int,
        required=True,
        help="Number of training steps",
    )

    # Model configuration
    # Note: Use "veomni_flash_attention_2_with_sp" or "veomni_flash_attention_3_with_sp"
    # for Ulysses sequence parallel tests. These implementations include the all-to-all
    # communication needed for sequence parallelism.
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=[
            "eager",
            "flash_attention_2",
            "flash_attention_3",
            "veomni_flash_attention_2_with_sp",
            "veomni_flash_attention_3_with_sp",
        ],
        help="Attention implementation. Use veomni_flash_attention_*_with_sp for Ulysses SP.",
    )

    # Parallelism configuration
    parser.add_argument(
        "--ulysses_sp_size",
        type=int,
        default=1,
        help="Ulysses sequence parallel size (1 = no SP)",
    )

    # Tolerance settings (required for test mode, optional for generate_baseline)
    parser.add_argument(
        "--rtol_loss",
        type=float,
        default=None,
        help="Relative tolerance for loss comparison (required for test mode)",
    )
    parser.add_argument(
        "--rtol_grad_norm",
        type=float,
        default=None,
        help="Relative tolerance for gradient norm comparison (required for test mode)",
    )

    args = parser.parse_args()

    # Convert argparse Namespace to TestArguments dataclass
    return TestArguments(
        mode=args.mode,
        baseline_dir=args.baseline_dir,
        model_name=args.model_name,
        data_path=args.data_path,
        data_format=args.data_format,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        max_seq_len=args.max_seq_len,
        num_train_steps=args.num_train_steps,
        attn_implementation=args.attn_implementation,
        ulysses_sp_size=args.ulysses_sp_size,
        rtol_loss=args.rtol_loss,
        rtol_grad_norm=args.rtol_grad_norm,
    )
