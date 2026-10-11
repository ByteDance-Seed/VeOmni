# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run one stack's DiffusionGemma forward (and optionally backward) and dump the result.

The two stacks pin transformers versions that cannot share a process (VeOmni 5.16,
NeMo 5.15+), so parity is dump-and-diff: run this once per venv against the same
checkpoint and seed, then ``compare_dumps.py`` the two files.

    VE=/opt/tiger/diffusion-gemma-veomni
    AM=/opt/tiger/diffusion-gemma-automodel
    $VE/.venv/bin/python $VE/scripts/parity/dump_forward.py --stack veomni \
        --ckpt /tmp/dg_parity/tiny_ckpt --out /tmp/dg_parity/veomni.pt --grads
    $AM/.venv/bin/python $VE/scripts/parity/dump_forward.py --stack nemo \
        --ckpt /tmp/dg_parity/tiny_ckpt --out /tmp/dg_parity/nemo.pt --grads

Everything runs on CPU in fp32. Not because training does -- it does not -- but
because bf16 puts a ~1e-2 floor under the comparison, and a disagreement that large
is indistinguishable from a real semantic difference.

What is compared, and why not more:

* ``encoder_hidden`` -- the causal pass over the clean sequence, before either stack's
  LM head. This is most of the model: embeddings, both RoPE variants, GQA with the
  distinct global head dim, the MoE, the norms.
* ``decoder_logits`` -- the denoising pass, softcap included. Disagreement here with
  a matching ``encoder_hidden`` isolates the fault to the decoder's mask/positions.

Self-conditioning is off. It is a second forward whose output feeds the first, so
leaving it on would mean a single number at the end standing for two passes; it is
worth its own comparison, not a share of this one.

``--inputs`` swaps the random draw for real tokenised SFT rows written by
``make_real_inputs.py``. At the 26B checkpoint the fp32 gradients are ~104 GiB per
stack, which no local disk here holds, so the stacked expert weights -- 3-D, and
~90% of the parameters -- are dumped as row and column norm profiles instead of
whole tensors. See ``gradient_signature``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Before torch: dynamo reads this at import. Parts of NeMo's CPU path compile, and an
# inductor-generated kernel is a third implementation nobody asked this to compare --
# quite apart from inductor needing a toolchain the venv does not necessarily have.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import re  # noqa: E402

import torch  # noqa: E402


sys.path.insert(0, str(Path(__file__).resolve().parent))

import parity_inputs  # noqa: E402


LAYER_NAME = re.compile(r"(?:^|\.)layers\.(\d+)$")

# Set from `--trace-layer`; a list so the runners can read it without being handed
# another parameter that only the debugging path uses.
TRACE_LAYER: list[int | None] = [None]


def attach_layer_trace(model, sink: dict[str, torch.Tensor]):
    """Record each trunk layer's first output, to find *where* two stacks part company.

    A single number off the end of 30 layers cannot tell a wrong layer from a right
    one whose input already drifted. The first call per layer is the encoder pass in
    both stacks -- the trunk is shared and the encoder runs before the canvas -- so
    later calls are dropped rather than overwriting it.

    The vision tower is skipped: VeOmni builds one and NeMo does not, and a text batch
    never enters it.
    """
    handles = []
    for name, module in model.named_modules():
        match = LAYER_NAME.search(name)
        if match is None or "vision" in name:
            continue
        index = int(match.group(1))

        def hook(_module, _args, output, index=index):
            key = f"layer_{index:03d}"
            if key not in sink:
                tensor = output[0] if isinstance(output, tuple) else output
                sink[key] = tensor.detach().to(torch.float32).clone()

        handles.append(module.register_forward_hook(hook))
    return handles


def attach_submodule_trace(model, layer: int, sink: dict[str, torch.Tensor]):
    """Record every submodule output inside one trunk layer, keyed relative to the layer.

    Keying by the path *below* ``layers.<n>`` is what makes the two stacks comparable
    without a rename table: they disagree on the container (``model.decoder.layers.0``
    against ``model.layers.0``) but not on what sits inside it, so ``self_attn.q_proj``
    is ``self_attn.q_proj`` on both sides.
    """
    handles = []
    for name, module in model.named_modules():
        match = re.search(rf"(?:^|\.)layers\.{layer}\.(.+)$", name)
        if match is None or "vision" in name:
            continue
        relative = match.group(1)

        def hook(_module, _args, output, relative=relative):
            key = f"sub::{relative}"
            if key in sink:
                return
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor):
                sink[key] = tensor.detach().to(torch.float32).clone()

        handles.append(module.register_forward_hook(hook))
    return handles


def load_geometry(ckpt: Path) -> tuple[int, int]:
    """``(vocab_size, canvas_length)`` straight from the checkpoint's config."""
    import json

    config = json.loads((ckpt / "config.json").read_text())
    return int(config["text_config"]["vocab_size"]), int(config["canvas_length"])


def load_veomni(ckpt: Path, canvas_length: int):
    from veomni.models.transformers.diffusion_gemma import align_nemo
    from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaForBlockDiffusion,
    )

    model = DiffusionGemmaForBlockDiffusion.from_pretrained(ckpt, dtype=torch.float32)
    # `flat_seq` is the mode whose decoder sees one key axis of `cache + K*C`, which is
    # the layout NeMo's single dense mask describes. `attn_dg_prefix` off keeps it on
    # dense sdpa: the Triton kernel is CUDA-only and is Step 4's subject, not this one.
    model.config.canvas_batching = "flat_seq"
    model.config.text_config.attn_dg_prefix = False
    # A no-op unless `--align-nemo` set the switch. It has to run here rather than after
    # the forward: what it changes is the parameter list, and the gradients are read off
    # that. See `align_nemo.fold_per_expert_scale`.
    align_nemo.fold_per_expert_scale(model)
    model.train()
    return model


def run_veomni(
    ckpt: Path,
    inputs: parity_inputs.ParityInputs,
    trace: dict[str, torch.Tensor] | None = None,
    self_conditioning: bool = False,
) -> dict[str, torch.Tensor]:
    model = load_veomni(ckpt, inputs.canvas_length)
    if trace is not None:
        attach_layer_trace(model, trace)
        if TRACE_LAYER[0] is not None:
            attach_submodule_trace(model, TRACE_LAYER[0], trace)

    context = parity_inputs.per_canvas_context_mask(inputs)
    # VeOmni wants `[B, K, cache + C]` with the canvas half all-True: a canvas always
    # attends to itself, and this mode carries no way to hide part of one.
    decoder_attention_mask = torch.nn.functional.pad(context, (0, inputs.canvas_length), value=True)

    batch = dict(
        input_ids=inputs.clean_input_ids,
        attention_mask=torch.ones_like(inputs.clean_input_ids, dtype=torch.bool),
        position_ids=inputs.encoder_position_ids,
        decoder_input_ids=inputs.canvas_ids.view(inputs.batch_size, inputs.num_canvases, inputs.canvas_length),
        decoder_attention_mask=decoder_attention_mask,
        decoder_position_ids=inputs.decoder_position_ids,
        use_cache=False,
    )

    conditioning = {}
    if self_conditioning:
        # The two-pass scheme lives in `DiffusionGemmaTrainer.preforward` on this side,
        # not in the model, so it is restated here. Every sample is conditioned, which
        # is what `do_self_conditioning=True` asks NeMo for.
        with torch.no_grad():
            embeddings = model(**batch, return_logits=False, return_self_conditioning_embeddings=True)
        conditioning = {
            "self_conditioning_embeddings": embeddings.self_conditioning_embeddings,
            "self_conditioning_mask": torch.ones(
                inputs.batch_size * inputs.num_canvases, dtype=torch.bool, device=inputs.canvas_ids.device
            ),
        }

    output = model(**batch, **conditioning, return_logits=True)
    return {
        "encoder_hidden": output.encoder_last_hidden_state,
        "decoder_logits": output.logits.reshape(inputs.batch_size, inputs.response_length, -1),
    }, model


def force_nemo_cpu_loading() -> None:
    """Point NeMo's loader at the CPU, which it has no parameter for.

    ``from_pretrained`` materialises weights on ``torch.cuda.current_device()``
    unconditionally (``_transformers/auto_model.py``). At 26B in fp32 that is
    ~104 GiB onto an 80 GiB card, and this comparison runs on the CPU anyway --
    a GPU reduction order would add a difference belonging to neither stack.

    Loading bf16 and upcasting afterwards would fit on a card and is bit-exact
    for the checkpoint's own weights, but not for NeMo's: it folds
    ``per_expert_scale`` into the expert down-projection *during* the load, and
    the released scale is not 1 (0.980--1.023 across experts), so the fold would
    round in bf16 and inject ~1e-3 into the very tensors being compared.

    Installed only when CUDA is genuinely absent, so nothing here reaches a run
    that does have a GPU.
    """
    import nemo_automodel._transformers.auto_model as auto_model

    # Two call sites read it purely to name a default device; nothing dereferences
    # the index once the placement below is in force.
    torch.cuda.current_device = lambda: 0

    stock = auto_model.apply_model_infrastructure

    def on_cpu(*args, **kwargs):
        kwargs["device"] = torch.device("cpu")
        return stock(*args, **kwargs)

    auto_model.apply_model_infrastructure = on_cpu


def widen_nemo_expert_fold() -> None:
    """Fold ``per_expert_scale`` in fp32 instead of the checkpoint's bf16.

    ``DiffusionGemmaStateDictAdapter`` multiplies ``down_proj`` by the scale and only
    then upcasts (``state_dict_adapter.py``: the fold is on the raw state-dict values,
    ``.to(self.dtype)`` comes after). Both operands arrive from a bf16 checkpoint, so
    the product is rounded to bf16 before being widened, and a run asking for fp32
    weights gets an expert down-projection carrying ~1.5e-3 of relative error.

    This is the diagnostic arm, not a fix: it says how much of the two stacks'
    disagreement that rounding accounts for. The fix belongs in the adapter.
    """
    from nemo_automodel.components.models.diffusion_gemma import state_dict_adapter

    def in_fp32(down, per_expert_scale):
        return down.to(torch.float32) * per_expert_scale[:, None, None].to(torch.float32)

    state_dict_adapter.DiffusionGemmaStateDictAdapter._fold_per_expert_scale = staticmethod(in_fp32)


def load_nemo(ckpt: Path, canvas_length: int, fp32_expert_fold: bool = False, self_conditioning: bool = False):
    if not torch.cuda.is_available():
        force_nemo_cpu_loading()
    if fp32_expert_fold:
        widen_nemo_expert_fold()

    from nemo_automodel import NeMoAutoModelForCausalLM
    from nemo_automodel.components.models.common import BackendConfig

    model = NeMoAutoModelForCausalLM.from_pretrained(
        # A str, not the Path: NeMo's loader branches on it being a config object
        # otherwise and asks a Path for `name_or_path`.
        str(ckpt),
        dtype=torch.float32,
        canvas_length=canvas_length,
        self_conditioning=self_conditioning,
        # Unfrozen, unlike every shipped recipe. Freezing is a training decision that
        # would simply delete the router gradients from the comparison, and the router
        # is where the two MoE implementations are most likely to disagree.
        freeze_router=False,
        backend=BackendConfig(
            attn="sdpa",
            linear="torch",
            rms_norm="torch_fp32",
            experts="torch_mm",
            dispatcher="torch",
            enable_hf_state_dict_adapter=True,
            enable_fsdp_optimizations=False,
        ),
    )
    # NeMo's loader places the model on the visible accelerator. Parity runs on CPU:
    # the comparison is about semantics, and a GPU reduction order would add a
    # difference that has nothing to do with the two implementations.
    model = model.to("cpu")
    model.train()
    return model


def run_nemo(
    ckpt: Path,
    inputs: parity_inputs.ParityInputs,
    trace: dict[str, torch.Tensor] | None = None,
    fp32_expert_fold: bool = False,
    self_conditioning: bool = False,
) -> dict[str, torch.Tensor]:
    from nemo_automodel.components.models.diffusion_gemma.attention_mask import (
        build_block_diffusion_training_mask,
    )

    model = load_nemo(ckpt, inputs.canvas_length, fp32_expert_fold, self_conditioning)
    if trace is not None:
        attach_layer_trace(model, trace)
        if TRACE_LAYER[0] is not None:
            attach_submodule_trace(model, TRACE_LAYER[0], trace)

    sliding_window = model.config.text_config.sliding_window
    mask_full, mask_sliding = build_block_diffusion_training_mask(
        prefix_lengths=inputs.prefix_lengths,
        response_length=inputs.response_length,
        enc_len=inputs.sequence_length,
        block_size=inputs.canvas_length,
        sliding_window=sliding_window,
        device="cpu",
        dtype=torch.float32,
    )

    output = model(
        input_ids=inputs.clean_input_ids,
        canvas_ids=inputs.canvas_ids,
        encoder_position_ids=inputs.encoder_position_ids,
        decoder_position_ids=inputs.decoder_position_ids,
        decoder_attention_mask={"full_attention": mask_full, "sliding_attention": mask_sliding},
        # NeMo owns the two-pass scheme inside `forward` and demands the per-example
        # coins whenever self-conditioning is on. All true: the comparison is of the
        # conditioned path, and a mixed draw would only test the gate.
        **({"do_self_conditioning": torch.ones(inputs.batch_size, dtype=torch.bool)} if self_conditioning else {}),
    )

    # NeMo's forward returns encoder *logits*; the hidden state it softcapped them from
    # is what is comparable without restating either stack's LM head here, so the encode
    # pass is re-run for it. Same inputs, same weights, no grad -- it is a read.
    with torch.no_grad():
        _, encoder_hidden = model.model(
            mode="encode",
            input_ids=inputs.clean_input_ids,
            position_ids=inputs.encoder_position_ids,
            padding_mask=None,
            return_hidden=True,
        )

    return {"encoder_hidden": encoder_hidden, "decoder_logits": output.logits}, model


def load_inputs(path: Path) -> tuple[parity_inputs.ParityInputs, torch.Tensor | None]:
    """Read the frozen real-data inputs, so neither stack builds its own."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    inputs = parity_inputs.ParityInputs(
        clean_input_ids=blob["clean_input_ids"],
        canvas_ids=blob["canvas_ids"],
        encoder_position_ids=blob["encoder_position_ids"],
        decoder_position_ids=blob["decoder_position_ids"],
        prefix_lengths=blob["prefix_lengths"],
        num_canvases=blob["num_canvases"],
        canvas_length=blob["canvas_length"],
    )
    return inputs, blob.get("supervised_mask")


def response_targets(inputs: parity_inputs.ParityInputs) -> torch.Tensor:
    """The clean response each canvas is denoising toward, gathered per example.

    Not a single slice: with uneven prompts each row's response starts at its own
    offset, and a shared slice would hand the shorter rows somebody else's tokens.
    Both stacks get whatever this returns, so parity does not depend on it being
    right -- only on it being the same. It is written to be right anyway, because a
    gradient taken against the wrong targets is a weaker test of the MoE routing.
    """
    offsets = inputs.prefix_lengths.unsqueeze(-1) + torch.arange(inputs.response_length)
    return inputs.clean_input_ids.gather(1, offsets)


def gradient_signature(grad: torch.Tensor) -> dict[str, torch.Tensor]:
    """Row and column norm profiles of a stacked expert gradient, plus scalars.

    The 26B model keeps ~90% of its parameters in 3-D ``(experts, ., .)`` tensors;
    dumping those in fp32 costs ~91 GiB per stack, which is not available here. A
    norm along each of the two inner axes keeps what a whole-tensor diff would be
    used for: a per-expert or per-row disagreement still shows, and the profile is
    ~4k numbers instead of 500M.

    Both axes are kept rather than one because NeMo stores these transposed. The
    comparison pairs this stack's rows with the other's columns, so a genuine
    scramble stays distinguishable from the known layout difference -- collapsing
    to a single norm would make the two indistinguishable.
    """
    return {
        "scalars": torch.stack([grad.norm(), grad.sum(), grad.abs().max()]),
        "norm_rows": grad.norm(dim=2),
        "norm_cols": grad.norm(dim=1),
    }


def unreduced_loss(
    logits: torch.Tensor, targets: torch.Tensor, supervised: torch.Tensor | None = None
) -> torch.Tensor:
    """Summed per-token cross-entropy -- deliberately *not* either stack's objective.

    The two normalize differently (VeOmni per sample and by supervised count, NeMo by
    global token count), and comparing gradients through two different scalars would
    report a constant factor as a disagreement. A plain sum is the same function of the
    logits on both sides, so any gradient difference is the model's.

    With real rows ``supervised`` restricts the sum to the tokens the recipe trains on;
    the canvas tail is pad, and letting the model be graded on predicting pad would put
    most of the gradient somewhere training never puts it.
    """
    per_token = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="none",
    )
    if supervised is not None:
        per_token = per_token * supervised.reshape(-1).to(per_token.dtype)
    return per_token.sum()


def _git_id_at(root: Path) -> str | None:
    """``<commit>`` for the repo containing ``root``, ``<commit>-dirty`` if edited."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        # Tracked files only. Counting untracked ones marks every dump dirty as soon
        # as a scratch script exists anywhere in the tree, which is always -- and a
        # flag that is always set carries nothing.
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None
    return f"{commit}-dirty" if dirty else commit


def _git_id(module: str) -> str | None:
    """The same, for the repo a stack happens to be imported from."""
    loaded = sys.modules.get(module)
    if loaded is None or getattr(loaded, "__file__", None) is None:
        return None
    return _git_id_at(Path(loaded.__file__).resolve().parent.parent)


def _digest(path: Path) -> str | None:
    """A short content hash, so a saved dump can still name the bytes it came from."""
    target = path / "config.json" if path.is_dir() else path
    if not target.is_file():
        return None
    return hashlib.sha256(target.read_bytes()).hexdigest()[:16]


def provenance(args: argparse.Namespace) -> dict:
    """What this dump was produced from.

    A dump outlives the shell that made it, and these comparisons turn on things no
    filename records: which checkpoint, which input rows, which alignment switches,
    which revision of either stack. Without them a kept baseline is only a number
    whose meaning has to be reconstructed from memory.
    """
    return {
        "argv": " ".join(sys.argv),
        "when": datetime.now().isoformat(timespec="seconds"),
        "ckpt": str(args.ckpt),
        "ckpt_config_sha": _digest(args.ckpt),
        "inputs": str(args.inputs) if args.inputs else None,
        "inputs_sha": _digest(args.inputs) if args.inputs else None,
        # The dumping script and the stack it dumps are separate revisions, and for a
        # NeMo run they are separate repos -- so `veomni` being absent here would
        # otherwise leave the run with no record of the code that produced the file.
        "script": _git_id_at(Path(__file__).resolve().parent),
        "veomni": _git_id("veomni"),
        "nemo_automodel": _git_id("nemo_automodel"),
        "torch": torch.__version__,
        "switches": {k: v for k, v in os.environ.items() if k.startswith("VEOMNI_ALIGN_NEMO_")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", choices=["veomni", "nemo"], required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--canvases", type=int, default=3)
    parser.add_argument("--prefix", type=int, nargs="+", default=[5])
    parser.add_argument("--inputs", type=Path, default=None, help="frozen real inputs from make_real_inputs.py")
    parser.add_argument(
        "--grads", action="store_true", help="also backward the summed CE and dump per-parameter grads"
    )
    parser.add_argument(
        "--layer-trace",
        action="store_true",
        help="also dump every trunk layer's encoder-pass output, to locate where two stacks diverge",
    )
    parser.add_argument(
        "--trace-layer",
        type=int,
        default=None,
        help="with --layer-trace, also dump every submodule output inside this trunk layer",
    )
    parser.add_argument(
        "--fp32-expert-fold",
        action="store_true",
        help=(
            "nemo only: fold per_expert_scale in fp32 rather than the checkpoint's bf16, "
            "to measure how much of the stacks' disagreement that rounding accounts for"
        ),
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help=(
            "load and dump the parameters, no forward. Separates 'the two stacks compute "
            "different things' from 'the two stacks loaded different weights', which at a "
            "released checkpoint is not a given: NeMo rewrites the expert tensors on load"
        ),
    )
    parser.add_argument(
        "--full-grads",
        action="store_true",
        help="dump stacked expert gradients whole instead of as norm profiles (~91 GiB at 26B)",
    )
    parser.add_argument(
        "--self-conditioning",
        action="store_true",
        help=(
            "run the two-pass self-conditioning scheme both stacks use in training, with every "
            "sample conditioned. Off by default because it doubles the decode; on, it is the only "
            "way the embedding table's self-conditioning gradient path is reached at all"
        ),
    )
    parser.add_argument(
        "--align-nemo",
        action="store_true",
        help=(
            "veomni only: adopt NeMo's training semantics where the two stacks compute the "
            "same function but hand the optimizer a different problem -- fold per_expert_scale "
            "into the experts, and let self-conditioning reach the embedding table. A no-op on "
            "the nemo side, which already has both. See veomni/.../diffusion_gemma/align_nemo.py"
        ),
    )
    args = parser.parse_args()

    if args.align_nemo:
        if args.stack != "veomni":
            parser.error("--align-nemo describes what VeOmni should adopt; NeMo already has it.")
        os.environ["VEOMNI_ALIGN_NEMO_EXPERT_SCALE_FOLD"] = "1"
        os.environ["VEOMNI_ALIGN_NEMO_SELF_COND_EMBED_GRAD"] = "1"

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)

    if args.inputs is not None:
        inputs, supervised = load_inputs(args.inputs)
    else:
        vocab_size, canvas_length = load_geometry(args.ckpt)
        inputs = parity_inputs.build(
            vocab_size=vocab_size,
            canvas_length=canvas_length,
            num_canvases=args.canvases,
            prefix_lengths=tuple(args.prefix),
            seed=args.seed,
        )
        supervised = None

    if args.weights_only:
        loader = load_veomni if args.stack == "veomni" else load_nemo
        model = loader(args.ckpt, inputs.canvas_length)
        dump = {}
        for name, parameter in model.named_parameters():
            weight = parameter.detach().to(torch.float32)
            if weight.dim() == 3 and not args.full_grads:
                for part, value in gradient_signature(weight).items():
                    dump[f"grad::{name}::{part}"] = value.clone()
            else:
                dump[f"grad::{name}"] = weight.clone()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"stack": args.stack, "seed": args.seed, "from": provenance(args), "tensors": dump}, args.out)
        print(f"[{args.stack}] wrote {len(dump)} parameter tensors to {args.out}")
        return

    TRACE_LAYER[0] = args.trace_layer
    trace: dict[str, torch.Tensor] | None = {} if args.layer_trace else None
    extra = {"self_conditioning": True} if args.self_conditioning else {}
    if args.fp32_expert_fold and args.stack == "nemo":
        extra["fp32_expert_fold"] = True
    runner = run_veomni if args.stack == "veomni" else run_nemo
    tensors, model = runner(args.ckpt, inputs, trace, **extra)
    dump = {name: value.detach().to(torch.float32).clone() for name, value in tensors.items()}
    if trace:
        dump.update(trace)

    if args.grads:
        model.zero_grad(set_to_none=True)
        unreduced_loss(tensors["decoder_logits"], response_targets(inputs), supervised).backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().to(torch.float32)
            if grad.dim() == 3 and not args.full_grads:
                for part, value in gradient_signature(grad).items():
                    dump[f"grad::{name}::{part}"] = value.clone()
            else:
                dump[f"grad::{name}"] = grad.clone()
            # 26B in fp32 is ~104 GiB of gradients; holding the dump and the live
            # grads at once is what makes this not fit. Each is released as it is
            # summarised, so peak stays at the model plus one tensor.
            parameter.grad = None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stack": args.stack, "seed": args.seed, "from": provenance(args), "tensors": dump}, args.out)

    print(f"[{args.stack}] wrote {len(dump)} tensors to {args.out}")
    for name in sorted(dump):
        if not name.startswith("grad::"):
            print(f"  {name}: {tuple(dump[name].shape)}")


if __name__ == "__main__":
    main()
