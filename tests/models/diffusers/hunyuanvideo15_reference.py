"""Load the pinned upstream CPU reference without its CUDA/SP launch dependencies."""

import ast
import sys
import types
from pathlib import Path

import einops
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention


def load_reference(root):
    root = Path(root)
    transformer_root = root / "hyvideo/models/transformers"
    for name, path in [
        ("_hv15_reference", transformer_root),
        ("_hv15_reference.modules", transformer_root / "modules"),
    ]:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    # Extract the original attention functions unchanged. Use eager flex_attention
    # rather than the source module's unconditional torch.compile CUDA setup.
    tree = ast.parse((transformer_root / "modules/attention.py").read_text())
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("attention", "parallel_attention", "sequence_parallel_attention")
    ]
    attention = types.ModuleType("_hv15_reference.modules.attention")
    attention.__dict__.update(
        torch=torch,
        F=F,
        einops=einops,
        Optional=__import__("typing").Optional,
        flex_attention=flex_attention,
        get_parallel_state=lambda: types.SimpleNamespace(sp_enabled=False),
        maybe_fallback_attn_mode=lambda _: "torch",
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), "upstream_attention.py", "exec"), attention.__dict__)
    sys.modules[attention.__name__] = attention
    # Only embed_layers needs a commons utility; execute its unchanged body
    # with the standard equivalent of upstream to_2tuple injected.
    for module_name in ("embed_layers", "mlp_layers"):
        path = transformer_root / f"modules/{module_name}.py"
        tree = ast.parse(path.read_text())
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == "hyvideo.commons")]
        embed = types.ModuleType(f"_hv15_reference.modules.{module_name}")
        embed.__package__ = "_hv15_reference.modules"
        embed.to_2tuple = torch.nn.modules.utils._pair
        exec(compile(tree, str(path), "exec"), embed.__dict__)
        sys.modules[embed.__name__] = embed
    tree = ast.parse((root / "hyvideo/models/text_encoders/byT5/__init__.py").read_text())
    mapper = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ByT5Mapper")
    scope = {"nn": torch.nn}
    exec(compile(ast.Module(body=[mapper], type_ignores=[]), "upstream_byt5.py", "exec"), scope)
    path = transformer_root / "hunyuanvideo_1_5_transformer.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        n
        for n in tree.body
        if not (
            isinstance(n, ast.ImportFrom) and n.module and (n.module.startswith("hyvideo.") or n.module == "loguru")
        )
    ]
    module = types.ModuleType("_hv15_reference.transformer")
    module.__package__ = "_hv15_reference"
    module.__dict__.update(
        ByT5Mapper=scope["ByT5Mapper"],
        maybe_fallback_attn_mode=lambda _: "torch",
        get_parallel_state=lambda: types.SimpleNamespace(sp_enabled=False),
    )
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module.HunyuanVideo_1_5_DiffusionTransformer
