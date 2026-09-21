import contextlib

import torch
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state_if_fully_sharded_module, _module_handle
from torch.distributed.fsdp._runtime_utils import (
    _post_backward_hook,
    _pre_backward_hook,
)
from torch.utils.checkpoint import (
    _get_autocast_kwargs,
    _get_device_module,
    _infer_device_type,
    check_backward_validity,
    detach_variable,
    get_device_states,
    set_device_states,
)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.device = _infer_device_type(*args)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(ctx.device)
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*args)

        # patch code, remove the extra allgather with use_reentrant + ckpt
        if not isinstance(ctx.run_function, torch.nn.Module):
            ctx.patch_module = ctx.run_function.__self__
        else:
            ctx.patch_module = ctx.run_function
        state = _get_module_fsdp_state_if_fully_sharded_module(ctx.patch_module)
        if state:
            handle = _module_handle(state, ctx.patch_module)
            if handle:
                handle._needs_pre_backward_unshard = True
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "When use_reentrant=True, torch.utils.checkpoint is incompatible"
                " with .grad() or passing an `inputs` parameter to .backward()."
                " To resolve this error, you can either set use_reentrant=False,"
                " or call .backward() without passing the `inputs` argument."
            )
        # patch code, remove the extra allgather with use_reentrant + ckpt
        handle = None
        state = _get_module_fsdp_state_if_fully_sharded_module(ctx.patch_module)
        if state:
            handle = _module_handle(state, ctx.patch_module)
            if handle:
                _pre_backward_hook(state, ctx.patch_module, handle, None)

        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states)
            detached_inputs = detach_variable(tuple(inputs))

            device_autocast_ctx = (
                torch.amp.autocast(device_type=ctx.device, **ctx.device_autocast_kwargs)
                if torch.amp.is_autocast_available(ctx.device)
                else contextlib.nullcontext()
            )
            with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError("none of output has requires_grad=True, this checkpoint() is not necessary")
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached_inputs)

        # patch code, remove the extra allgather with use_reentrant + ckpt
        if handle:
            _post_backward_hook(state, handle, None)

        return (None, None) + grads
