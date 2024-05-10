import torch
import importlib
import torch.nn as nn
from einops import rearrange


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, dims=3):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.conv_nd = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d if dims == 3 else None
        self.to_qkv = self.conv_nd(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = self.conv_nd(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape[:4]
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) ... -> qkv b heads c (...)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w ...) -> b (heads c) h w ...', heads=self.heads, h=h, w=w)
        return self.to_out(out)
    
    
def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) if torch.is_tensor(x) else x for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) if torch.is_tensor(x) else x for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_tensors_without_none = [x for x in ctx.input_tensors if torch.is_tensor(x)]
        input_tensors_indices = [ix for ix, x in enumerate(ctx.input_tensors) if not torch.is_tensor(x)]
        input_grads = torch.autograd.grad(
            output_tensors,
            input_tensors_without_none + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        output_grads = [None] * (len(ctx.input_tensors) + len(ctx.input_params))
        ii = 0
        for ix in range(len(ctx.input_tensors) + len(ctx.input_params)):
            if ix not in input_tensors_indices:
                output_grads[ix] = input_grads[ii]
                ii += 1
            else: output_grads[ix] = None

        output_grads = tuple(output_grads)
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + output_grads