# type: ignore
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
"""
Loose implementation by @fmassa of SimpleFSDP (https://arxiv.org/abs/2411.00284)
Naive implem of FSDP using parametrizations, then all optimisations are done at compile-time
In practice, it's efficient."""

import contextlib
import operator
from typing import Any

import torch
import torch.distributed._functional_collectives
from xformers import get_python_lib

# Tweak if needed
PREFILL = 50  # 10
MAX_DISTANCE = 200  # 80
GROUP_SIZE = 50  # 10
K_REDUCE_SCATTER = 20  # 10


_LOCAL_STATE_DICT_MODE = False


class SimpleFSDP(torch.nn.Module):
    def __init__(
        self,
        module,
        process_group=None,
        compute_dtype=None,
        reduce_dtype=None,
        sync_module_states=True,
    ):
        super().__init__()
        _extend_once()
        self.module = module
        if process_group is None:
            process_group = torch.distributed.distributed_c10d._get_default_group()
        self.process_group = process_group
        if sync_module_states:
            self._broadcast_module()

        mods = list(self.module.modules())
        for mod in mods:
            params = list(mod._parameters.items())
            for name, p in params:
                if p is not None:
                    if p.numel() > 0:
                        torch.nn.utils.parametrize.register_parametrization(
                            mod,
                            name,
                            AllGatherParametrization(process_group, compute_dtype, reduce_dtype),
                            unsafe=True,
                        )
                    else:
                        # if condition is false we just cast empty tensors to avoid dtype promotions
                        torch.nn.utils.parametrize.register_parametrization(
                            mod,
                            name,
                            CastTensor(compute_dtype),
                            unsafe=True,
                        )
        self._register_state_dict_hook(consolidate_state_dict_hook)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.module, name)

    def _broadcast_module(self):
        module_states = list(self.module.parameters()) + list(self.module.buffers())
        module_states = [x.detach() for x in module_states]
        broadcast_bucket_size = int(250 * 1024 * 1024)
        if len(module_states) > 0:
            torch.distributed._broadcast_coalesced(self.process_group, module_states, broadcast_bucket_size, 0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


@contextlib.contextmanager
def local_state_dicts():
    global _LOCAL_STATE_DICT_MODE
    prev = _LOCAL_STATE_DICT_MODE
    _LOCAL_STATE_DICT_MODE = True
    yield
    _LOCAL_STATE_DICT_MODE = prev


def _batched_all_gather(data, world_size, group_name):
    flatten = [x.view(-1) for x in data]
    catted = torch.cat(flatten)
    catted = torch.ops._c10d_functional.all_gather_into_tensor(catted, world_size, group_name)
    param_numel = [x.numel() for x in data]

    catted = catted.view(world_size, -1)
    catted = catted.split(param_numel, 1)
    return [catted[i].view((world_size, *data[i].shape)) for i in range(len(data))]


def _batched_all_gather_meta(data, world_size, group_name):
    return [
        torch.empty((world_size, *data[i].shape), dtype=data[i].dtype, device=data[i].device) for i in range(len(data))
    ]


def _batched_all_gather_wait(catted, i, world_size):
    torch.ops._c10d_functional.wait_tensor(catted[0])
    return catted[i].flatten(0, 1)


def _batched_all_gather_wait_meta(catted, i, world_size):
    return catted[i].flatten(0, 1)


def _batched_reduce_scatter(data: list[torch.Tensor], world_size: int, group_name: str) -> list[torch.Tensor]:
    shapes = [(x.shape[0] // world_size,) + x.shape[1:] for x in data]
    numels = [x.numel() // world_size for x in data]

    r = [x.reshape(world_size, -1) for x in data]
    out: torch.Tensor = torch.cat(r, dim=1).flatten()

    # see why we use "avg" in the _AllGather Function
    out = torch.ops._c10d_functional.reduce_scatter_tensor(out, "avg", world_size, group_name)
    out = out.split(numels, 0)
    return [x.view(s) for x, s in zip(out, shapes, strict=False)]


def _batched_reduce_scatter_meta(data: list[torch.Tensor], world_size: int, group_name: str) -> list[torch.Tensor]:
    return [d.chunk(world_size)[0] for d in data]


def _batched_reduce_scatter_wait(catted, i):
    torch.ops._c10d_functional.wait_tensor(catted[0])
    return catted[i]


def _batched_reduce_scatter_wait_meta(catted, i):
    return catted[i]


_xformers_lib = get_python_lib()
_xformers_lib.define("batched_all_gather(Tensor[] t, int world_size, str group_name) -> Tensor[]")
_xformers_lib.impl("batched_all_gather", _batched_all_gather, "CUDA")
_xformers_lib.impl("batched_all_gather", _batched_all_gather_meta, "Meta")

_xformers_lib.define("batched_reduce_scatter(Tensor[] t, int world_size, str group_name) -> Tensor[]")
_xformers_lib.impl("batched_reduce_scatter", _batched_reduce_scatter, "CUDA")
_xformers_lib.impl("batched_reduce_scatter", _batched_reduce_scatter_meta, "Meta")

_xformers_lib.define("batched_all_gather_wait(Tensor[] t, int i, int world_size) -> Tensor")
_xformers_lib.impl("batched_all_gather_wait", _batched_all_gather_wait, "CUDA")
_xformers_lib.impl("batched_all_gather_wait", _batched_all_gather_wait_meta, "Meta")

_xformers_lib.define("batched_reduce_scatter_wait(Tensor[] t, int i) -> Tensor")
_xformers_lib.impl("batched_reduce_scatter_wait", _batched_reduce_scatter_wait, "CUDA")
_xformers_lib.impl("batched_reduce_scatter_wait", _batched_reduce_scatter_wait_meta, "Meta")


def _split_with_sizes(t, split_sizes):
    # this function is needed because torch.split_with_sizes_copy has both a fallback
    # and a decomp and fails if put directly in the graph. Plus, it has an out argument
    # which is also problematic
    world_size = t.shape[0]
    out = [torch.empty((world_size, s), dtype=t.dtype, device=t.device) for s in split_sizes]
    torch.split_with_sizes_copy(t, split_sizes, -1, out=out)
    return out


if torch.__version__ >= (2, 5):
    _xformers_lib.define("_split_with_sizes(Tensor t, int[] shape) -> Tensor[]", tags=torch.Tag.flexible_layout)
else:
    _xformers_lib.define("_split_with_sizes(Tensor t, int[] shape) -> Tensor[]")

_xformers_lib.impl("_split_with_sizes", _split_with_sizes, "CUDA")
_xformers_lib.impl("_split_with_sizes", _split_with_sizes, "Meta")


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group_size, group_name, forward_dtype, backward_dtype):
        out = torch.ops._c10d_functional.all_gather_into_tensor(x.to(dtype=forward_dtype), group_size, group_name)
        out = torch.ops._c10d_functional.wait_tensor(out)
        ctx.group_name = group_name
        ctx.group_size = group_size
        ctx.forward_dtype = forward_dtype
        ctx.backward_dtype = backward_dtype
        return out

    @staticmethod
    def backward(ctx, grad):
        # [NOTE] Why we use "avg" instead of sum
        # The original gradient of all_gather is actually a reduce_scatter
        # with a "sum" reduction mode. But we use "avg" to mimic the behavior
        # of PyTorch's FSDP. They do it because in practice, you'd want to perform
        # an all_reduce of the loss before backpropagation with an average over
        # world_size (which assumes the loss reduces over the batch size). Given
        # that PyTorch's FSDP doesn't assume the user does this all_reduce, they then
        # need to perform this division in the gradient computation.
        grad = torch.ops._c10d_functional.reduce_scatter_tensor(
            grad.to(ctx.backward_dtype),
            "avg",
            ctx.group_size,
            ctx.group_name,
        )
        grad = torch.ops._c10d_functional.wait_tensor(grad)
        return grad, None, None, None, None


def _all_gather(x, group_size, group_name, forward_dtype, backward_dtype):
    return _AllGather.apply(x, group_size, group_name, forward_dtype, backward_dtype)


class AllGatherParametrization(torch.nn.Module):
    def __init__(self, process_group, forward_dtype=None, backward_dtype=None):
        super().__init__()
        self.rank = process_group.rank()
        self.world_size = process_group.size()
        self.group_name = process_group.group_name
        self.shape_0 = None
        self.forward_dtype = forward_dtype
        self.backward_dtype = backward_dtype

    def forward(self, x):
        out = _all_gather(x, self.world_size, self.group_name, self.forward_dtype, self.backward_dtype)
        return out[: self.shape_0]

    def right_inverse(self, x):
        group_size = self.world_size
        self.shape_0 = x.shape[0]
        out = x.tensor_split(group_size, dim=0)[self.rank]
        expected_size = (self.shape_0 + group_size - 1) // group_size
        pad_value = expected_size - out.shape[0]
        if pad_value != 0:
            pad = [0] * (x.ndim * 2)
            pad[-1] = pad_value
            out = torch.nn.functional.pad(out, pad)
        else:
            out = out.clone()
        return out


class CastTensor(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        return x.to(self.dtype)


# conditional hook to get the full state dict by looking into the modules to finally execute the parameterization
# a bit as in pytorch-fsdp actually
def consolidate_state_dict_hook(self, state_dict, prefix, local_metadata):
    if _LOCAL_STATE_DICT_MODE:
        return state_dict

    keys = list(state_dict.keys())
    for k in keys:
        if not k.startswith(prefix):
            continue
        strpath = k.removeprefix(prefix)
        strpath = strpath.replace("parametrizations.", "").replace(".original", "").replace("module.", "")
        param = self.module
        for s in strpath.split("."):
            param = getattr(param, s)
        state_dict[prefix + strpath] = param
        del state_dict[k]

    return state_dict


#### COMPILATION
# A bit of help to the compiler to optimize the graph better
def _move_ops_to_top(graph, ops_to_top):
    def _get_order(graph):
        order = {}
        inv_order = {}
        for idx, node in enumerate(graph.nodes):
            order[node] = idx
            inv_order[idx] = node
        return order, inv_order

    non_placeholder_nodes = []
    the_ops = []
    for node in graph.nodes:
        if node.op == "call_function" and getattr(node.target, "overloadpacket", "") in ops_to_top:
            the_ops.append(node)
        elif node.op != "placeholder":
            non_placeholder_nodes.append(node)

    # move potential casts to the beginning of the graph
    # this needs to be improved
    for node in the_ops:
        inputs = node.all_input_nodes[0]
        if inputs.op != "placeholder":
            assert inputs.all_input_nodes[0].op == "placeholder"
            if inputs != non_placeholder_nodes[0]:
                non_placeholder_nodes[0].prepend(inputs)

    # prefill
    prefill = PREFILL
    prefill = min(prefill, len(the_ops))
    for _ in range(prefill):
        op = the_ops.pop(0)
        if op != non_placeholder_nodes[0]:
            non_placeholder_nodes[0].prepend(op)

    max_distance = MAX_DISTANCE
    group_size = GROUP_SIZE
    i = 0
    n2 = None
    while the_ops:
        n1 = the_ops.pop(0)
        order, inv_order = _get_order(graph)
        o_n1 = order[n1]
        if i % group_size == 0:
            new_location = o_n1 - max_distance + group_size - 1
            new_location = max(new_location, 0)
            n2 = inv_order[new_location]
        if n2 != n1:
            n2.prepend(n1)
        i += 1


def _move_ops_to_bottom(graph, ops_to_bottom):
    next_op_constraint = {
        torch.ops._c10d_functional.reduce_scatter_tensor,
    }
    non_placeholder_nodes = []
    the_ops = []
    for node in graph.nodes:
        if (
            node.op == "call_function"
            and getattr(node.target, "overloadpacket", "") in ops_to_bottom
            and getattr(node.args[0].target, "overloadpacket", "") in next_op_constraint
        ):
            the_ops.append(node)
        elif node.op not in ("placeholder", "output"):
            non_placeholder_nodes.append(node)

    while the_ops:
        op = the_ops.pop()
        if non_placeholder_nodes[-1]._next != op:
            non_placeholder_nodes[-1].append(op)


def fuse_all_gather(graph, ops_to_group):
    groups = [[]]
    last_n = -1

    for i, j in enumerate(list(graph.nodes)):
        if getattr(j.target, "overloadpacket", "") in ops_to_group:
            if last_n == i - 1:
                groups[-1].append(j)
            else:
                groups.append([j])
            last_n = i
    groups.pop(0)

    for group in groups:
        world_size = group[0].args[1]
        group_name = group[0].args[2]
        with graph.inserting_before(group[0]):
            inputs = [i.all_input_nodes[0] for i in group]
            # do batched all_gather
            out_flat = [
                graph.call_function(torch.ops.aten.view.default, args=(i, [i.meta["val"].numel()])) for i in inputs
            ]
            out_cat = graph.call_function(torch.ops.aten.cat.default, args=(out_flat,))
            out_ag = graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor,
                args=(out_cat, world_size, group_name),
            )

            new_size = sum([i.meta["val"].numel() for i in inputs])
            out_view = graph.call_function(torch.ops.aten.view.default, args=(out_ag, (world_size, new_size)))
            split_sizes = []
            split_shapes = []
            for i in inputs:
                split_sizes.append(i.meta["val"].numel())
                split_shapes.append(i.meta["val"].shape)
        # add new wait
        wait_ops = [list(i.users.keys())[0] for i in group]  # noqa: RUF015
        # replace old waits with indexing on the output
        for i, wait_op in enumerate(wait_ops):
            with graph.inserting_before(wait_op):
                if i == 0:
                    waited = graph.call_function(
                        torch.ops._c10d_functional.wait_tensor.default,
                        args=(out_view,),
                    )
                    # fixes AssertionError:
                    # both a fallback and a decomp for same op: aten.split_with_sizes_copy.default
                    res_split = graph.call_function(
                        torch.ops.xformers_python._split_with_sizes.default,
                        args=(waited, split_sizes),
                    )

                res_s_i = graph.call_function(operator.getitem, args=(res_split, i))

                new_shape = (world_size * split_shapes[i][0],) + split_shapes[i][1:]
                new_op = graph.call_function(torch.ops.aten.view.default, args=(res_s_i, new_shape))

            wait_op.replace_all_uses_with(new_op)
            new_op.meta.update(wait_op.meta)
            graph.erase_node(wait_op)

        for node in group:
            graph.erase_node(node)
    return graph


def fuse_reduce_scatter(graph, ops_to_group):
    groups = []
    counter = 0
    K = K_REDUCE_SCATTER

    for j in list(graph.nodes):
        if getattr(j.target, "overloadpacket", "") in ops_to_group:
            if counter % K == 0:
                groups.append([j])
            else:
                groups[-1].append(j)
            counter += 1

    for group in groups:
        world_size = group[0].args[2]
        group_name = group[0].args[3]
        with graph.inserting_before(group[-1]):
            inputs = [i.all_input_nodes[0] for i in group]
            # do batched all_gather
            out = graph.call_function(
                torch.ops.xformers_python.batched_reduce_scatter.default,
                args=(inputs, world_size, group_name),
            )
        # add new wait
        wait_ops = [list(i.users.keys())[0] for i in group]  # noqa: RUF015
        # replace old waits with indexing on the output
        for i, wait_op in enumerate(wait_ops):
            with graph.inserting_before(wait_op):
                new_op = graph.call_function(
                    torch.ops.xformers_python.batched_reduce_scatter_wait.default,
                    args=(out, i),
                )
            wait_op.replace_all_uses_with(new_op)
            new_op.meta.update(wait_op.meta)
            graph.erase_node(wait_op)

        for node in group:
            graph.erase_node(node)
    return graph


def _remove_no_ops(graph):
    def replace_no_op(node):
        replacement = node.args[0]
        if not all(isinstance(arg, torch.fx.Node) for arg in node.args):
            return
        node.replace_all_uses_with(replacement)
        replacement.meta.update(node.meta)
        graph.erase_node(node)

    for node in graph.nodes:
        if node.op != "call_function":
            continue

        if (
            node.target == torch.ops.aten.alias.default
            and len(node.users) == 1
            and list(node.users.keys())[0].target == torch.ops.aten.alias.default  # noqa: RUF015
        ):
            replace_no_op(node)


def post_grad_custom_post_pass(gm):
    _remove_no_ops(gm)

    ops_to_top = {torch.ops._c10d_functional.all_gather_into_tensor}
    _move_ops_to_top(gm, ops_to_top)
    gm = fuse_all_gather(gm, ops_to_top)

    ops_to_bottom = {torch.ops._c10d_functional.wait_tensor}
    _move_ops_to_bottom(gm, ops_to_bottom)

    ops_scatter = {
        torch.ops._c10d_functional.reduce_scatter_tensor,
    }
    gm = fuse_reduce_scatter(gm, ops_scatter)


EXTENDED = False


def _extend_once():
    global EXTENDED
    if not EXTENDED:
        _extend()
    EXTENDED = True


def _extend():
    from torch._inductor.fx_passes.joint_graph import patterns as joint_graph_patterns
    from torch._inductor.fx_passes.post_grad import pass_patterns as post_grad_patterns_all
    from torch._inductor.pattern_matcher import fwd_only, register_replacement

    post_grad_patterns = post_grad_patterns_all[1]  # medium priority
    # workaround https://github.com/pytorch/pytorch/issues/97894
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def randperm_index_pattern(rng_state_output, x, slice_shape):
        index = torch.ops.higher_order.run_with_rng_state(
            rng_state_output,
            torch.ops.aten.randperm.default,
            x.shape[0],
            device=x.device,
        )[:slice_shape]
        return torch.ops.aten.index(x, (index,)), index

    def randperm_index_replacement(rng_state_output, x, slice_shape):
        index = torch.ops.higher_order.run_with_rng_state(
            rng_state_output,
            torch.ops.aten.randperm.default,
            x.shape[0],
            device=x.device,
        )[:slice_shape]
        return torch.ops.aten._unsafe_index(x, (index,)), index

    rng_state = torch.cuda.get_rng_state()

    pattern = register_replacement(  # noqa: F841
        randperm_index_pattern,
        randperm_index_replacement,
        [rng_state, torch.empty(4, 8, device=device)],
        fwd_only,
        [post_grad_patterns, joint_graph_patterns],
        scalar_workaround={"slice_shape": 42},
    )


if True:
    torch._inductor.config.post_grad_custom_post_pass = post_grad_custom_post_pass
