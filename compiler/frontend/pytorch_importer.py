from __future__ import annotations

import torch
from torch import fx, nn

from compiler.ir.graph import Graph
from compiler.ir.node import Node
from compiler.ir.tensor import Tensor


_SUPPORTED_MODULES = {
    nn.Linear: "Linear",
    nn.ReLU: "ReLU",
    nn.Conv2d: "Conv2d",
    nn.BatchNorm2d: "BatchNorm2d",
    nn.MaxPool2d: "MaxPool2d",
    nn.AvgPool2d: "AvgPool2d",
    nn.Flatten: "Flatten",
}


def _tensor_meta(node: fx.Node) -> tuple[tuple[int, ...] | None, str | None]:
    meta = node.meta.get("tensor_meta")
    if meta is None:
        return None, None
    return tuple(meta.shape), str(meta.dtype).removeprefix("torch.")


def _target_name(target: object) -> str:
    return getattr(target, "__name__", str(target))


def import_model(model: nn.Module, example_input: torch.Tensor) -> Graph:
    """Import a traced PyTorch module into the project-owned IR."""
    model = model.eval()
    traced = fx.symbolic_trace(model)
    fx.passes.shape_prop.ShapeProp(traced).propagate(example_input)
    graph = Graph(name=model.__class__.__name__)
    modules = dict(traced.named_modules())

    for fx_node in traced.graph.nodes:
        shape, dtype = _tensor_meta(fx_node)
        if fx_node.op == "placeholder":
            tensor = Tensor(
                tensor_id=fx_node.name,
                shape=shape,
                dtype=dtype,
                is_graph_input=True,
            )
            graph.add_tensor(tensor)
            graph.inputs.append(fx_node.name)
            continue
        if fx_node.op == "get_attr":
            value = traced
            for part in str(fx_node.target).split("."):
                value = getattr(value, part)
            if not isinstance(value, torch.Tensor):
                raise NotImplementedError(f"Only tensor attributes are supported: {fx_node.target}")
            value = value.detach()
            graph.add_tensor(Tensor(
                tensor_id=fx_node.name,
                shape=tuple(value.shape),
                dtype=str(value.dtype).removeprefix("torch."),
                value=value,
                is_constant=True,
            ))
            continue
        if fx_node.op == "call_module":
            module = modules[str(fx_node.target)]
            operation = next((name for cls, name in _SUPPORTED_MODULES.items() if isinstance(module, cls)), None)
            if operation is None:
                raise NotImplementedError(f"Unsupported module: {module.__class__.__name__}")
            inputs = [str(arg.name) for arg in fx_node.args if isinstance(arg, fx.Node)]
            parameters = {}
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
                weight_id = f"{fx_node.name}_weight"
                bias_id = f"{fx_node.name}_bias"
                parameter_values = [(weight_id, module.weight), (bias_id, module.bias)]
                if isinstance(module, nn.BatchNorm2d):
                    parameter_values.extend([
                        (f"{fx_node.name}_running_mean", module.running_mean),
                        (f"{fx_node.name}_running_var", module.running_var),
                    ])
                for tensor_id, value in parameter_values:
                    if value is None:
                        continue
                    graph.add_tensor(Tensor(
                        tensor_id=tensor_id,
                        shape=tuple(value.shape),
                        dtype=str(value.dtype).removeprefix("torch."),
                        value=value.detach(),
                        is_constant=True,
                    ))
                    inputs.append(tensor_id)
                if isinstance(module, nn.Linear):
                    parameters.update(in_features=module.in_features, out_features=module.out_features)
                elif isinstance(module, nn.Conv2d):
                    parameters.update(
                        stride=module.stride, padding=module.padding,
                        dilation=module.dilation, groups=module.groups,
                    )
                else:
                    parameters.update(eps=module.eps, momentum=module.momentum)
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                parameters.update(
                    kernel_size=module.kernel_size, stride=module.stride,
                    padding=module.padding, dilation=getattr(module, "dilation", 1),
                    ceil_mode=module.ceil_mode,
                )
            elif isinstance(module, nn.Flatten):
                parameters.update(start_dim=module.start_dim, end_dim=module.end_dim)
            graph.add_node(Node(fx_node.name, operation, inputs, [fx_node.name], parameters=parameters))
            graph.tensors[fx_node.name].shape = shape
            graph.tensors[fx_node.name].dtype = dtype
            continue
        if fx_node.op == "call_function":
            target = _target_name(fx_node.target)
            operation = {
                "add": "Add", "mul": "Mul", "relu": "ReLU",
                "torch.add": "Add", "torch.mul": "Mul", "torch.relu": "ReLU",
                "linear": "Linear", "torch._C._nn.linear": "Linear",
            }.get(target)
            if operation is None:
                raise NotImplementedError(f"Unsupported function: {fx_node.target}")
            inputs = [str(arg.name) for arg in fx_node.args if isinstance(arg, fx.Node)]
            graph.add_node(Node(fx_node.name, operation, inputs, [fx_node.name]))
            graph.tensors[fx_node.name].shape = shape
            graph.tensors[fx_node.name].dtype = dtype
            continue
        if fx_node.op == "output":
            output_node = fx_node.args[0]
            output_nodes = output_node if isinstance(output_node, (tuple, list)) else (output_node,)
            for output in output_nodes:
                if not isinstance(output, fx.Node):
                    raise NotImplementedError("Only tensor outputs are supported")
                graph.outputs.append(output.name)
                graph.tensors[output.name].is_graph_output = True
            continue
        raise NotImplementedError(f"Unsupported FX node kind: {fx_node.op}")

    graph.validate()
    return graph
