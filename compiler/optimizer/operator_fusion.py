from __future__ import annotations

import torch

from compiler.ir.graph import Graph


def fuse_conv_batchnorm_relu(graph: Graph) -> int:
    """Fuse inference-only Conv2d -> BatchNorm2d -> ReLU chains."""
    fused = 0
    for conv in list(graph.topological_nodes()):
        if conv.operation != "Conv2d" or len(conv.outputs) != 1:
            continue
        conv_output = graph.tensors[conv.outputs[0]]
        if len(conv_output.consumers) != 1:
            continue
        batchnorm_id = next(iter(conv_output.consumers))
        batchnorm = graph.nodes[batchnorm_id]
        if batchnorm.operation != "BatchNorm2d" or len(batchnorm.outputs) != 1:
            continue
        batchnorm_output = graph.tensors[batchnorm.outputs[0]]
        if len(batchnorm_output.consumers) != 1:
            continue
        relu_id = next(iter(batchnorm_output.consumers))
        relu = graph.nodes[relu_id]
        if relu.operation != "ReLU" or relu_id not in graph.nodes:
            continue

        conv_weight = graph.tensors[conv.inputs[1]].value
        conv_bias = graph.tensors[conv.inputs[2]].value if len(conv.inputs) > 2 else torch.zeros(
            conv_weight.shape[0], dtype=conv_weight.dtype
        )
        bn_weight = graph.tensors[batchnorm.inputs[1]].value
        bn_bias = graph.tensors[batchnorm.inputs[2]].value
        running_mean = graph.tensors[batchnorm.inputs[3]].value
        running_var = graph.tensors[batchnorm.inputs[4]].value
        eps = batchnorm.parameters["eps"]
        scale = bn_weight / torch.sqrt(running_var + eps)
        fused_weight = conv_weight * scale.reshape(-1, 1, 1, 1)
        fused_bias = (conv_bias - running_mean) * scale + bn_bias

        weight_id = f"{conv.node_id}_fused_weight"
        bias_id = f"{conv.node_id}_fused_bias"
        graph.add_tensor(graph.tensors[conv.inputs[1]].__class__(
            tensor_id=weight_id, shape=tuple(fused_weight.shape),
            dtype=str(fused_weight.dtype).removeprefix("torch."),
            value=fused_weight, is_constant=True,
        ))
        graph.add_tensor(graph.tensors[conv.inputs[1]].__class__(
            tensor_id=bias_id, shape=tuple(fused_bias.shape),
            dtype=str(fused_bias.dtype).removeprefix("torch."),
            value=fused_bias, is_constant=True,
        ))
        graph.tensors[conv.inputs[1]].consumers.discard(conv.node_id)
        if len(conv.inputs) > 2:
            graph.tensors[conv.inputs[2]].consumers.discard(conv.node_id)
        graph.tensors[conv.inputs[0]].consumers.add(conv.node_id)
        conv.inputs = [conv.inputs[0], weight_id, bias_id]
        graph.tensors[weight_id].consumers.add(conv.node_id)
        graph.tensors[bias_id].consumers.add(conv.node_id)

        old_conv_output = conv.outputs[0]
        final_output = relu.outputs[0]
        graph.tensors[old_conv_output].consumers.discard(batchnorm_id)
        graph.tensors[batchnorm_output.tensor_id].consumers.discard(relu_id)
        graph.tensors[final_output].producer = conv.node_id
        conv.operation = "FusedConvBatchNormReLU"
        conv.outputs = [final_output]
        del graph.tensors[old_conv_output]
        del graph.tensors[batchnorm_output.tensor_id]
        del graph.nodes[batchnorm_id]
        del graph.nodes[relu_id]
        fused += 1
    graph.validate()
    return fused