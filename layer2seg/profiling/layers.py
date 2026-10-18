"""Observe real module calls without replacing model weights or computation."""

from dataclasses import dataclass

import torch
from torch import nn

from ..features import LAYER_TYPES, layer_features


def scalar(value):
    # The historical compact encoding stores the first spatial component.
    return value[0] if isinstance(value, (tuple, list)) else value


def describe(module):
    if isinstance(module, nn.Conv2d):
        return "conv2d", [module.in_channels, module.out_channels, scalar(module.kernel_size),
                          scalar(module.stride), scalar(module.padding), scalar(module.dilation),
                          module.groups, int(module.bias is not None)]
    if isinstance(module, nn.Linear):
        return "linear", [module.in_features, module.out_features, int(module.bias is not None)]
    if isinstance(module, nn.MaxPool2d):
        return "maxpool2d", [scalar(module.kernel_size), scalar(module.stride or module.kernel_size),
                             scalar(module.padding), scalar(module.dilation)]
    if isinstance(module, nn.AvgPool2d):
        return "avgpool2d", [scalar(module.kernel_size), scalar(module.stride or module.kernel_size), scalar(module.padding)]
    if isinstance(module, nn.BatchNorm2d):
        return "batchnorm2d", [module.num_features, int(module.affine), int(module.track_running_stats)]
    if isinstance(module, nn.LayerNorm):
        kind = "layernorm2d" if "LayerNorm2d" in type(module).__name__ else "layernorm"
        return kind, list(module.normalized_shape) + [int(module.elementwise_affine)]
    if isinstance(module, nn.GELU):
        return "gelu", [int(module.approximate == "tanh")]
    if isinstance(module, nn.Softmax):
        return "softmax", [module.dim]
    if isinstance(module, nn.ZeroPad2d):
        return "zeropad2d", [scalar(module.padding)]
    if isinstance(module, nn.Upsample):
        return "upsample", [scalar(module.size) if module.size is not None else 0,
                            scalar(module.scale_factor) if module.scale_factor is not None else 0,
                            int(module.mode == "nearest")]
    for cls, kind in [(nn.LeakyReLU, "leakyrelu"), (nn.ReLU6, "relu6"), (nn.ReLU, "relu"),
                      (nn.SiLU, "silu"), (nn.Hardsigmoid, "hardsigmoid"),
                      (nn.Hardswish, "hardswish"), (nn.Dropout, "dropout")]:
        if isinstance(module, cls):
            return kind, [int(module.inplace)]
    for cls, kind in [(nn.AdaptiveAvgPool2d, "adaptiveavgpool2d"), (nn.Softmax2d, "softmax2d"),
                      (nn.Tanh, "tanh"), (nn.Sigmoid, "sigmoid"), (nn.Flatten, "flatten"),
                      (nn.Unflatten, "unflatten"), (nn.Identity, "identity")]:
        if isinstance(module, cls):
            return kind, []
    # Optional torchvision/custom modules need no torchvision import here.
    name = type(module).__name__
    if name == "FrozenBatchNorm2d":
        return "batchnorm2d", [module.weight.numel(), 1, 1]
    if name == "StochasticDepth":
        return "stochasticdepth", [int(module.mode == "row")]
    if name in {"Permute", "SwapAxes", "WindowPartition", "WindowDepartition"}:
        return name.lower(), []
    if name in {"Concat", "CustomConcat"}:
        return "concat", [module.d]
    kind = getattr(module, "name", None)
    if kind in LAYER_TYPES and hasattr(module, "prof_features"):
        return kind, list(module.prof_features)
    return "", []


def first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for child in value:
            result = first_tensor(child)
            if result is not None:
                return result
    return None


def leaf_modules(model):
    return [(name, module) for name, module in model.named_modules()
            if not list(module.children()) and not isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict))]


@dataclass
class Layer:
    name: str
    module: nn.Module
    kind: str
    shapes: list
    parameters: list


@torch.no_grad()
def discover(model, inputs, allow_unsupported=False):
    layers, handles = [], []

    def hook(name):
        def record(module, args, output):
            kind, parameters = describe(module)
            if not kind and not allow_unsupported:
                raise ValueError(f"Unsupported executed module {name}: {type(module).__name__}; use --allow_unsupported to record a placeholder")
            shapes = []
            if kind:
                incoming, outgoing = first_tensor(args), first_tensor(output)
                if incoming is None or outgoing is None:
                    raise ValueError(f"Expected tensor input/output for {name}")
                # Legacy logs omit the batch dimension; concat inputs are a tensor list.
                in_shape = list(incoming.shape) if args and isinstance(args[0], (list, tuple)) else list(incoming.shape[1:])
                shapes = [in_shape, list(outgoing.shape[1:])]
                layer_features([kind, shapes, parameters])
            layers.append(Layer(name, module, kind, shapes, parameters))
        return record

    try:
        for name, module in leaf_modules(model):
            handles.append(module.register_forward_hook(hook(name)))
        model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    if not layers or not any(layer.kind for layer in layers):
        raise ValueError("No supported module calls were observed")
    return layers


class PrefixComplete(Exception):
    """Only this exception is consumed when deliberately stopping a prefix."""


class TraceHooks:
    def __init__(self, layers, before=None, after=None, model=None):
        self.layers, self.before, self.after = layers, before, after
        self.model = model
        self.handles, self.stack = [], []
        self.position, self.stop = 0, None

    def __enter__(self):
        try:
            modules = (module for _, module in leaf_modules(self.model)) if self.model is not None else (layer.module for layer in self.layers)
            for module in dict.fromkeys(modules):
                self.handles.append(module.register_forward_pre_hook(self.pre))
                self.handles.append(module.register_forward_hook(self.post))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def pre(self, module, args):
        index = self.position
        if index >= len(self.layers) or module is not self.layers[index].module:
            raise ValueError("Module execution order changed between profiling runs")
        self.position += 1
        self.stack.append(index)
        if self.before:
            self.before(index, module, args)

    def post(self, module, args, output):
        index = self.stack.pop()
        if self.after:
            self.after(index, module, args, output)
        if index == self.stop:
            raise PrefixComplete()

    def run(self, model, inputs, stop=None):
        self.position, self.stop, self.stack = 0, stop, []
        completed = False
        try:
            result = model(inputs)
        except PrefixComplete:
            if stop is None:
                raise
            completed, result = True, None
        expected = len(self.layers) if stop is None else stop + 1
        if self.position != expected or (stop is not None and not completed):
            raise ValueError("Module execution count changed between profiling runs")
        return result
