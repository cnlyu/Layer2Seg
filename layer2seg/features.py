"""The fixed feature and measurement layouts used by the main method."""

import numpy as np

LAYER_TYPES = {
    "conv2d": 0, "linear": 1, "maxpool2d": 2, "avgpool2d": 3,
    "adaptiveavgpool2d": 4, "batchnorm2d": 5, "layernorm": 6,
    "layernorm2d": 7, "relu": 8, "gelu": 9, "silu": 10, "relu6": 11,
    "leakyrelu": 12, "softmax": 13, "softmax2d": 14, "tanh": 15,
    "sigmoid": 16, "hardsigmoid": 17, "hardswish": 18, "zeropad2d": 19,
    "dropout": 20, "flatten": 21, "unflatten": 22, "permute": 23,
    "identity": 24, "swapaxes": 25, "windowpartition": 26,
    "windowdepartition": 27, "stochasticdepth": 28, "upsample": 29,
    "concat": 30, "add": 31, "mul": 32, "split": 33, "squeeze": 34,
    "matmul": 35,
}
PRIMARY_INDICES = (2, 6)
# Original research prior for operators without a fitted training model.
FALLBACK_PRIORS = [10., 10., 10., 10., 0., 0., 0., 0.]


def layer_features(layer):
    """Read structural fields only; timing/energy labels are never inference inputs."""
    kind, shapes, parameters = layer[:3]
    if kind not in LAYER_TYPES:
        raise ValueError(f"Unsupported layer type: {kind!r}")
    inputs, outputs = shapes
    if len(inputs) > 4 or len(outputs) > 3 or len(parameters) > 8:
        raise ValueError(f"Feature dimensions exceed the released schema for {kind}")
    result = ([LAYER_TYPES[kind]] + [1] * (4 - len(inputs)) + list(inputs)
              + [1] * (3 - len(outputs)) + list(outputs)
              + list(parameters) + [0] * (8 - len(parameters)))
    if len(result) != 16 or not np.isfinite(result).all():
        raise ValueError(f"Invalid feature vector for {kind}")
    return result


def layer_targets(layer):
    """Four times in rounded microseconds, followed by four energies in mJ."""
    values = [round(1000 * value, 0) for value in layer[3]] + list(layer[5])
    if len(values) != 8 or not np.isfinite(values).all():
        raise ValueError("Expected four finite timing and four finite energy labels")
    return values


def rnn_features(features, priors):
    if len(features) != len(priors):
        raise ValueError("Layer features and MLP priors must have equal lengths")
    previous = 0
    result = []
    for feature, prior in zip(features, priors):
        if len(prior) != 8:
            raise ValueError("The main method requires eight MLP outputs")
        result.append(list(feature) + list(prior) + [previous])
        previous = feature[0]
    return result


class MinMax:
    """The main method's [0, 1] scaler, stored as plain numeric parameters.

    Applying x * scale + offset preserves the existing scikit-learn MinMaxScaler
    transformation, including its treatment of constant/near-constant columns.
    No scikit-learn objects are needed to load released checkpoints.
    """

    def __init__(self, scale, offset):
        self.scale = np.asarray(scale, dtype=np.float64)
        self.offset = np.asarray(offset, dtype=np.float64)
        if self.scale.shape != (16,) or self.offset.shape != (16,):
            raise ValueError("Expected 16 scaler coefficients")

    @classmethod
    def fit(cls, features):
        values = np.asarray(features, dtype=np.float64)
        lower = values.min(axis=0)
        span = values.max(axis=0) - lower
        span[span < 10 * np.finfo(values.dtype).eps] = 1.
        scale = 1. / span
        return cls(scale, -lower * scale)

    def transform(self, features):
        values = np.array(features, dtype=np.float64, copy=True)
        values *= self.scale
        values += self.offset
        return values

    def to_dict(self):
        return {"scale": self.scale.tolist(), "offset": self.offset.tolist()}

    @classmethod
    def from_dict(cls, values):
        return cls(values["scale"], values["offset"])
