"""Merge the five original measurement modes into the released JSON schema."""

import json
import math
from pathlib import Path

from ..data import load_config, sha256
from ..features import layer_features, layer_targets

MODES = "tlscp"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def number(value, label):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Expected finite nonnegative {label}, got {value}")
    return value


def indexed(rows, count, label):
    result = {}
    for row in rows:
        index = row[0]
        if not isinstance(index, int) or index in result or not 0 <= index < count:
            raise ValueError(f"Invalid or duplicate layer index in {label}: {index}")
        result[index] = row[1:]
    if set(result) != set(range(count)):
        raise ValueError(f"Missing layer indices in {label}")
    return result


def merge_measurements(logs):
    """Retain legacy keys/units, monotonic prefix correction, and energy formula."""
    if set(logs) != set(MODES):
        raise ValueError("All five measurement modes t, l, s, c, p are required")
    first = logs["t"]
    count = first["number of layers"]
    if not isinstance(count, int) or count < 1:
        raise ValueError("A measurement must contain at least one layer")
    for mode, log in logs.items():
        for key in ["model name", "input size", "number of layers"]:
            if log[key] != first[key]:
                raise ValueError(f"Measurement identity/settings mismatch in mode {mode}: {key}")
    total = first["data"]
    if len(total) != 1 or number(total[0], "total time") <= 0:
        raise ValueError("Expected one positive, independently measured total time")
    features = indexed(logs["l"]["data"], count, "layer measurements")
    layers = {}
    for index in range(count):
        if len(features[index]) < 4:
            raise ValueError("Missing structural features or profiler timings")
        kind, shapes, parameters, times = features[index][:4]
        if len(times) < 2:
            raise ValueError("Expected CPU and CUDA profiler timing channels")
        layer = [kind, shapes, parameters, [number(t, "profiler time") for t in times[:2]] + [0., 0.],
                 [0.], [0.] * 4]
        if kind:
            layer_features(layer)
        layers[str(index)] = layer
    for mode, channel in [("s", 2), ("c", 3)]:
        cumulative = indexed(logs[mode]["data"], count, mode)
        previous = 0.
        for index in range(count):
            value = max(previous, number(cumulative[index][0], "cumulative time"))
            if not layers[str(index)][0]:
                value = previous
            layers[str(index)][3][channel] = round(value - previous, 3)
            previous = value
    powers = indexed(logs["p"]["data"], count, "power measurements")
    total_energy = 0.
    for index in range(count):
        layer = layers[str(index)]
        watts = number(powers[index][0], "power")
        if watts <= 0:
            raise ValueError("Power readings must be positive; missing sensors cannot be replaced by zero")
        layer[4] = [watts]
        # Milliseconds times watts gives millijoules; preserve original rounding.
        layer[5] = [round(time_ms * watts, 3) for time_ms in layer[3]]
        total_energy += layer[3][2] * watts
        layer_targets(layer)
    return {"model name": first["model name"], "measurement type": list(MODES),
            "number of layers": count, "input size": [first["input size"]],
            "itterations": first["itterations"], "total": list(total),
            "total_energy": [round(total_energy, 3)], "layers": layers}


def merge_directory(raw_directory, destination):
    records, seen = {}, set()
    for total_path in sorted(Path(raw_directory).glob("*_t.json")):
        stem = total_path.name[:-7]
        logs = {mode: json.loads((total_path.parent / f"{stem}_{mode}.json").read_text()) for mode in MODES}
        record = merge_measurements(logs)
        key = record["model name"], tuple(record["input size"])
        if key in seen:
            raise ValueError(f"Duplicate model/input configuration: {key}")
        seen.add(key)
        records[str(len(records))] = record
    if not records:
        raise ValueError(f"No complete measurements found in {raw_directory}")
    write_json(destination, records)
    return records


def write_training_config(gpu, dataset_path, output_path):
    config = load_config()
    config["datasets"] = {gpu: {"file": f"{gpu}/all_logs.json", "sha256": sha256(dataset_path)}}
    write_json(output_path, config)
