"""Four timing channels and layer-boundary power sampling on the original model."""

import torch

from .layers import TraceHooks, discover


def cuda_device(value):
    device = torch.device(value)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Profiling requires an NVIDIA GPU and CUDA-enabled PyTorch; CPU fallback is only for predictors")
    index = device.index if device.index is not None else 0
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is unavailable")
    return torch.device(f"cuda:{index}")


def synchronize(device):
    torch.cuda.synchronize(device)


def warm_up(run, count, device):
    for _ in range(count):
        run()
        synchronize(device)


def timed(run, warmup, runs, device):
    warm_up(run, warmup, device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    synchronize(device)
    start.record()
    for _ in range(runs):
        run()
        synchronize(device)
    end.record()
    synchronize(device)
    return round(start.elapsed_time(end) / runs, 3)


def layer_timings(model, inputs, layers, warmup, runs, device):
    ranges = {}

    def before(index, module, args):
        annotation = torch.profiler.record_function(f"PP:{index}")
        ranges[index] = annotation
        annotation.__enter__()

    def after(index, module, args, output):
        ranges.pop(index).__exit__(None, None, None)

    try:
        with TraceHooks(layers, before, after, model=model) as hooks:
            schedule = torch.profiler.schedule(wait=1, warmup=warmup, active=runs, repeat=1)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                  torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as profiler:
                for _ in range(1 + warmup + runs):
                    hooks.run(model, inputs)
                    profiler.step()
            synchronize(device)
    finally:
        for annotation in reversed(list(ranges.values())):
            annotation.__exit__(None, None, None)
    timings = {}
    for event in profiler.key_averages():
        if event.key.startswith("PP:"):
            index = int(event.key[3:])
            if event.count != runs:
                raise ValueError(f"Layer {index} has {event.count} profiler calls; expected {runs}")
            gpu_time = event.device_time if hasattr(event, "device_time") else event.cuda_time
            timings[index] = [round(event.cpu_time / 1000., 3), round(gpu_time / 1000., 3)]
    if set(timings) != set(range(len(layers))):
        raise ValueError("PyTorch profiler did not capture every layer")
    return [[i, layer.kind, layer.shapes, layer.parameters, timings[i]] for i, layer in enumerate(layers)]


def prefix_timings(model, inputs, layers, warmup, runs, device):
    values = []
    with TraceHooks(layers, model=model) as hooks:
        for index in range(len(layers)):
            value = timed(lambda: hooks.run(model, inputs, stop=index), warmup, runs, device)
            values.append([index, value])
    return values


def cumulative_timings(model, inputs, layers, warmup, runs, device):
    events = [torch.cuda.Event(enable_timing=True) for _ in layers]
    start = torch.cuda.Event(enable_timing=True)
    sums = [0.] * len(layers)
    warm_up(lambda: model(inputs), warmup, device)
    with TraceHooks(layers, after=lambda index, *args: events[index].record(), model=model) as hooks:
        for _ in range(runs):
            start.record()
            hooks.run(model, inputs)
            synchronize(device)
            for index, event in enumerate(events):
                sums[index] += start.elapsed_time(event)
    return [[i, round(value / runs, 3)] for i, value in enumerate(sums)]


def layer_power(model, inputs, layers, warmup, runs, device, reader):
    sums = [0.] * len(layers)

    def after(index, *args):
        sums[index] += reader.read()

    warm_up(lambda: model(inputs), warmup, device)
    with TraceHooks(layers, after=after, model=model) as hooks:
        for _ in range(runs):
            hooks.run(model, inputs)
            synchronize(device)
    return [[i, round(value / runs, 3)] for i, value in enumerate(sums)]


def collect_model(model, name, size, device, warmup, runs, reader, allow_unsupported=False):
    device = cuda_device(device)
    with torch.cuda.device(device), torch.no_grad():
        return _collect_model(model, name, size, device, warmup, runs, reader, allow_unsupported)


def _collect_model(model, name, size, device, warmup, runs, reader, allow_unsupported):
    if warmup < 0 or runs < 1:
        raise ValueError("Warm-up must be nonnegative and runs must be positive")
    model.eval().to(device=device, dtype=torch.float32)
    inputs = torch.randn(1, 3, size, size, device=device)
    layers = discover(model, inputs, allow_unsupported)
    synchronize(device)
    header = {"model name": name, "input size": size, "number of layers": len(layers),
              "itterations": [1, warmup, runs, 1]}
    measurements = {}
    # Whole-model time is measured without hooks, independently of every prefix.
    functions = [("t", "total", lambda: [timed(lambda: model(inputs), warmup, runs, device)]),
                 ("l", "layers", lambda: layer_timings(model, inputs, layers, warmup, runs, device)),
                 ("s", "sequential", lambda: prefix_timings(model, inputs, layers, warmup, runs, device)),
                 ("c", "cumulative", lambda: cumulative_timings(model, inputs, layers, warmup, runs, device)),
                 ("p", "power", lambda: layer_power(model, inputs, layers, warmup, runs, device, reader))]
    for mode, label, function in functions:
        print(f"  {name}, input {size}: {label}", flush=True)
        measurements[mode] = {**header, "measurement type": label, "data": function()}
    layout = [{"index": i, "module": layer.name, "class": type(layer.module).__name__,
               "type": layer.kind} for i, layer in enumerate(layers)]
    return measurements, layout
