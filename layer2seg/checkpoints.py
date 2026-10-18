"""Portable checkpoints contain tensors and plain Python metadata, not model objects."""

from pathlib import Path
import torch

from .features import LAYER_TYPES, MinMax
from .models import CumulativeMLP, LayerMLP, SegmentRNN, SegmentTransformer

METHODS = ["LR", "MLP", "MLP2", "Transformer", "Layer2Seg"]


def select_methods(names):
    if names == ["all"]:
        return list(METHODS)
    aliases = {name.lower(): name for name in METHODS}
    aliases.update({"mlp^2": "MLP2", "mlp²": "MLP2"})
    try:
        result = [aliases[name.lower()] for name in names]
    except KeyError as error:
        raise ValueError(f"Select methods from {', '.join(METHODS)}, or all") from error
    if not result or len(set(result)) != len(result):
        raise ValueError("Select distinct methods")
    return result


def resolve_device(device="auto"):
    """Prefer the first CUDA device; use CPU when CUDA is unavailable."""
    if device is None or str(device) == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        selected = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError("Select --device auto, cpu, or cuda:N") from error
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            return "cpu"
        return f"cuda:{selected.index if selected.index is not None else 0}"
    if selected.type != "cpu":
        raise ValueError("Select --device auto, cpu, or cuda:N")
    return str(selected)


def save_layer_checkpoint(path, mlps, scalers, metadata, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "kind": "layer_mlps",
        "mlps": [{name: tensor.detach().cpu() for name, tensor in model.state_dict().items()} for model in mlps],
        "scalers": [scaler.to_dict() if scaler is not None else None for scaler in scalers],
        "metadata": metadata, "history": history,
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_layer_checkpoint(path, device="auto"):
    device = resolve_device(device)
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload["format_version"] != 2 or payload.get("kind") != "layer_mlps" or len(payload["mlps"]) != len(LAYER_TYPES):
        raise ValueError("Unsupported layer checkpoint format or MLP bank size")
    if len(payload["scalers"]) != len(LAYER_TYPES):
        raise ValueError("Checkpoint scaler count differs from the MLP bank size")
    # Constructing modules for loading must not perturb a caller's training RNG.
    with torch.random.fork_rng(devices=[]):
        mlps = [LayerMLP() for _ in LAYER_TYPES]
    for model, state in zip(mlps, payload["mlps"]):
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
    scalers = [MinMax.from_dict(item) if item is not None else None for item in payload["scalers"]]
    return mlps, scalers, payload["metadata"]


def layer_checkpoint_path(root, gpu, seed):
    return Path(root).expanduser().resolve() / f"seed_{seed}" / f"{gpu}.layers.pt"


def method_path(root, gpu, seed, method):
    return Path(root).expanduser().resolve() / f"seed_{seed}" / f"{gpu}.{method.lower()}.pt"


def save_method(path, method, state, metadata, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 1, "method": method, "state": state,
               "metadata": metadata, "history": history}
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_method(path, method, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format_version"] != 1 or payload["method"] != method:
        raise ValueError("Checkpoint method/format differs from the requested predictor")
    if method == "LR":
        return payload["state"], payload["metadata"]
    with torch.random.fork_rng(devices=[]):
        model = {"MLP2": CumulativeMLP, "Transformer": SegmentTransformer, "Layer2Seg": SegmentRNN}[method]()
    model.load_state_dict(payload["state"], strict=True)
    model.eval().to(device)
    return model, payload["metadata"]
