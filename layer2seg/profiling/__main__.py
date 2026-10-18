"""Collect a compatible timing/power dataset on a new NVIDIA GPU."""

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import platform
import re

import torch

from ..data import ROOT, sha256
from .dataset import MODES, merge_directory, merge_measurements, write_json, write_training_config
from .measurements import collect_model, cuda_device
from .power import open_power


def model_cases(config):
    if not isinstance(config, dict):
        raise ValueError("Model configuration must be a JSON object")
    cases = []
    for name, spec in config.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"Use a simple model identifier without path separators: {name}")
        if not isinstance(spec, dict) or spec.get("source") not in {"torchvision", "factory"}:
            raise ValueError(f"Model {name} needs source torchvision or factory")
        if not isinstance(spec.get("kwargs", {}), dict):
            raise ValueError(f"Model kwargs must be a dictionary: {name}")
        sizes = spec.get("input_size", [224])
        if not isinstance(sizes, list) or not sizes or any(not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in sizes):
            raise ValueError(f"Model {name} needs a list of positive integer input sizes")
        if len(set(sizes)) != len(sizes):
            raise ValueError(f"Duplicate input sizes for {name}")
        cases.extend((name, size, spec) for size in sizes)
    if not cases:
        raise ValueError("The model configuration is empty")
    return cases


def load_model(name, spec):
    kwargs = dict(spec.get("kwargs", {}))
    if spec["source"] == "torchvision":
        try:
            import torchvision
        except ImportError as error:
            raise ValueError("Collection of torchvision models needs a torchvision build matching PyTorch") from error
        namespace = getattr(torchvision.models, spec["path"]) if "path" in spec else torchvision.models
        factory = getattr(namespace, name)
        kwargs.setdefault("weights", None)
        if name in {"googlenet", "inception_v3"}:
            kwargs.setdefault("init_weights", False)
    else:
        module, function = spec["factory"].split(":", 1)
        factory = getattr(importlib.import_module(module), function)
    model = factory(**kwargs)
    if not isinstance(model, torch.nn.Module):
        raise ValueError(f"Factory for {name} did not return a torch.nn.Module")
    return model


def collect(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.gpu):
        raise ValueError("Use a GPU identifier without spaces or path separators, e.g. MyGPU")
    output = args.output_root.expanduser().resolve() / args.gpu
    raw_directory = output / "raw"
    dataset_path = output / "all_logs.json"
    config_path = output / "train_config.json"
    metadata_path = output / "profiling_metadata.json"
    if args.merge_only:
        records = merge_directory(raw_directory, dataset_path)
        write_training_config(args.gpu, dataset_path, config_path)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            metadata["dataset_sha256"] = sha256(dataset_path)
            metadata["labels_rebuilt_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(metadata_path, metadata)
        print(f"Merged {len(records)} configurations: {dataset_path}", flush=True)
        print(f"Training configuration: {config_path}", flush=True)
        return
    if args.warmup < 0 or args.runs < 1 or args.power_interval <= 0 or args.threads < 1:
        raise ValueError("Warm-up must be nonnegative; runs, threads, and power interval must be positive")
    if not 0 <= args.seed < 65536:
        raise ValueError("Seeds must be between 0 and 65535")
    config = json.loads(args.models_config.read_text())
    cases = model_cases(config)
    device = cuda_device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    signature = {"gpu": args.gpu, "models": config, "warmup": args.warmup, "runs": args.runs,
                 "seed": args.seed, "device": str(device), "threads": args.threads,
                 "batch_size": 1, "dtype": "float32",
                 "allow_unsupported": args.allow_unsupported, "power_backend": args.power_backend,
                 "power_rail": args.power_rail, "power_interval": args.power_interval,
                 "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                 "gpu_name": torch.cuda.get_device_name(device), "machine": platform.machine()}
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            raise ValueError(f"Output already contains files: {output}; use --resume or a new output root")
        previous = json.loads(metadata_path.read_text())
        if previous["settings"] != signature:
            raise ValueError("Resume settings differ from the saved collection")
        metadata = previous
    else:
        metadata = {"settings": signature, "started_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "running", "completed": {}}
    output.mkdir(parents=True, exist_ok=True)
    raw_directory.mkdir(exist_ok=True)
    metadata["status"] = "running"
    metadata.pop("error", None)
    write_json(metadata_path, metadata)
    try:
        with open_power(args.power_backend, device.index, args.power_rail, args.power_interval) as reader:
            if "power" in metadata and metadata["power"] != reader.metadata:
                raise ValueError("Resume power sensor differs from the saved collection")
            metadata["power"] = reader.metadata
            write_json(metadata_path, metadata)
            print(f"GPU: {signature['gpu_name']}; power: {reader.metadata['scope']}", flush=True)
            for index, (name, size, spec) in enumerate(cases):
                key = f"{name}_{size}"
                if key in metadata["completed"]:
                    for mode in MODES:
                        path = raw_directory / f"{key}_{mode}.json"
                        if sha256(path) != metadata["completed"][key]["raw_sha256"][mode]:
                            raise ValueError(f"Completed raw measurement changed: {path}")
                    print(f"Reusing {key}", flush=True)
                    continue
                # Per-case seeds also keep fresh/resumed collection model initialization identical.
                torch.manual_seed(args.seed + index)
                model = load_model(name, spec)
                logs, layout = collect_model(model, name, size, device, args.warmup, args.runs,
                                             reader, args.allow_unsupported)
                merge_measurements(logs)  # Validate before saving any completed case.
                for mode, log in logs.items():
                    write_json(raw_directory / f"{key}_{mode}.json", log)
                metadata["completed"][key] = {"layers": layout,
                    "raw_sha256": {mode: sha256(raw_directory / f"{key}_{mode}.json") for mode in MODES}}
                write_json(metadata_path, metadata)
                del model
                torch.cuda.empty_cache()
            records = merge_directory(raw_directory, dataset_path)
            write_training_config(args.gpu, dataset_path, config_path)
            metadata["dataset_sha256"] = sha256(dataset_path)
            metadata["status"] = "complete"
            metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(metadata_path, metadata)
    except BaseException as error:
        metadata["status"], metadata["error"] = "failed", str(error)
        write_json(metadata_path, metadata)
        raise
    print(f"Saved {len(records)} configurations: {dataset_path}", flush=True)
    print(f"Training configuration: {config_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="Name for the new dataset, e.g. MyGPU")
    parser.add_argument("--models_config", type=Path, default=ROOT / "configs/profiling_models.json")
    parser.add_argument("--output_root", type=Path, default=ROOT / "runs/profiling")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--power_backend", choices=["auto", "torch", "jtop"], default="auto")
    parser.add_argument("--power_rail", default="auto", help="Jetson rail name, auto, or total")
    parser.add_argument("--power_interval", type=float, default=.01, help="Requested jtop refresh interval in seconds")
    parser.add_argument("--allow_unsupported", action="store_true", help="Keep unsupported executed modules as blank placeholders")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge_only", action="store_true", help="Rebuild labels from raw measurement JSONs; no GPU required")
    args = parser.parse_args()
    try:
        collect(args)
    except Exception as error:
        parser.exit(1, f"Profiling failed: {error}\n")


if __name__ == "__main__":
    main()
