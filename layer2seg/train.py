"""Train LR and the shared layer MLP bank with optional refinement models."""

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import random
import sys
from time import perf_counter
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .checkpoints import resolve_device, save_layer_checkpoint
from .data import Dataset, ROOT, family, load_config, select_gpus
from .features import LAYER_TYPES, MinMax, rnn_features
from .losses import ScaledL1
from .models import LayerMLP
from .predict import predict_priors


class _Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, text):
        self.terminal.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


@contextmanager
def training_log(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", buffering=1) as logfile:
        with redirect_stdout(_Tee(sys.stdout, logfile)), redirect_stderr(_Tee(sys.stderr, logfile)):
            try:
                yield
            except BaseException:
                traceback.print_exc(file=logfile)
                raise


def format_duration(seconds):
    hours, remaining = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remaining, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@contextmanager
def training_stage(label):
    started = perf_counter()
    print(f"\n[Stage: {label}] started at {datetime.now(timezone.utc).isoformat()}", flush=True)
    try:
        yield
    except BaseException:
        print(f"[Stage: {label}] stopped after {format_duration(perf_counter() - started)}", flush=True)
        raise
    else:
        print(f"[Stage: {label}] completed in {format_duration(perf_counter() - started)}", flush=True)


def seed_everything(seed):
    if not 0 <= seed < 65536:
        raise ValueError("Seeds must be between 0 and 65535")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def loaders(features, targets, ratio, batch_size, validation_shuffle=False):
    count = int(ratio * len(features))
    if not 0 < count < len(features):
        raise ValueError("Training and validation partitions must both be nonempty")
    x = torch.tensor(np.asarray(features), dtype=torch.float32)
    y = torch.tensor(np.asarray(targets), dtype=torch.float32)
    train = DataLoader(TensorDataset(x[:count], y[:count]), batch_size=batch_size, shuffle=True)
    validation = DataLoader(TensorDataset(x[count:], y[count:]),
                            batch_size=min(batch_size, len(x) - count), shuffle=validation_shuffle)
    return train, validation


def fit(model, training, validation, optimizer, epochs, device, label, scheduler=None, clip_norm=None):
    started = perf_counter()
    criterion = ScaledL1()
    history = {"train": [], "validation": []}
    if clip_norm is not None:
        history.update({"skipped_train_batches": [], "skipped_validation_batches": []})
    model.to(device)
    for epoch in range(epochs):
        epoch_started = perf_counter()
        model.train()
        training_loss = 0.
        train_batches = 0
        skipped_train = dict.fromkeys(["inputs", "outputs", "loss", "gradients", "gradient_norm"], 0)
        for features, targets in training:
            features, targets = features.to(device), targets.to(device)
            if clip_norm is not None and (not torch.isfinite(features).all() or not torch.isfinite(targets).all()):
                skipped_train["inputs"] += 1
                continue
            optimizer.zero_grad()
            outputs = model(features)
            if clip_norm is not None and not torch.isfinite(outputs).all():
                skipped_train["outputs"] += 1
                continue
            loss = criterion(outputs, targets)
            if clip_norm is not None and not torch.isfinite(loss):
                skipped_train["loss"] += 1
                continue
            loss.backward()
            if clip_norm is not None:
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                    optimizer.zero_grad(set_to_none=True)
                    skipped_train["gradients"] += 1
                    continue
                if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)):
                    optimizer.zero_grad(set_to_none=True)
                    skipped_train["gradient_norm"] += 1
                    continue
            optimizer.step()
            if clip_norm is not None and any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise ValueError("Transformer parameters became nonfinite after the optimizer step")
            training_loss += loss.item()
            train_batches += 1
        if scheduler is not None:
            scheduler.step()
        model.eval()
        validation_loss = 0.
        validation_batches = 0
        skipped_validation = dict.fromkeys(["inputs", "outputs", "loss"], 0)
        with torch.no_grad():
            for features, targets in validation:
                if clip_norm is not None and (not torch.isfinite(features).all() or not torch.isfinite(targets).all()):
                    skipped_validation["inputs"] += 1
                    continue
                outputs = model(features.to(device))
                if clip_norm is not None and not torch.isfinite(outputs).all():
                    skipped_validation["outputs"] += 1
                    continue
                loss = criterion(outputs, targets.to(device))
                if clip_norm is not None and not torch.isfinite(loss):
                    skipped_validation["loss"] += 1
                    continue
                validation_loss += loss.item()
                validation_batches += 1
        if clip_norm is None and (not train_batches or not validation_batches):
            raise ValueError(f"{label}: no finite training/validation batches")
        # Match the original Transformer loop: an all-skipped epoch is recorded as NaN and can recover later.
        train_loss = training_loss / train_batches if train_batches else float("nan")
        val_loss = validation_loss / validation_batches if validation_batches else float("nan")
        history["train"].append(train_loss)
        history["validation"].append(val_loss)
        if clip_norm is not None:
            history["skipped_train_batches"].append(skipped_train)
            history["skipped_validation_batches"].append(skipped_validation)
        now = perf_counter()
        print(f"{label}: epoch {epoch + 1}/{epochs}, train={train_loss:.6f}, validation={val_loss:.6f}, "
              f"epoch_time={now - epoch_started:.2f}s, elapsed={format_duration(now - started)}", flush=True)
    model.cpu()
    return history


def train_layers(dataset, config, seed, device):
    seed_everything(seed)
    mlps = [LayerMLP() for _ in LAYER_TYPES]
    scalers, history = [None] * len(mlps), {}
    # Match the original shuffle-before-family-filter procedure for each operator.
    by_kind = {kind: [] for kind in LAYER_TYPES.values()}
    for record in dataset.records:
        for feature, target in zip(record.features, record.targets):
            by_kind[feature[0]].append((record.name, feature, target))
    for kind, index in LAYER_TYPES.items():
        samples = list(by_kind[index])
        random.shuffle(samples)
        samples = [sample for sample in samples if family(sample[0], config["held_out_families"]) is None]
        if len(samples) <= 10:
            mlps[index].zero_parameters()
            history[kind] = {"train": [], "validation": [], "num_samples": len(samples), "fallback": True}
            print(f"Layer MLP ({kind}): {len(samples)} training-pool layers; retaining the main-method fallback", flush=True)
            continue
        features = [sample[1] for sample in samples]
        targets = [sample[2] for sample in samples]
        split = int(config["training_ratio"] * len(features))
        scalers[index] = MinMax.fit(features[:split])
        scaled = scalers[index].transform(features)
        batch_size = 32 if len(features) < 1000 else 256
        training, validation = loaders(scaled, targets, config["training_ratio"], batch_size, validation_shuffle=True)
        mlps[index].to(device)
        optimizer = torch.optim.AdamW(mlps[index].parameters(), lr=config["mlp_learning_rate"], weight_decay=.01)
        history[kind] = fit(mlps[index], training, validation, optimizer,
                            config["mlp_epochs"], device, f"Layer MLP ({kind})")
        history[kind]["num_samples"] = len(samples)
    return mlps, scalers, history


def training_windows(dataset, mlps, scalers, config, device):
    features, targets = [], []
    for model in mlps:
        model.eval().to(device)
    length = config["sequence_length"]
    for record in dataset.records:
        if family(record.name, config["held_out_families"]) is not None:
            continue
        raw = record.features
        sequence = rnn_features(raw, predict_priors(raw, mlps, scalers, device))
        labels = record.targets
        for start in range(len(sequence) - length + 1):
            features.append(sequence[start:start + length])
            targets.append([sum(row[2] for row in labels[start:start + length]),
                            sum(row[6] for row in labels[start:start + length])])
    for model in mlps:
        model.cpu()
    return features, targets


def train_layer_bank(dataset, config, seed, device, destination):
    device = resolve_device(device)
    if config["sequence_length"] != 7:
        raise ValueError("This release implements the main length-7 model")
    if not 0 < config["training_ratio"] < 1:
        raise ValueError("The training ratio must be between zero and one")
    if config["mlp_epochs"] < 1:
        raise ValueError("Layer training requires a positive epoch count")
    mlps, scalers, layer_history = train_layers(dataset, config, seed, device)
    metadata = {"gpu": dataset.gpu, "seed": seed, "sequence_length": config["sequence_length"],
                "dataset_sha256": dataset.sha256, "config": config,
                "training_device": device, "torch_version": str(torch.__version__),
                "numpy_version": str(np.__version__)}
    save_layer_checkpoint(destination, mlps, scalers, metadata, {"layers": layer_history})
    print(f"Saved shared layer MLP bank: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", nargs="+", default=["2080Ti"], help="Dataset GPU names, or all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024])
    parser.add_argument("--methods", nargs="+", default=["all"], help="LR, MLP, MLP2, Transformer, Layer2Seg, or all")
    parser.add_argument("--device", default="auto", help="Default: CUDA if available, otherwise CPU; accepts cpu or cuda:N")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--output_root", type=Path, default=ROOT / "runs/checkpoints")
    parser.add_argument("--log_root", type=Path, default=ROOT / "runs/logs", help="Training text logs, grouped by seed and GPU")
    parser.add_argument("--mlp_epochs", type=int, help="Override the main 400-epoch setting")
    parser.add_argument("--rnn_epochs", type=int, help="Override the main 400-epoch setting")
    parser.add_argument("--mlp2_epochs", type=int, help="Override the 400-epoch MLP2 setting")
    parser.add_argument("--transformer_epochs", type=int, help="Override the 400-epoch Transformer setting")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        config = load_config(args.config)
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError("Select distinct seeds")
        for key in ["mlp_epochs", "rnn_epochs", "mlp2_epochs", "transformer_epochs"]:
            value = getattr(args, key)
            if value is not None:
                if value < 1:
                    raise ValueError("Epoch counts must be positive")
                config[key] = value
        args.device = resolve_device(args.device)
        print(f"Training device: {args.device}", flush=True)
        from .training_baselines import train_methods
        from .checkpoints import select_methods
        methods = select_methods(args.methods)
        for gpu in select_gpus(config, args.gpu):
            dataset = Dataset(gpu, config, args.data_root)
            for seed in args.seeds:
                log_path = args.log_root / f"seed_{seed}" / f"{gpu}_train.txt"
                print(f"Training log: {log_path}", flush=True)
                with training_log(log_path):
                    started = perf_counter()
                    print(f"\nStarted: {datetime.now(timezone.utc).isoformat()}", flush=True)
                    print(f"GPU: {gpu}; seed: {seed}; device: {args.device}; methods: {', '.join(methods)}", flush=True)
                    print(f"Dataset: {dataset.path}", flush=True)
                    print(f"Checkpoint root: {args.output_root}", flush=True)
                    train_methods(dataset, config, seed, args.device, args.output_root, methods)
                    print(f"Training complete. Total elapsed: {format_duration(perf_counter() - started)}", flush=True)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
