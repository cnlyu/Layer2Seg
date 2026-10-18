"""Train LR and method-specific refinements using a shared layer MLP bank."""

import random
import numpy as np
import torch

from .checkpoints import layer_checkpoint_path, load_layer_checkpoint, method_path, save_method
from .data import family, sha256
from .features import LAYER_TYPES, MinMax
from .models import CumulativeMLP, SegmentRNN, SegmentTransformer
from .predict import predict_priors
from .train import fit, loaders, seed_everything, train_layer_bank, training_stage, training_windows


def metadata_for(dataset, config, seed):
    return {"gpu": dataset.gpu, "seed": seed, "dataset_sha256": dataset.sha256,
            "sequence_length": 7, "config": config}


def fit_ridge(features, targets, alpha):
    """Centered multi-output Ridge with training-only time/energy target scaling."""
    x, y = np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    time_scale = float(np.abs(y[:, :4]).mean() + 1e-5)
    energy_scale = float(np.abs(y[:, 4:]).mean() + 1e-5)
    y = y.copy()
    y[:, :4] /= time_scale
    y[:, 4:] /= energy_scale
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    centered = x - x_mean
    coefficients = np.linalg.solve(centered.T @ centered + alpha * np.eye(x.shape[1]),
                                   centered.T @ (y - y_mean)).T
    return {"coef": coefficients.tolist(), "intercept": (y_mean - x_mean @ coefficients.T).tolist(),
            "time_scale": time_scale, "energy_scale": energy_scale}


def train_lr(dataset, config, seed, destination):
    seed_everything(seed)
    groups = {index: [] for index in LAYER_TYPES.values()}
    for record in dataset.records:
        for feature, target in zip(record.features, record.targets):
            groups[feature[0]].append((record.name, feature, target))
    states, history = [], {}
    for kind, index in LAYER_TYPES.items():
        samples = list(groups[index])
        random.shuffle(samples)
        samples = [row for row in samples if family(row[0], config["held_out_families"]) is None]
        if len(samples) <= 10:
            states.append(None)
            continue
        order = np.random.permutation(len(samples))
        x = np.asarray([samples[i][1] for i in order], dtype=np.float64)
        y = np.asarray([samples[i][2] for i in order], dtype=np.float64)
        split = int(config["training_ratio"] * len(samples))
        scaler = MinMax.fit(x[:split])
        scaled = scaler.transform(x)
        state = fit_ridge(scaled[:split], y[:split], config.get("lr_alpha", 1.))
        state["scaler"] = scaler.to_dict()
        states.append(state)
        history[kind] = {"num_samples": len(samples), "training_samples": split,
                         "validation_samples": len(samples) - split}
        print(f"LR {kind}: {split} training, {len(samples) - split} validation layers", flush=True)
    save_method(destination, "LR", states, metadata_for(dataset, config, seed), history)


def cumulative_samples(dataset, mlps, scalers, config, device):
    features, targets = [], []
    for record in dataset.records:
        if family(record.name, config["held_out_families"]) is not None:
            continue
        raw, labels = record.features, record.targets
        priors = predict_priors(raw, mlps, scalers, device)
        cumulative, previous = [0., 0.], 0
        for index, (feature, prior, target) in enumerate(zip(raw, priors, labels)):
            # Measured previous cumulative targets are used only during training.
            features.append([index] + list(feature) + list(prior) + list(cumulative) + [previous])
            cumulative = [cumulative[0] + target[2], cumulative[1] + target[6]]
            targets.append(list(cumulative))
            previous = feature[0]
    return features, targets


def train_head(dataset, config, seed, device, bank, method, destination):
    seed_everything(seed)
    mlps, scalers, _ = load_layer_checkpoint(bank, device)
    if method == "MLP2":
        features, targets = cumulative_samples(dataset, mlps, scalers, config, device)
    else:
        features, targets = training_windows(dataset, mlps, scalers, config, device)
    samples = list(zip(features, targets))
    random.shuffle(samples)
    if len(samples) < 2:
        raise ValueError(f"Too few training samples for {method}")
    features, targets = zip(*samples)
    batch_size = config["rnn_batch_size"] if method == "Layer2Seg" else 512
    training, validation = loaders(features, targets, config["training_ratio"], batch_size,
                                   validation_shuffle=method == "MLP2")
    model = {"MLP2": CumulativeMLP, "Transformer": SegmentTransformer, "Layer2Seg": SegmentRNN}[method]()
    if method != "MLP2":
        model.initialize()
    model.to(device)
    scheduler, clip_norm = None, None
    if method == "MLP2":
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=.5)
        epochs = config.get("mlp2_epochs", 400)
    elif method == "Transformer":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        clip_norm, epochs = 1., config.get("transformer_epochs", 400)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["rnn_learning_rate"], weight_decay=config["rnn_weight_decay"])
        epochs = config["rnn_epochs"]
    history = fit(model, training, validation, optimizer, epochs, device, method,
                  scheduler=scheduler, clip_norm=clip_norm)
    metadata = {**metadata_for(dataset, config, seed), "layer_bank_sha256": sha256(bank),
                "training_samples": len(samples), "training_device": device}
    save_method(destination, method, model.state_dict(), metadata, history)
    print(f"Saved {method} refinement: {destination}", flush=True)


def train_methods(dataset, config, seed, device, output_root, methods):
    if "LR" in methods:
        destination = method_path(output_root, dataset.gpu, seed, "LR")
        with training_stage("LR"):
            train_lr(dataset, config, seed, destination)
    neural = [method for method in methods if method != "LR"]
    if not neural:
        return
    bank = layer_checkpoint_path(output_root, dataset.gpu, seed)
    if "MLP" in neural or not bank.exists():
        with training_stage("Layer MLPs (MLP)"):
            train_layer_bank(dataset, config, seed, device, bank)
    else:
        print(f"Reusing shared layer bank: {bank}", flush=True)
    mlps, scalers, metadata = load_layer_checkpoint(bank, "cpu")
    if any(metadata[key] != expected for key, expected in
           [("gpu", dataset.gpu), ("seed", seed), ("dataset_sha256", dataset.sha256)]):
        raise ValueError("Existing layer-bank identity differs from this training request")
    del mlps, scalers
    for method in neural:
        if method == "MLP":
            continue
        destination = method_path(output_root, dataset.gpu, seed, method)
        with training_stage(method):
            train_head(dataset, config, seed, device, bank, method, destination)
