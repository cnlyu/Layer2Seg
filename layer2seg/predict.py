"""Predict a model or contiguous segment selected from a released dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoints import layer_checkpoint_path, load_layer_checkpoint, load_method, method_path, resolve_device, select_methods
from .data import Dataset, ROOT, load_config, sha256
from .features import FALLBACK_PRIORS, MinMax, PRIMARY_INDICES, rnn_features


def load_refinement(path, method, bank, bank_metadata, device):
    model, metadata = load_method(path, method, device)
    if any(metadata[key] != bank_metadata[key] for key in ["gpu", "seed", "dataset_sha256", "sequence_length"]):
        raise ValueError("Refinement checkpoint and layer bank identities differ")
    if metadata["layer_bank_sha256"] != sha256(bank):
        raise ValueError("Refinement checkpoint was trained with a different layer bank")
    return model


class Layer2Seg:
    def __init__(self, layer_checkpoint, refinement_checkpoint, device="auto"):
        self.device = resolve_device(device)
        self.mlps, self.scalers, self.metadata = load_layer_checkpoint(layer_checkpoint, self.device)
        self.rnn = load_refinement(refinement_checkpoint, "Layer2Seg", layer_checkpoint, self.metadata, self.device)
        self.sequence_length = int(self.metadata["sequence_length"])
        if self.sequence_length != 7:
            raise ValueError("This release implements the main length-7 model")

    @torch.inference_mode()
    def layer_priors(self, features):
        return predict_priors(features, self.mlps, self.scalers, self.device)

    @torch.inference_mode()
    def predict(self, features, start=0, stop=None):
        """Return dense [time_us, energy_mJ] estimates for [start, stop).

        All inputs are structural features. No measured labels enter prediction.
        Windows restart at the requested segment, as in the research predictor.
        """
        stop = len(features) if stop is None else stop
        if not 0 <= start < stop <= len(features):
            raise ValueError("Require 0 <= start < stop <= number of supported layers")
        segment = features[start:stop]
        priors = self.layer_priors(segment)
        sequence = rnn_features(segment, priors)
        totals = [[0., 0.] for _ in segment]
        counts = [0] * len(segment)
        for end in range(len(segment)):
            begin = max(0, end - self.sequence_length + 1)
            window = sequence[begin:end + 1]
            window = [[0.] * 25 for _ in range(self.sequence_length - len(window))] + window
            values = self.rnn(torch.tensor([window], dtype=torch.float32, device=self.device)).tolist()[0]
            for dimension, prior_index in enumerate(PRIMARY_INDICES):
                weights = [max(float(priors[index][prior_index]), 0.) for index in range(begin, end + 1)]
                weight_sum = sum(weights)
                # Original allocation rule for a window with no positive layer priors.
                weights = [weight / weight_sum for weight in weights] if weight_sum > 0 else [1. / len(weights)] * len(weights)
                for index, weight in zip(range(begin, end + 1), weights):
                    totals[index][dimension] += float(values[dimension]) * weight
            for index in range(begin, end + 1):
                counts[index] += 1
        dense = np.asarray([[value / count for value in row] for row, count in zip(totals, counts)])
        if not np.isfinite(dense).all():
            raise ValueError("Nonfinite Layer2Seg predictions")
        return dense


class Predictor(Layer2Seg):
    """Dataset inference for the five methods, sharing one layer-MLP bank."""

    def __init__(self, root, gpu, seed, method="Layer2Seg", device="auto"):
        self.method = select_methods([method])[0]
        self.device = resolve_device(device)
        self.sequence_length = 7
        if self.method == "LR":
            self.device = "cpu"  # The linear baseline uses NumPy, not CUDA tensors.
            self.regressors, self.metadata = load_method(method_path(root, gpu, seed, "LR"), "LR", self.device)
        else:
            bank = layer_checkpoint_path(root, gpu, seed)
            self.mlps, self.scalers, self.metadata = load_layer_checkpoint(bank, self.device)
            if self.metadata["sequence_length"] != self.sequence_length:
                raise ValueError("This release implements the main length-7 model")
            head = method_path(root, gpu, seed, self.method)
            if self.method != "MLP":
                self.refiner = load_refinement(head, self.method, bank, self.metadata, self.device)
                if self.method in ["Transformer", "Layer2Seg"]:
                    self.rnn = self.refiner
        if self.metadata["gpu"] != gpu or self.metadata["seed"] != seed:
            raise ValueError("Checkpoint GPU/seed differs from the requested GPU/seed")

    @torch.inference_mode()
    def predict(self, features, start=0, stop=None):
        stop = len(features) if stop is None else stop
        if not 0 <= start < stop <= len(features):
            raise ValueError("Require 0 <= start < stop <= number of supported layers")
        if self.method in ["Layer2Seg", "Transformer"]:
            return super().predict(features, start, stop)
        segment = features[start:stop]
        if self.method == "LR":
            rows = [list(FALLBACK_PRIORS) for _ in segment]
            for kind in sorted({int(feature[0]) for feature in segment}):
                model = self.regressors[kind]
                if model is None:
                    continue
                indices = [i for i, feature in enumerate(segment) if feature[0] == kind]
                scaled = MinMax.from_dict(model["scaler"]).transform([segment[i] for i in indices])
                values = scaled @ np.asarray(model["coef"], dtype=np.float64).T + np.asarray(model["intercept"], dtype=np.float64)
                values[:, :4] *= model["time_scale"]
                values[:, 4:] *= model["energy_scale"]
                values = np.maximum(values, 0.).astype(np.float32)
                for index, value in zip(indices, values):
                    rows[index] = value.tolist()
            dense = np.asarray(rows, dtype=np.float64)[:, PRIMARY_INDICES]
        else:
            priors = self.layer_priors(segment)
            if self.method == "MLP":
                dense = np.asarray(priors, dtype=np.float64)[:, PRIMARY_INDICES]
            else:
                cumulative, previous, rows = [0., 0.], 0, []
                for index, (feature, prior) in enumerate(zip(segment, priors)):
                    values = [index] + list(feature) + list(prior) + cumulative + [previous]
                    predicted = self.refiner(torch.tensor(values, dtype=torch.float32, device=self.device)).tolist()
                    rows.append([a - b for a, b in zip(predicted, cumulative)])
                    cumulative, previous = predicted, feature[0]
                dense = np.asarray(rows, dtype=np.float64)
        if not np.isfinite(dense).all():
            raise ValueError(f"Nonfinite {self.method} predictions")
        return dense


@torch.inference_mode()
def predict_priors(features, mlps, scalers, device):
    priors = []
    for feature in features:
        kind = int(feature[0])
        if not 0 <= kind < len(mlps) or scalers[kind] is None:
            priors.append(list(FALLBACK_PRIORS))
            continue
        scaled = scalers[kind].transform([feature])
        prior = mlps[kind](torch.tensor(scaled, dtype=torch.float32, device=device)).tolist()[0]
        if len(prior) != 8 or not np.isfinite(prior).all():
            raise ValueError(f"Invalid MLP prediction for operator {kind}")
        priors.append(prior)
    return priors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="2080Ti", help="Target GPU dataset/checkpoint identifier")
    parser.add_argument("--model", required=True, help="Exact model name in the selected dataset")
    parser.add_argument("--input_size", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--method", default="Layer2Seg", help="LR, MLP, MLP2, Transformer, Layer2Seg, or all")
    parser.add_argument("--device", default="auto", help="Default: CUDA if available, otherwise CPU; accepts cpu or cuda:N")
    parser.add_argument("--checkpoint_root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--start", type=int, default=0, help="First supported-layer index, zero-based")
    parser.add_argument("--stop", type=int, help="Exclusive final supported-layer index")
    parser.add_argument("--output", type=Path, help="Optional JSON output; otherwise print to stdout")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        config = load_config(args.config)
        dataset = Dataset(args.gpu, config, args.data_root)
        record = dataset.find(args.model, args.input_size)
        features = record.features
        results = []
        for method in select_methods([args.method]):
            predictor = Predictor(args.checkpoint_root, args.gpu, args.seed, method, args.device)
            if predictor.metadata["dataset_sha256"] != dataset.sha256:
                raise ValueError("Checkpoint training dataset differs from the selected dataset")
            dense = predictor.predict(features, args.start, args.stop)
            total = [sum(row[index] for row in dense.tolist()) for index in range(2)]
            results.append({"gpu": args.gpu, "seed": args.seed, "method": method, "device": predictor.device, "model": record.name,
                            "input_size": record.input_size, "start": args.start,
                            "stop": len(features) if args.stop is None else args.stop,
                            "predicted_time_ms": total[0] / 1000., "predicted_energy_mj": total[1]})
        text = json.dumps(results[0] if len(results) == 1 else results, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        else:
            print(text, end="")
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(1, f"Prediction failed: {error}\n")


if __name__ == "__main__":
    main()
