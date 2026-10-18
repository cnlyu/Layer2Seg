"""Read the eight released datasets without modifying them."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .features import LAYER_TYPES, layer_features, layer_targets

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path=None):
    path = Path(path).expanduser().resolve() if path else ROOT / "configs/main.json"
    config = json.loads(path.read_text())
    return config


def select_gpus(config, names):
    selected = list(config["datasets"]) if names == ["all"] else list(names)
    if not selected or len(set(selected)) != len(selected) or any(name not in config["datasets"] for name in selected):
        raise ValueError(f"Select distinct GPUs from: {', '.join(config['datasets'])}, or all")
    return selected


def family(name, held_out):
    return next((item for item in held_out if item in name), None)


@dataclass
class ModelRecord:
    name: str
    input_size: list
    layers: list
    total_time_ms: float
    total_energy_mj: float

    @property
    def features(self):
        return [layer_features(layer) for layer in self.layers]

    @property
    def targets(self):
        return [layer_targets(layer) for layer in self.layers]


class Dataset:
    def __init__(self, gpu, config, data_root=None):
        entry = config["datasets"][gpu]
        base = Path(data_root).expanduser().resolve() if data_root else ROOT / "data"
        self.path = base / entry["file"]
        self.sha256 = sha256(self.path)
        if self.sha256 != entry["sha256"]:
            raise ValueError(f"Dataset checksum differs: {self.path}")
        raw = json.loads(self.path.read_text())
        self.gpu = gpu
        self.records = []
        seen = set()
        for record in raw.values():
            identity = (record["model name"], tuple(record["input size"]))
            if identity in seen:
                raise ValueError(f"Duplicate dataset model/input identity: {identity}")
            seen.add(identity)
            # JSON insertion order is the historical supported-layer order.
            layers = [layer for layer in record["layers"].values() if layer[0] in LAYER_TYPES]
            self.records.append(ModelRecord(identity[0], list(identity[1]), layers,
                                            float(record["total"][0]), float(record["total_energy"][0])))

    def find(self, name, input_size):
        for record in self.records:
            if record.name == name and record.input_size == list(input_size):
                return record
        raise ValueError(f"No {name} with input size {input_size} in the {self.gpu} dataset")

    def test_records(self, held_out):
        return [record for record in self.records if family(record.name, held_out) is not None]
