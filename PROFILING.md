# Collect a new dataset

Profiling is optional: the downloaded data and checkpoints are ready to use.
To collect your own data, run from `release/` on Linux with an NVIDIA GPU and
CUDA-enabled PyTorch. Use a matching torchvision build.

## Collect timing and power

Desktop NVIDIA GPU (PyTorch 2.1 or newer):

```bash
python -m pip install -e '.[profiling]'
python -m layer2seg.profiling --gpu MyGPU
```

Jetson (requires a running jtop service):

```bash
python -m pip install -e '.[jetson]'
python -m layer2seg.profiling --gpu MyJetson
```

`--gpu` names the dataset. Collection uses `cuda:0` and 100 warm-up/100 measured
runs by default. Use `--device`, `--warmup`, or `--runs` to change these settings;
add `--resume` to continue an interrupted collection.

Choose models and input sizes in [configs/profiling_models.json](configs/profiling_models.json).
The example collects ResNet18 for training and MobileNetV2 for testing at size 224.

## Saved files

Files are saved under `runs/profiling/MyGPU/`:

- `all_logs.json`: dataset with timing, power, and derived energy labels.
- `train_config.json`: training and evaluation settings.
- `raw/` and `profiling_metadata.json`: measurements and collection progress.

See [DATASET.md](DATASET.md) for the format and units.

## Use the dataset

```bash
python -m layer2seg.train --gpu MyGPU --config runs/profiling/MyGPU/train_config.json --data_root runs/profiling
python -m layer2seg.evaluate --gpu MyGPU --config runs/profiling/MyGPU/train_config.json --data_root runs/profiling --checkpoint_root runs/checkpoints
```

Include both training models and held-out families. Edit `train_config.json`
to change the training settings or held-out families.
