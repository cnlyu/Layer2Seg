# Dataset

The released datasets contain 4,081 GPU/model/input configurations, including
545 held-out cases per seed. See [README.md](README.md) for download and usage.
Use the GPU identifiers below with `--gpu`, or use `--gpu all`.

| GPU identifier | Configurations | Held-out |
|---|---:|---:|
| `2080Ti` | 626 | 87 |
| `3080Ti` | 630 | 89 |
| `4060Ti` | 620 | 89 |
| `A4000` | 330 | 89 |
| `A6000` | 620 | 89 |
| `Jetson_Orin_AGX` | 437 | 37 |
| `Jetson_Orin_Nano` | 334 | 15 |
| `Jetson_Xavier_NX` | 484 | 50 |

## Format

Each `data/<gpu>/all_logs.json` contains records with `model name`, `input size`,
ordered `layers`, whole-model timing `total`, and energy estimate `total_energy`.
A layer record contains:

```text
[type, [input_shape, output_shape], parameters, times, powers, energies]
```

Times are in milliseconds, power in watts, and energy in millijoules. Model time
uses the separately measured `total`; `total_energy` is a time/power-derived estimate.
Predictor inputs use only layer types, shapes, and parameters. The encoding is in
[features.py](layer2seg/features.py); file checksums are in `data/manifest.json`.

## Training and evaluation

Each GPU has its own predictors. EfficientNet, MobileNet, ShuffleNet, MnasNet,
GoogLeNet, and Inception are held out for testing. Remaining samples use a 90/10
training/validation split.

MAPE gives families equal weight within GPU and GPUs equal weight overall. MdAPE
pools case errors within the reported GPU, family, or overall scope. Metrics are
computed separately per seed, then summarized across seeds.
