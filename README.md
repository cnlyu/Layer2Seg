# Layer2Seg

**Paper:** Layer2Seg: A Unified, Lightweight Predictor of Inference Time and Energy for On-the-Edge DNN Partitioning

**Authors:** Chengjin Lyu, Mohsen Nourazar, Bart Goossens  
**Affiliation:** IPI-imec, Ghent University, Ghent, Belgium

Training and evaluation for Layer2Seg, LR, MLP, MLP², and Transformer on eight GPU
datasets, with pretrained checkpoints for seeds 2024, 2025, and 2026.

## Download and setup

Use Python 3.10 or newer. Download `data.zip` and `checkpoints.zip` from
[Google Drive](https://drive.google.com/drive/folders/14BleJatRWWiMDucZMnYQw7mjL5KAj9qe?usp=sharing)
using the **Download** option. Place both archives in `release/` and run the following
commands from that directory:

```bash
unzip data.zip
unzip checkpoints.zip
python -m pip install -e .
```

After extraction, datasets are in `data/<gpu>/all_logs.json` and pretrained
checkpoints in `checkpoints/seed_<seed>/`. Keep the checkpoint folders intact.
Training and evaluation use CUDA when available, otherwise CPU. `--gpu` selects
the target GPU dataset; `--device` selects the device used to run the predictor.

## Evaluate pretrained checkpoints

Evaluate all eight GPU datasets and three seeds:

```bash
python -m layer2seg.evaluate --gpu all --seeds 2024 2025 2026 --checkpoint_root checkpoints
```

For one GPU and seed:

```bash
python -m layer2seg.evaluate --gpu 2080Ti --seeds 2024 --checkpoint_root checkpoints
```

All five methods are evaluated by default. Add `--methods Layer2Seg` to evaluate
only Layer2Seg.

Pretrained results across eight GPU datasets and seeds 2024, 2025, and 2026
(mean ± sample SD, %; lower is better):

| Method | Time MAPE | Time MdAPE | Energy MAPE | Energy MdAPE |
|---|---:|---:|---:|---:|
| LR | 94.85 ± 14.68 | 68.37 ± 11.31 | 167.19 ± 28.79 | 156.02 ± 30.40 |
| MLP | 34.10 ± 0.30 | 30.90 ± 0.29 | 30.66 ± 0.87 | 27.47 ± 1.01 |
| MLP² | 19.34 ± 1.10 | 14.83 ± 1.57 | 25.15 ± 2.86 | 16.14 ± 2.21 |
| Transformer | 39.67 ± 3.76 | 15.77 ± 2.13 | 29.21 ± 5.36 | 21.95 ± 6.54 |
| Layer2Seg | 13.43 ± 0.95 | 10.20 ± 0.17 | 15.36 ± 1.61 | 10.99 ± 1.48 |

## Retrain and evaluate

Train and evaluate all eight GPU datasets and three seeds:

```bash
python -m layer2seg.train --gpu all --seeds 2024 2025 2026

python -m layer2seg.evaluate --gpu all --seeds 2024 2025 2026 --checkpoint_root runs/checkpoints
```

For one GPU and seed:

```bash
python -m layer2seg.train --gpu 2080Ti --seeds 2024

python -m layer2seg.evaluate --gpu 2080Ti --seeds 2024 --checkpoint_root runs/checkpoints
```

Add `--methods Layer2Seg` to train and evaluate only Layer2Seg.
Training uses 400 epochs for neural models. Reruns overwrite the selected
checkpoints and training log. Supplied checkpoints retain their original training settings.

## Outputs

| Output | Location |
|---|---|
| New checkpoints | `runs/checkpoints/seed_<seed>/` |
| Training log | `runs/logs/seed_<seed>/<gpu>_train.txt` |
| Raw predictions | `runs/evaluation/seed_<seed>/<gpu>/` |
| Single-seed, per-GPU summaries | `runs/main/seed_<seed>/<gpu>/` |
| All-GPU, three-seed summaries | `runs/main/multiseed/seed_2024_2025_2026/` |

Start with `summary.csv` or `summary.txt`; GPU and DNN-family tables are alongside
them. Overall tables report MAPE and MdAPE with sample SD across seeds, using two
decimal places. To rebuild tables from saved predictions, add `--report_only`
to the evaluation command.

## Collect data on new hardware

Profiling is optional. See [PROFILING.md](PROFILING.md) to collect your own dataset,
[DATASET.md](DATASET.md) for its format and GPU identifiers, and
[configs/main.json](configs/main.json) for training settings.
Use `--help` with either command for more options.

Run checks with `python -m unittest discover -s tests`.

## License and citation

Code and released checkpoints are licensed under [MIT](LICENSE). The released
profiling dataset is licensed under [CC BY 4.0](DATA_LICENSE); credit Chengjin Lyu,
Mohsen Nourazar, and Bart Goossens and identify any changes when sharing it.

If you use this release in research, please cite the paper listed above.
