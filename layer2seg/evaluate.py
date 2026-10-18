"""Evaluate held-out DNN families, with CSV outputs and no plotting dependencies."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

from .checkpoints import resolve_device, select_methods
from .data import Dataset, ROOT, family, load_config, select_gpus
from .predict import Predictor

MODEL_FIELDS = ["gpu", "seed", "method", "family", "model", "input_size", "num_layers",
                "measured_time_ms", "predicted_time_ms", "time_ape_percent",
                "measured_energy_mj", "predicted_energy_mj", "energy_ape_percent"]
LAYER_FIELDS = ["gpu", "seed", "method", "family", "model", "input_size", "layer_index",
                "measured_time_ms", "predicted_time_ms", "measured_energy_mj", "predicted_energy_mj"]


METRICS = ["time_mape_percent", "time_mdape_percent", "energy_mape_percent", "energy_mdape_percent"]
PROTOCOL = "gpu_family_balanced_mape_pooled_mdape_v1"
GPU_DISPLAY = {
    "2080Ti": "RTX 2080 Ti", "3080Ti": "RTX 3080 Ti", "4060Ti": "RTX 4060 Ti",
    "A4000": "RTX A4000", "A6000": "RTX A6000", "Jetson_Orin_AGX": "Jetson AGX Orin",
    "Jetson_Orin_Nano": "Jetson Orin Nano", "Jetson_Xavier_NX": "Jetson Xavier NX",
}
FAMILY_DISPLAY = {
    "efficientnet": "EfficientNets", "mobilenet": "MobileNets", "shufflenet": "ShuffleNets",
    "mnasnet": "MnasNets", "googlenet": "GoogLeNets", "inception": "Inceptions",
}
METHOD_DISPLAY = {"MLP2": "MLP²"}


def write_csv(path, rows, fields, rounded=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.2f}" if rounded and isinstance(value, float) else value
                             for key, value in row.items()})


def ape(predicted, measured):
    return 100 * abs(predicted - measured) / measured if measured > 0 else None


def summarize(rows, families=None):
    """Keep case APEs at full precision when computing means and medians."""
    if families is not None and {row["family"] for row in rows} != set(families):
        raise ValueError("Incomplete GPU-family coverage for balanced MAPE")
    result = {"num_cases": len(rows)}
    for metric in ["time", "energy"]:
        errors = [row[f"{metric}_ape_percent"] for row in rows
                  if row[f"{metric}_ape_percent"] is not None]
        result[f"{metric}_num_valid_cases"] = len(errors)
        result[f"{metric}_mape_percent"] = statistics.mean(errors) if errors else None
        if families is not None:
            groups = [[row[f"{metric}_ape_percent"] for row in rows
                       if row["family"] == name and row[f"{metric}_ape_percent"] is not None] for name in families]
            if any(not values for values in groups):
                raise ValueError("Incomplete GPU-family coverage for balanced MAPE")
            result[f"{metric}_mape_percent"] = statistics.mean(statistics.mean(values) for values in groups)
        result[f"{metric}_mdape_percent"] = statistics.median(errors) if errors else None
    return result


def aggregate(rows, gpu, group):
    """Aggregate case statistics computed separately within each predictor seed."""
    counts = {row["num_cases"] for row in rows}
    if len(counts) != 1:
        raise ValueError("Model/input counts differ across predictor seeds")
    result = {"gpu": gpu, "family": group, "num_seeds": len(rows), "num_cases_per_seed": counts.pop()}
    for metric in METRICS:
        values = [row[metric] for row in rows if row[metric] is not None]
        result[f"{metric}_mean"] = statistics.mean(values) if values else None
        result[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else None
    return result


def group_tables(models, config, gpus, seeds):
    """GPU-family-balanced MAPE; pooled instance MdAPE at each reporting scope."""
    per_seed, by_gpu, by_dnn = [], [], []
    statistics_by_group = {}
    families = config["held_out_families"]
    for gpu in gpus:
        for seed in seeds:
            cases = models[gpu, seed]
            for group in families + ["Average"]:
                selected = cases if group == "Average" else [row for row in cases if row["family"] == group]
                if selected:
                    row = {"gpu": gpu, "seed": seed, "family": group,
                           **summarize(selected, families if group == "Average" else None)}
                    per_seed.append(row)
                    statistics_by_group[gpu, group, seed] = row
        for group in families + ["Average"]:
            values = [statistics_by_group[gpu, group, seed] for seed in seeds
                      if (gpu, group, seed) in statistics_by_group]
            if values:
                if len(values) != len(seeds):
                    raise ValueError(f"Family coverage differs across seeds: {gpu}/{group}")
                by_gpu.append(aggregate(values, gpu, group))
    for group in families:
        for gpu in gpus:
            values = [statistics_by_group[gpu, group, seed] for seed in seeds
                      if (gpu, group, seed) in statistics_by_group]
            if values:
                by_dnn.append(aggregate(values, gpu, group))
        balanced = []
        for seed in seeds:
            selected = [statistics_by_group[gpu, group, seed] for gpu in gpus
                        if (gpu, group, seed) in statistics_by_group]
            if selected:
                row = summarize([value for gpu in gpus for value in models[gpu, seed] if value["family"] == group])
                for metric in METRICS:
                    if "mdape" in metric:
                        continue
                    values = [value[metric] for value in selected if value[metric] is not None]
                    row[metric] = statistics.mean(values) if values else None
                balanced.append(row)
        if balanced:
            by_dnn.append(aggregate(balanced, "Average", group))
    if not per_seed:
        raise ValueError("No held-out configurations were found")
    return per_seed, by_gpu, by_dnn


def overall_summary(per_seed, gpus, seeds, methods, models):
    """Balanced MAPE and pooled-instance MdAPE, computed within each seed first."""
    lookup = {(row["gpu"], row["seed"], row["method"]): row
              for row in per_seed if row["family"] == "Average"}
    rows = []
    for method in methods:
        row = {"method": METHOD_DISPLAY.get(method, method)}
        for target in ["time", "energy"]:
            values = []
            for seed in seeds:
                gpu_values = [lookup[gpu, seed, method][f"{target}_mape_percent"] for gpu in gpus]
                gpu_values = [value for value in gpu_values if value is not None]
                if gpu_values:
                    values.append(statistics.mean(gpu_values))
            row[f"{target}_mape_mean"] = statistics.mean(values) if values else None
            row[f"{target}_mape_sd"] = statistics.stdev(values) if len(values) > 1 else None
            medians = []
            for seed in seeds:
                errors = [case[f"{target}_ape_percent"] for gpu in gpus
                          for case in (models[gpu, seed, method] if (gpu, seed, method) in models else models[gpu, seed])
                          if case[f"{target}_ape_percent"] is not None]
                if errors:
                    medians.append(statistics.median(errors))
            row[f"{target}_mdape_mean"] = statistics.mean(medians) if medians else None
            row[f"{target}_mdape_sd"] = statistics.stdev(medians) if len(medians) > 1 else None
        rows.append(row)
    return rows


def method_comparison_summary(rows, gpus, seeds):
    columns = [(target, metric) for target in ["time", "energy"]
               for metric in ["mape_mean", "mape_sd", "mdape_mean", "mdape_sd"]]
    table = [["Method"] + [f"{target} {metric} (%)" for target in ["Time", "Energy"]
                           for metric in ["MAPE", "MAPE SD", "MdAPE", "MdAPE SD"]]]
    table.extend([row["method"]] + [f"{row[f'{target}_{metric}']:.2f}" if row[f"{target}_{metric}"] is not None else "N/A"
                                   for target, metric in columns] for row in rows)
    widths = [max(len(row[column]) for row in table) for column in range(len(table[0]))]
    lines = ["Method comparison (lower error is better)",
             f"GPUs: {', '.join(gpus)}; seeds: {', '.join(map(str, seeds))}"]
    if len(seeds) > 1:
        lines.append("MAPE and MdAPE are means across seeds; SD columns are sample SDs across seeds.")
    lines.append("")
    for index, row in enumerate(table):
        lines.append("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip())
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def wide_table(statistics_rows, group, heading, display_names, methods, metric):
    """Mean seed-level MAPE or MdAPE, with Time and Energy column blocks."""
    lookup = {(row[group], row["method"]): row for row in statistics_rows}
    present = list(dict.fromkeys(row[group] for row in statistics_rows))
    groups = [name for name in display_names if name in present]
    groups += [name for name in present if name not in display_names]
    rows = []
    for name in groups:
        row = {heading: display_names.get(name, name)}
        for target, label in [("time", "Time"), ("energy", "Energy")]:
            for method in methods:
                source = lookup[name, method]
                if metric == "mape":
                    for stat, title in [("mean", "Mean"), ("sd", "SD")]:
                        row[f"{label} - {METHOD_DISPLAY.get(method, method)} {title}"] = source[f"{target}_{metric}_percent_{stat}"]
                else:
                    row[f"{label} - {METHOD_DISPLAY.get(method, method)}"] = source[f"{target}_{metric}_percent_mean"]
        rows.append(row)
    return rows


def save_reports(models, config, gpus, seeds, destination, device, methods=None):
    methods = methods or ["Layer2Seg"]
    per_seed, by_gpu, by_dnn = [], [], []
    for method in methods:
        view = {(gpu, seed): models[gpu, seed, method] if (gpu, seed, method) in models else models[gpu, seed]
                for gpu in gpus for seed in seeds}
        tables = group_tables(view, config, gpus, seeds)
        for output, rows in zip([per_seed, by_gpu, by_dnn], tables):
            output.extend({"gpu": row["gpu"], "family": row["family"], "method": method,
                           **{key: value for key, value in row.items() if key not in ("gpu", "family")}}
                          for row in rows)
    family_order = {name: index for index, name in enumerate(config["held_out_families"] + ["Average"])}
    gpu_order = {name: index for index, name in enumerate(gpus + ["Average"])}
    method_order = {name: index for index, name in enumerate(methods)}
    per_seed.sort(key=lambda row: (gpu_order[row["gpu"]], seeds.index(row["seed"]), family_order[row["family"]], method_order[row["method"]]))
    by_gpu.sort(key=lambda row: (gpu_order[row["gpu"]], family_order[row["family"]], method_order[row["method"]]))
    by_dnn.sort(key=lambda row: (family_order[row["family"]], gpu_order[row["gpu"]], method_order[row["method"]]))
    write_csv(destination / "per_seed_metrics.csv", per_seed, list(per_seed[0]), rounded=True)
    overall = overall_summary(per_seed, gpus, seeds, methods, models)
    write_csv(destination / "summary.csv", overall, list(overall[0]), rounded=True)
    gpu_totals = [row for row in by_gpu if row["family"] == "Average"]
    family_totals = [row for row in by_dnn if row["gpu"] == "Average"]
    for metric in ["mape", "mdape"]:
        suffix = "" if metric == "mape" else "_mdape"
        for name, rows, group, heading, labels in [
            ("gpu", gpu_totals, "gpu", "GPU Platform", GPU_DISPLAY),
            ("dnn", family_totals, "family", "DNN Family", FAMILY_DISPLAY),
        ]:
            table = wide_table(rows, group, heading, labels, methods, metric)
            if metric == "mape":
                average = {heading: "Average"}
                for target, label in [("time", "Time"), ("energy", "Energy")]:
                    for row in overall:
                        for stat, title in [("mean", "Mean"), ("sd", "SD")]:
                            average[f"{label} - {row['method']} {title}"] = row[f"{target}_mape_{stat}"]
                table.append(average)
            write_csv(destination / f"summary{suffix}_by_{name}.csv", table, list(table[0]), rounded=True)
    case_counts = {row["gpu"]: row["num_cases"] for row in per_seed
                   if row["seed"] == seeds[0] and row["method"] == methods[0] and row["family"] == "Average"}
    metadata = {
        "metric_protocol": PROTOCOL, "source_files": config.get("report_source_files", []),
        "gpus": gpus, "seeds": seeds, "device": device, "methods": methods, "held_out_families": config["held_out_families"],
        "num_seeds": len(seeds), "num_gpus": len(gpus), "num_cases_per_seed": sum(case_counts.values()),
        "cases_per_gpu_per_seed": case_counts,
        "aggregation": "Within each GPU/family/seed, compute MAPE and MdAPE across model/input cases. Report the mean and sample SD of these statistics across seeds; no pooled seed median.",
        "overall_mape": "Mean case APE within GPU/family; equal family weight within GPU and equal GPU weight overall per seed. summary.csv reports mean/sample SD of seed-level balanced means.",
        "overall_mdape": "Median of all selected GPUs' case APEs pooled within each seed, then mean and sample SD of seed medians; no pooling across seeds or averaging GPU medians. summary.csv reports time_mdape_mean/sd and energy_mdape_mean/sd.",
        "wide_tables": "summary_by_gpu.csv and summary_by_dnn.csv contain separate Mean and sample SD columns across seed-level MAPE percentages. summary_mdape_by_gpu.csv and summary_mdape_by_dnn.csv contain mean seed-level MdAPE percentages. Column blocks are Time then Energy; no seed or case-count columns. MLP2 is displayed as MLP².",
        "family_metrics": "MAPE averages GPU-family MAPEs equally across GPUs. MdAPE is the ordinary median of all family instances pooled across GPUs within each seed; mean/sample SD across seeds.",
        "prediction_source": "APE computed from release predictions and dataset totals at full precision; no per-case rounding before aggregation.",
        "average_rows": "MAPE table footers match summary.csv; mean GPU rows and mean family rows also match overall before rounding. Per-GPU Average uses equal family weight for MAPE and pooled instances for MdAPE. SDs are computed across seeds, not averaged across groups. MdAPE tables have no Average footer.",
        "precision": "Case APEs, group statistics, and seed statistics retain full precision throughout calculations. Summary tables and per_seed_metrics.csv are formatted to two decimals only when written; terminal and text summaries also use two decimals. Raw prediction CSVs retain full precision.",
        "ground_truth": "Model time uses independently measured total; layer/segment time uses prefix-derived labels. Model energy uses the released total_energy estimate. APE excludes non-positive targets.",
        "scope": "LR, MLP, MLP2, Transformer, and Layer2Seg; per-GPU predictors within the released profiling regime",
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    comparison = method_comparison_summary(overall, gpus, seeds)
    (destination / "summary.txt").write_text(comparison + "\n", encoding="utf-8")
    return comparison


def report_directory(root, config, gpus, seeds):
    if len(seeds) == 1:
        destination = root / f"seed_{seeds[0]}"
    else:
        destination = root / "multiseed" / ("seed_" + "_".join(map(str, sorted(seeds))))
    if len(gpus) == 1:
        destination /= gpus[0]
    elif set(gpus) != set(config["datasets"]):
        destination /= "gpus_" + "_".join(gpu for gpu in config["datasets"] if gpu in gpus)
    return destination


def save_evaluation_reports(models, config, gpus, seeds, root, device, methods):
    scopes = [([gpu], [seed]) for seed in seeds for gpu in gpus]
    if len(gpus) > 1:
        scopes.extend((gpus, [seed]) for seed in seeds)
    if len(seeds) > 1:
        scopes.extend(([gpu], seeds) for gpu in gpus)
        if len(gpus) > 1:
            scopes.append((gpus, seeds))
    for selected_gpus, selected_seeds in scopes:
        destination = report_directory(root, config, selected_gpus, selected_seeds)
        comparison = save_reports(models, config, selected_gpus, selected_seeds, destination, device, methods)
    return comparison, destination


def evaluate(dataset, predictor, config, seed, destination):
    if predictor.metadata["gpu"] != dataset.gpu or predictor.metadata["seed"] != seed:
        raise ValueError("Checkpoint GPU/seed differs from the evaluation request")
    if predictor.metadata["dataset_sha256"] != dataset.sha256:
        raise ValueError("Checkpoint training dataset differs from the evaluation dataset")
    models, layers = [], []
    for record in dataset.test_records(config["held_out_families"]):
        dense = predictor.predict(record.features).tolist()
        predicted_time = sum(row[0] for row in dense) / 1000.
        predicted_energy = sum(row[1] for row in dense)
        base = {"gpu": dataset.gpu, "seed": seed, "method": getattr(predictor, "method", "Layer2Seg"), "family": family(record.name, config["held_out_families"]),
                "model": record.name, "input_size": "x".join(map(str, record.input_size))}
        models.append({**base, "num_layers": len(dense), "measured_time_ms": record.total_time_ms,
                       "predicted_time_ms": predicted_time,
                       "time_ape_percent": ape(predicted_time, record.total_time_ms),
                       "measured_energy_mj": record.total_energy_mj, "predicted_energy_mj": predicted_energy,
                       "energy_ape_percent": ape(predicted_energy, record.total_energy_mj)})
        for index, (target, prediction) in enumerate(zip(record.targets, dense)):
            layers.append({**base, "layer_index": index, "measured_time_ms": target[2] / 1000.,
                           "predicted_time_ms": prediction[0] / 1000.,
                           "measured_energy_mj": target[6], "predicted_energy_mj": prediction[1]})
    write_csv(destination / "model_predictions.csv", models, MODEL_FIELDS)
    write_csv(destination / "layer_predictions.csv", layers, LAYER_FIELDS)
    return models


def load_saved_predictions(root, gpus, seeds, methods):
    models, sources, references = {}, [], {}
    for gpu in gpus:
        for seed in seeds:
            for method in methods:
                path = Path(root) / f"seed_{seed}" / gpu
                if method != "Layer2Seg":
                    path /= method
                path /= "model_predictions.csv"
                data = path.read_bytes()
                with path.open(newline="") as handle:
                    reader = csv.DictReader(handle)
                    if set(MODEL_FIELDS) - set(reader.fieldnames or []):
                        raise ValueError(f"Missing raw prediction fields: {path}")
                    rows = list(reader)
                cases = {}
                for row in rows:
                    if (row["gpu"], int(row["seed"]), row["method"]) != (gpu, seed, method):
                        raise ValueError(f"Incorrect raw prediction scope: {path}")
                    key = row["family"], row["model"], row["input_size"]
                    if key in cases:
                        raise ValueError(f"Duplicate raw prediction case: {path}")
                    row["seed"], row["num_layers"] = int(row["seed"]), int(row["num_layers"])
                    for target, unit in [("time", "ms"), ("energy", "mj")]:
                        for kind in ["measured", "predicted"]:
                            field = f"{kind}_{target}_{unit}"
                            row[field] = float(row[field])
                            if not math.isfinite(row[field]):
                                raise ValueError(f"Nonfinite raw prediction: {path}")
                        field = f"{target}_ape_percent"
                        row[field] = float(row[field]) if row[field] else None
                        expected = ape(row[f"predicted_{target}_{unit}"], row[f"measured_{target}_{unit}"])
                        if ((expected is None) != (row[field] is None) or expected is not None
                                and not math.isclose(expected, row[field], rel_tol=1e-10, abs_tol=1e-10)):
                            raise ValueError(f"Inconsistent raw APE: {path}")
                    cases[key] = (row["measured_time_ms"], row["measured_energy_mj"])
                if not cases or cases != references.setdefault(gpu, cases):
                    raise ValueError(f"Raw case/GT mismatch across seeds or methods: {path}")
                if path.read_bytes() != data:
                    raise ValueError(f"Raw predictions changed during collection: {path}")
                sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()})
                models[gpu, seed, method] = rows
    return models, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", nargs="+", default=["2080Ti"], help="Dataset GPU names, or all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024])
    parser.add_argument("--methods", nargs="+", default=["all"], help="LR, MLP, MLP2, Transformer, Layer2Seg, or all")
    parser.add_argument("--device", default="auto", help="Default: CUDA if available, otherwise CPU; accepts cpu or cuda:N")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--checkpoint_root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--output_root", type=Path, default=ROOT / "runs/evaluation",
                        help="Raw model/layer prediction CSV root (unchanged)")
    parser.add_argument("--report_root", type=Path, default=ROOT / "runs/main",
                        help="Generated main tables and statistics root")
    parser.add_argument("--report_only", action="store_true", help="Rebuild reports from saved raw predictions; no checkpoints, datasets, or inference")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        config = load_config(args.config)
        gpus = select_gpus(config, args.gpu)
        methods = select_methods(args.methods)
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError("Select distinct seeds")
        if args.report_only:
            models, sources = load_saved_predictions(args.output_root, gpus, args.seeds, methods)
            comparison, destination = save_evaluation_reports(models, {**config, "report_source_files": sources},
                gpus, args.seeds, args.report_root, "saved_predictions", methods)
            print(f"\n{comparison}\nReports saved to: {destination.resolve()}", flush=True)
            return
        args.device = resolve_device(args.device)
        print(f"Predictor device: {args.device}", flush=True)
        models_by_run = {}
        for gpu in gpus:
            dataset = Dataset(gpu, config, args.data_root)
            for seed in args.seeds:
                for method in methods:
                    predictor = Predictor(args.checkpoint_root, gpu, seed, method, args.device)
                    destination = args.output_root / f"seed_{seed}" / gpu
                    if method != "Layer2Seg":
                        destination = destination / method
                    models = evaluate(dataset, predictor, config, seed, destination)
                    models_by_run[gpu, seed, method] = models
                    print(f"{gpu}, seed {seed}, {method}: {len(models)} held-out model/input cases -> {destination}", flush=True)
                    del predictor
        comparison, destination = save_evaluation_reports(models_by_run, config, gpus, args.seeds,
                                                         args.report_root, args.device, methods)
        print(f"\n{comparison}\n", flush=True)
        print(f"Reports saved to: {destination.resolve()}", flush=True)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
