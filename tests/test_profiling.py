"""Profiling schema, execution boundaries, and sensor-unit checks without a GPU."""

from contextlib import nullcontext, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from layer2seg.data import Dataset, load_config
from layer2seg.evaluate import evaluate
from layer2seg.predict import Predictor
from layer2seg.profiling.__main__ import collect, load_model, model_cases
from layer2seg.profiling.dataset import merge_directory, merge_measurements, write_json, write_training_config
from layer2seg.profiling.layers import TraceHooks, describe, discover
from layer2seg.profiling.measurements import collect_model, layer_timings
from layer2seg.profiling.power import JetsonPower, TorchPower, watts
from layer2seg.training_baselines import train_methods


def measurements(name="model_fixture", size=4):
    header = {"model name": name, "input size": size, "number of layers": 3,
              "itterations": [1, 2, 3, 1]}
    data = {"t": [9.],
            "l": [[0, "relu", [[3, size, size], [3, size, size]], [0], [.5, .2]],
                  [1, "identity", [[3, size, size], [3, size, size]], [], [.6, .3]],
                  [2, "relu", [[3, size, size], [3, size, size]], [0], [.7, .4]]],
            "s": [[0, 1.], [1, .8], [2, 1.75]],
            "c": [[0, .3], [1, .6], [2, 1.]],
            "p": [[0, 10.], [1, 20.], [2, 30.]]}
    return {mode: {**header, "data": values} for mode, values in data.items()}


class ProfilingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_merge_preserves_independent_total_and_legacy_label_construction(self):
        record = merge_measurements(measurements())
        self.assertEqual(record["itterations"], [1, 2, 3, 1])
        self.assertEqual(record["total"], [9.])
        self.assertEqual([layer[3][2] for layer in record["layers"].values()], [1., 0., .75])
        self.assertEqual([layer[3][3] for layer in record["layers"].values()], [.3, .3, .4])
        self.assertEqual(record["layers"]["2"][5], [21., 12., 22.5, 12.])
        self.assertEqual(record["total_energy"], [32.5])

    def test_merge_rejects_incomplete_mismatched_and_invalid_measurements(self):
        for mutation in [lambda logs: logs.pop("p"),
                         lambda logs: logs["p"].update({"input size": 8}),
                         lambda logs: logs["s"]["data"].pop(),
                         lambda logs: logs["p"]["data"].append([0, 10.]),
                         lambda logs: logs["p"]["data"][0].__setitem__(1, 0.),
                         lambda logs: logs["s"]["data"][0].__setitem__(1, float("nan"))]:
            with self.subTest(mutation=mutation):
                logs = measurements()
                mutation(logs)
                with self.assertRaises(ValueError):
                    merge_measurements(logs)

    def test_legacy_extra_statistics_and_separate_power_run_counts_are_supported(self):
        logs = measurements()
        logs["p"]["itterations"] = [1, 20, 30, 1]
        for row in logs["l"]["data"]:
            row.append([0., .039])
        self.assertEqual(merge_measurements(logs), merge_measurements(measurements()))

    def test_energy_total_uses_unrounded_products_as_in_original_merger(self):
        logs = measurements()
        logs["s"]["data"] = [[0, .001], [1, .002], [2, .003]]
        logs["p"]["data"] = [[i, .4] for i in range(3)]
        record = merge_measurements(logs)
        self.assertEqual(sum(layer[5][2] for layer in record["layers"].values()), 0.)
        self.assertEqual(record["total_energy"], [.001])

    def test_blank_placeholders_do_not_create_prefix_increments(self):
        logs = measurements()
        logs["l"]["data"][0] = [0, "", [], [], [.1, .1]]
        record = merge_measurements(logs)
        self.assertEqual(record["layers"]["0"][3][2:], [0., 0.])

    def test_saved_dataset_and_generated_configuration_work_with_release_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "MyGPU"
            for mode, log in measurements().items():
                write_json(folder / f"raw/model_fixture_4_{mode}.json", log)
            merge_directory(folder / "raw", folder / "all_logs.json")
            write_training_config("MyGPU", folder / "all_logs.json", folder / "train_config.json")
            config = load_config(folder / "train_config.json")
            dataset = Dataset("MyGPU", config, root)
            self.assertEqual(list(config["datasets"]), ["MyGPU"])
            self.assertEqual(dataset.records[0].features[0], [8, 1, 3, 4, 4, 3, 4, 4] + [0] * 8)
            self.assertEqual(dataset.records[0].targets[0], [500., 200., 1000., 300., 5., 2., 10., 3.])
            with redirect_stdout(io.StringIO()):
                collect(types.SimpleNamespace(gpu="MyGPU", output_root=root, merge_only=True))
            self.assertEqual(json.loads((folder / "all_logs.json").read_text())["0"]["total"], [9.])

    def test_merged_dataset_supports_training_and_evaluation_of_all_five_methods(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "MyGPU"
            for name in ["model_0", "model_1", "model_2", "mobilenet_fixture"]:
                logs = measurements(name)
                for log in logs.values():
                    log["number of layers"] = 14
                logs["l"]["data"] = [[i, "relu", [[3, 4, 4], [3, 4, 4]], [0], [.5, .2]] for i in range(14)]
                for mode in ["s", "c"]:
                    logs[mode]["data"] = [[i, float(i + 1)] for i in range(14)]
                logs["p"]["data"] = [[i, 10.] for i in range(14)]
                for mode, log in logs.items():
                    write_json(folder / f"raw/{name}_4_{mode}.json", log)
            merge_directory(folder / "raw", folder / "all_logs.json")
            write_training_config("MyGPU", folder / "all_logs.json", folder / "train_config.json")
            config = load_config(folder / "train_config.json")
            config.update({key: 1 for key in ["mlp_epochs", "rnn_epochs", "mlp2_epochs", "transformer_epochs"]})
            dataset = Dataset("MyGPU", config, root)
            methods = ["LR", "MLP", "MLP2", "Transformer", "Layer2Seg"]
            with redirect_stdout(io.StringIO()):
                train_methods(dataset, config, 2024, "cpu", root / "weights", methods)
            for method in methods:
                predictor = Predictor(root / "weights", "MyGPU", 2024, method, "cpu")
                rows = evaluate(dataset, predictor, config, 2024, root / "results" / method)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["model"], "mobilenet_fixture")
                self.assertEqual(rows[0]["measured_time_ms"], 9.)  # Independent total, not sum of 14 prefixes.
                self.assertEqual(rows[0]["measured_energy_mj"], 140.)

    def test_actual_execution_order_and_repeated_module_calls(self):
        class Reused(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU()
                self.conv = nn.Conv2d(3, 3, 1, bias=False)

            def forward(self, inputs):
                return self.relu(self.conv(self.relu(inputs)))

        model = Reused().eval()
        inputs = torch.randn(1, 3, 4, 4)
        expected = model(inputs).detach()
        layers = discover(model, inputs)
        self.assertEqual([layer.name for layer in layers], ["relu", "conv", "relu"])
        self.assertEqual(layers[1].parameters[-1], 0)
        self.assertEqual(layers[0].shapes, [[3, 4, 4], [3, 4, 4]])
        observed = []
        with TraceHooks(layers, after=lambda i, *args: observed.append(i)) as hooks:
            actual = hooks.run(model, inputs).detach()
            self.assertIsNone(hooks.run(model, inputs, stop=1))
        np.testing.assert_array_equal(expected, actual)
        self.assertEqual(observed, [0, 1, 2, 0, 1])
        self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks for module in model.modules()))

    def test_real_model_errors_are_propagated_and_hooks_are_cleaned_up(self):
        class Broken(nn.ReLU):
            fail = False

            def forward(self, x):
                if self.fail:
                    raise RuntimeError("real failure")
                return super().forward(x)

        model, inputs = Broken(), torch.ones(1, 3, 2, 2)
        layers = discover(model, inputs)
        model.fail = True
        with self.assertRaisesRegex(RuntimeError, "real failure"):
            with TraceHooks(layers) as hooks:
                hooks.run(model, inputs, stop=0)
        self.assertFalse(model._forward_hooks or model._forward_pre_hooks)

    def test_unsupported_layers_and_changed_execution_are_explicit(self):
        model = nn.Sequential(nn.ReLU(), nn.ELU())
        inputs = torch.ones(1, 3, 2, 2)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            discover(model, inputs)
        layers = discover(model, inputs, allow_unsupported=True)
        self.assertEqual([layer.kind for layer in layers], ["relu", ""])
        model[0] = nn.Identity()
        with self.assertRaisesRegex(ValueError, "execution"):
            with TraceHooks(layers) as hooks:
                hooks.run(model, inputs)
        class Branch(nn.Module):
            def __init__(self):
                super().__init__()
                self.first, self.optional, self.extra = nn.ReLU(), nn.Identity(), False

            def forward(self, x):
                x = self.first(x)
                return self.optional(x) if self.extra else x

        branch = Branch()
        layers = discover(branch, inputs)
        branch.extra = True
        with self.assertRaisesRegex(ValueError, "execution order"):
            with TraceHooks(layers, model=branch) as hooks:
                hooks.run(branch, inputs)

    def test_model_configuration_and_factory_loading(self):
        spec = {"source": "factory", "factory": "torch.nn:Identity"}
        self.assertEqual(len(model_cases({"custom": {**spec, "input_size": [4, 8]}})), 2)
        self.assertIsInstance(load_model("custom", spec), nn.Identity)
        with self.assertRaises(ValueError):
            model_cases({"bad/name": spec})
        with self.assertRaises(ValueError):
            model_cases({"custom": {**spec, "input_size": [4, 4]}})
        self.assertEqual(describe(nn.LeakyReLU(inplace=True)), ("leakyrelu", [1]))

    def test_layer_profiler_annotations_capture_each_execution_once_per_active_run(self):
        model, inputs = nn.Sequential(nn.ReLU(), nn.Identity()), torch.randn(1, 3, 4, 4)
        layers = discover(model, inputs)
        profile = torch.profiler.profile

        def cpu_profile(**kwargs):
            kwargs["activities"] = [torch.profiler.ProfilerActivity.CPU]
            return profile(**kwargs)

        with patch("torch.profiler.profile", side_effect=cpu_profile), patch("layer2seg.profiling.measurements.synchronize"):
            rows = layer_timings(model, inputs, layers, warmup=1, runs=3, device="cpu")
        self.assertEqual([row[:2] for row in rows], [[0, "relu"], [1, "identity"]])
        self.assertTrue(all(len(row[4]) == 2 for row in rows))

    def test_collection_resume_validates_raw_files_and_reuses_completed_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models_path = root / "models.json"
            write_json(models_path, {"model_fixture": {"source": "factory", "factory": "torch.nn:Identity", "input_size": [4]}})
            args = types.SimpleNamespace(gpu="MyGPU", output_root=root, merge_only=False,
                models_config=models_path, warmup=2, runs=3, power_interval=.01, threads=1,
                seed=2024, device="cuda:0", allow_unsupported=False, power_backend="torch", power_rail="auto", resume=False)
            reader = types.SimpleNamespace(metadata={"backend": "fixture", "scope": "test fixture"})
            with patch("layer2seg.profiling.__main__.cuda_device", return_value=torch.device("cuda:0")), \
                 patch("torch.cuda.get_device_name", return_value="Fixture"), \
                 patch("torch.cuda.manual_seed_all"), \
                 patch("layer2seg.profiling.__main__.open_power", side_effect=lambda *args: nullcontext(reader)), \
                 patch("layer2seg.profiling.__main__.collect_model", return_value=(measurements(), [])) as measure, \
                 redirect_stdout(io.StringIO()):
                collect(args)
                self.assertEqual(measure.call_count, 1)
                args.resume = True
                collect(args)
                self.assertEqual(measure.call_count, 1)
                config = load_config(root / "MyGPU/train_config.json")
                self.assertEqual(len(Dataset("MyGPU", config, root).records), 1)
                path = root / "MyGPU/raw/model_fixture_4_p.json"
                path.write_text(path.read_text() + " ")
                with self.assertRaisesRegex(ValueError, "changed"):
                    collect(args)

    def test_torch_power_queries_selected_device_and_converts_units(self):
        with patch.object(torch.cuda, "power_draw", return_value=120500, create=True) as power_draw:
            reader = TorchPower(1)
            self.assertEqual(reader.metadata["api"], "torch.cuda.power_draw")
            self.assertEqual(reader.metadata["device"], "cuda:1")
            self.assertEqual(reader.read(), 120.5)
            self.assertEqual(power_draw.call_count, 2)
            power_draw.assert_called_with(torch.device("cuda:1"))
            power_draw.return_value = 0
            with self.assertRaisesRegex(ValueError, "Invalid power sensor"):
                reader.read()
            reader.close()
        for value in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                watts(value)

    def test_desktop_power_requires_torch_power_draw(self):
        with patch.object(torch.cuda, "power_draw", None, create=True):
            with self.assertRaisesRegex(ValueError, "PyTorch >=2.1.*torch.cuda.power_draw"):
                TorchPower(0)

    def test_jetson_reports_selected_rail_and_does_not_silently_use_board_total(self):
        class Monitor:
            interval = .1
            power = {"rail": {"VDD_GPU_SOC": {"power": 3250}}, "tot": {"power": 8000}}
            def __init__(self, **kwargs): pass
            def start(self): pass
            def ok(self, **kwargs): return True
            def close(self): pass

        with patch.dict("sys.modules", {"jtop": types.SimpleNamespace(jtop=Monitor)}):
            reader = JetsonPower("auto", .01)
            self.assertEqual(reader.read(), 3.25)
            self.assertEqual(reader.metadata["rail"], "VDD_GPU_SOC")
            reader.close()
            with self.assertRaises(ValueError):
                JetsonPower("missing-rail", .01)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("LAYER2SEG_TEST_PROFILING_GPU") == "1",
                         "Real GPU test requires CUDA, power sensors, and LAYER2SEG_TEST_PROFILING_GPU=1")
    def test_real_cuda_collection_with_power_sensor(self):
        from layer2seg.profiling.power import open_power
        model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        with open_power("auto", 0) as reader:
            logs, layout = collect_model(model, "model_cuda_fixture", 16, "cuda:0", 1, 2, reader)
        record = merge_measurements(logs)
        self.assertEqual(len(layout), 3)
        self.assertGreater(record["total"][0], 0.)
        self.assertGreater(record["total_energy"][0], 0.)


if __name__ == "__main__":
    unittest.main()
