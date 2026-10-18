"""Small numerical and end-to-end checks, independent of the research repository."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import json
import re
import shutil
import statistics
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from layer2seg.checkpoints import layer_checkpoint_path, load_layer_checkpoint, load_method, method_path, select_methods
from layer2seg.data import Dataset
from layer2seg.evaluate import evaluate, group_tables, main as evaluate_main, save_evaluation_reports, save_reports, summarize
from layer2seg.features import MinMax, layer_features, rnn_features
from layer2seg.losses import ScaledL1
from layer2seg.predict import Layer2Seg, Predictor
from layer2seg.train import fit, main as train_main, train_layer_bank
from layer2seg.training_baselines import cumulative_samples, fit_ridge, train_methods


class NumericalTests(unittest.TestCase):
    def test_ridge_regularization_intercept_and_target_scale(self):
        state = fit_ridge([[-1.], [1.]], [[2.] * 8, [6.] * 8], alpha=1.)
        values = np.asarray([[-1.], [1.]]) @ np.asarray(state["coef"]).T + state["intercept"]
        values[:, :4] *= state["time_scale"]
        values[:, 4:] *= state["energy_scale"]
        np.testing.assert_allclose(values, [[8. / 3.] * 8, [16. / 3.] * 8])
        self.assertEqual(select_methods(["lr", "mlp^2"]), ["LR", "MLP2"])
    def test_structural_features_padding_and_parameter_order(self):
        layer = ["conv2d", [[1, 3, 8, 8], [4, 6, 6]], [3, 4, 3, 1, 0, 1, 1, 1]]
        self.assertEqual(layer_features(layer), [0, 1, 3, 8, 8, 4, 6, 6, 3, 4, 3, 1, 0, 1, 1, 1])
        layer = ["linear", [[2, 4], [2]], [4, 2, 1]]
        self.assertEqual(layer_features(layer), [1, 1, 1, 2, 4, 1, 1, 2, 4, 2, 1, 0, 0, 0, 0, 0])

    def test_scaler_fits_training_only_and_preserves_constant_columns(self):
        features = np.asarray([[1.] * 16, [3.] * 16, [100.] * 16])
        features[:, 0] = 8.
        scaler = MinMax.fit(features[:2])
        scaled = scaler.transform(features)
        self.assertEqual(scaled[0, 0], 0.)
        self.assertEqual(scaled[1, 0], 0.)
        self.assertEqual(scaled[-1, 1], 49.5)  # Validation data cannot change the fit.
        np.testing.assert_array_equal(scaler.transform(features), MinMax.from_dict(scaler.to_dict()).transform(features))

    def test_joint_loss_uses_separate_time_and_energy_scales(self):
        actual = ScaledL1()(torch.tensor([[12., 3.], [24., 6.]]), torch.tensor([[10., 2.], [20., 4.]]))
        expected = 3. / (15. + 1e-5) + 1.5 / (3. + 1e-5)
        self.assertAlmostEqual(actual.item(), expected, places=6)

    def test_transformer_all_skipped_training_epoch_can_recover(self):
        torch.manual_seed(2024)
        model = torch.nn.Linear(1, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        batches = [(torch.tensor([[1.], [2.]]), torch.tensor([[1., 2.], [2., 4.]]))]
        initial = model.weight.detach().clone()
        calls = 0

        def intermittent_bad_gradient(gradient):
            nonlocal calls
            calls += 1
            return torch.full_like(gradient, float("nan")) if calls == 1 else gradient

        handle = model.weight.register_hook(intermittent_bad_gradient)
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                history = fit(model, batches, batches, optimizer, 2, "cpu", "Transformer", clip_norm=1.)
        finally:
            handle.remove()
        self.assertTrue(np.isnan(history["train"][0]))
        self.assertTrue(np.isfinite(history["validation"]).all())
        self.assertTrue(np.isfinite(history["train"][1]))
        self.assertEqual(history["skipped_train_batches"][0]["gradients"], 1)
        self.assertEqual(sum(history["skipped_train_batches"][1].values()), 0)
        self.assertEqual(optimizer.state[model.weight]["step"].item(), 1)
        self.assertFalse(torch.equal(model.weight, initial))
        self.assertTrue(torch.isfinite(model.weight).all())
        self.assertIn("Transformer: epoch 2/2, train=", output.getvalue())
        self.assertNotIn("skipped", output.getvalue())

    def test_transformer_all_skipped_validation_epoch_can_recover(self):
        class IntermittentValidation(torch.nn.Linear):
            validations = 0

            def forward(self, features):
                values = super().forward(features)
                if not self.training:
                    self.validations += 1
                    if self.validations == 1:
                        return values * float("nan")
                return values

        torch.manual_seed(2024)
        model = IntermittentValidation(1, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        batches = [(torch.tensor([[1.], [2.]]), torch.tensor([[1., 2.], [2., 4.]]))]
        with redirect_stdout(io.StringIO()):
            history = fit(model, batches, batches, optimizer, 2, "cpu", "Transformer", clip_norm=1.)
        self.assertTrue(np.isfinite(history["train"]).all())
        self.assertTrue(np.isnan(history["validation"][0]))
        self.assertTrue(np.isfinite(history["validation"][1]))
        self.assertEqual(history["skipped_validation_batches"][0]["outputs"], 1)

    def test_previous_layer_encoding_restarts_at_a_segment(self):
        features = [[kind] + [1] * 15 for kind in [5, 8]]
        values = rnn_features(features, [[0.] * 8] * 2)
        self.assertEqual([row[-1] for row in values], [0, 5])
        self.assertTrue(all(len(row) == 25 for row in values))

    def test_sliding_padding_zero_prior_distribution_and_overlap_averaging(self):
        class Constant(torch.nn.Module):
            def forward(self, values):
                self.last_shape = tuple(values.shape)
                return torch.tensor([[7., 14.]])
        predictor = Layer2Seg.__new__(Layer2Seg)
        predictor.device = "cpu"
        predictor.sequence_length = 7
        predictor.rnn = Constant()
        predictor.layer_priors = lambda features: [[0.] * 8 for _ in features]
        values = predictor.predict([[0.] * 16, [0.] * 16])
        np.testing.assert_array_equal(values, [[5.25, 10.5], [3.5, 7.]])
        self.assertEqual(predictor.rnn.last_shape, (1, 7, 25))
        with self.assertRaises(ValueError):
            predictor.predict([[0.] * 16], 0, 0)

    def test_report_weighting_matches_equal_gpu_overall_and_family(self):
        def cases(groups):
            return [{"family": name, "time_ape_percent": float(error), "energy_ape_percent": float(error)}
                    for name, errors in groups.items() for error in errors]
        models = {("a", 1): cases({"x": [0], "y": [100]}),
                  ("a", 2): cases({"x": [10], "y": [100]}),
                  ("b", 1): cases({"x": [20, 40, 60], "y": [100]}),
                  ("b", 2): cases({"x": [30, 50, 70], "y": [100]})}
        config = {"held_out_families": ["x", "y"]}
        _, detailed_gpu, detailed_dnn = group_tables(models, config, ["a", "b"], [1, 2])
        self.assertEqual([(r["gpu"], r["family"]) for r in detailed_gpu],
                         [("a", "x"), ("a", "y"), ("a", "Average"), ("b", "x"), ("b", "y"), ("b", "Average")])
        self.assertEqual([(r["family"], r["gpu"]) for r in detailed_dnn],
                         [("x", "a"), ("x", "b"), ("x", "Average"), ("y", "a"), ("y", "b"), ("y", "Average")])
        pooled = detailed_dnn[2]
        self.assertEqual(pooled["num_cases_per_seed"], 4)
        self.assertEqual(pooled["time_mape_percent_mean"], 25.)
        self.assertAlmostEqual(pooled["time_mape_percent_sd"], 7.07106781)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_reports(models, config, ["a", "b"], [1, 2], root, "cpu")
            with (root / "summary_by_gpu.csv").open() as handle:
                gpu_rows = list(csv.DictReader(handle))
            with (root / "summary_by_dnn.csv").open() as handle:
                dnn_rows = list(csv.DictReader(handle))
            self.assertEqual([r["GPU Platform"] for r in gpu_rows], ["a", "b", "Average"])
            self.assertEqual([r["DNN Family"] for r in dnn_rows], ["x", "y", "Average"])
            self.assertEqual(dnn_rows[0]["Time - Layer2Seg Mean"], "25.00")
            self.assertEqual(dnn_rows[0]["Time - Layer2Seg SD"], "7.07")
            with (root / "summary.csv").open() as handle:
                overall = list(csv.DictReader(handle))[0]
            self.assertEqual(list(overall), ["method", "time_mape_mean", "time_mape_sd", "time_mdape_mean", "time_mdape_sd",
                                            "energy_mape_mean", "energy_mape_sd", "energy_mdape_mean", "energy_mdape_sd"])
            self.assertEqual(overall["time_mape_mean"], "62.50")
            self.assertEqual(overall["time_mape_sd"], "3.54")
            self.assertEqual(overall["time_mdape_mean"], "55.00")
            self.assertEqual(overall["time_mdape_sd"], "7.07")
            comparison = (root / "summary.txt").read_text()
            self.assertIn("SD columns are sample SDs across seeds.", comparison)
            self.assertIn("Time MdAPE (%)", comparison)
            self.assertIn("Time MdAPE SD (%)", comparison)
            self.assertRegex(comparison, r"Layer2Seg\s+62\.50\s+3\.54\s+55\.00\s+7\.07")
            for rows in [gpu_rows, dnn_rows]:
                for target, heading in [('time', 'Time'), ('energy', 'Energy')]:
                    for stat, title in [('mean', 'Mean'), ('sd', 'SD')]:
                        self.assertEqual(rows[-1][f'{heading} - Layer2Seg {title}'], overall[f'{target}_mape_{stat}'])
            with (root / "summary_mdape_by_dnn.csv").open() as handle:
                median_rows = list(csv.DictReader(handle))
            self.assertEqual(median_rows[0]["Time - Layer2Seg"], "35.00")

    def test_wide_reports_display_method_names_and_distinguish_median_from_mean(self):
        rows = [{"family": "efficientnet", "time_ape_percent": float(error), "energy_ape_percent": float(error)}
                for error in [.04, 10.04, 80.04]]
        methods = ["MLP2", "Layer2Seg"]
        models = {("2080Ti", 2024, method): rows for method in methods}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_reports(models, {"held_out_families": ["efficientnet"]}, ["2080Ti"], [2024], root, "cpu", methods)
            with (root / "summary_by_gpu.csv").open() as handle:
                means = list(csv.DictReader(handle))
            with (root / "summary_mdape_by_gpu.csv").open() as handle:
                medians = list(csv.DictReader(handle))
            self.assertEqual(list(means[0]), ["GPU Platform"] + [f'{target} - {method} {stat}'
                             for target in ['Time', 'Energy'] for method in ['MLP²', 'Layer2Seg'] for stat in ['Mean', 'SD']])
            self.assertEqual(means[0]["GPU Platform"], "RTX 2080 Ti")
            self.assertEqual(means[0]["Time - MLP² Mean"], "30.04")
            self.assertEqual(means[0]["Time - MLP² SD"], "")
            self.assertEqual(medians[0]["Time - MLP²"], "10.04")
            with (root / "summary.csv").open() as handle:
                overall = list(csv.DictReader(handle))
            self.assertEqual([row["method"] for row in overall], ["MLP²", "Layer2Seg"])
            self.assertEqual(overall[0]["time_mape_sd"], "")
            self.assertEqual(overall[0]["time_mdape_mean"], "10.04")
            self.assertEqual(overall[0]["time_mdape_sd"], "")
            self.assertEqual(overall[0]["energy_mdape_sd"], "")
            comparison = (root / "summary.txt").read_text()
            self.assertIn("MLP²", comparison)
            self.assertIn("Layer2Seg", comparison)
            self.assertIn("30.04", comparison)
            self.assertNotIn("±", comparison)
            self.assertEqual(rows[0]["time_ape_percent"], .04)

    def test_overall_mdape_averages_seed_medians_without_pooling_seeds_or_gpu_medians(self):
        seeds = [1, 2, 3]
        models = {(gpu, seed): [{"family": "x", "time_ape_percent": error,
                                "energy_ape_percent": 2 * error} for error in errors]
                  for seed, offset in zip(seeds, [0., 10., 30.])
                  for gpu, errors in [("a", [offset] * 3), ("b", [100.]) ]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_reports(models, {"held_out_families": ["x"]}, ["a", "b"], seeds, root, "cpu")
            with (root / "summary.csv").open() as handle:
                overall = next(csv.DictReader(handle))
            self.assertEqual(overall["time_mdape_mean"], "13.33")
            self.assertEqual(overall["energy_mdape_mean"], "26.67")
            self.assertEqual(overall["time_mdape_sd"], f"{statistics.stdev([0., 10., 30.]):.2f}")
            self.assertEqual(overall["energy_mdape_sd"], f"{statistics.stdev([0., 20., 60.]):.2f}")
            pooled = statistics.median(case["time_ape_percent"] for cases in models.values() for case in cases)
            self.assertEqual(pooled, 20.)
            self.assertIn("13.33", (root / "summary.txt").read_text())

    def test_report_statistics_keep_full_precision_until_export(self):
        seeds = [1, 2, 3]
        models = {}
        for seed, offset in zip(seeds, [0., .03, .06]):
            for gpu, errors in [("a", [.014, .024, .034]), ("b", [.044])]:
                models[gpu, seed] = [{"family": "x", "time_ape_percent": error + offset,
                                      "energy_ape_percent": 2 * (error + offset)} for error in errors]
        before = json.dumps(models["a", 1])
        per_seed, _, by_family = group_tables(models, {"held_out_families": ["x"]}, ["a", "b"], seeds)
        first = next(row for row in per_seed if row["gpu"] == "a" and row["seed"] == 1 and row["family"] == "Average")
        self.assertAlmostEqual(first["time_mape_percent"], .024)
        self.assertAlmostEqual(first["time_mdape_percent"], .024)
        family = next(row for row in by_family if row["gpu"] == "Average")
        self.assertAlmostEqual(family["time_mape_percent_mean"], .064)
        self.assertAlmostEqual(family["time_mdape_percent_mean"], .059)
        self.assertAlmostEqual(family["time_mape_percent_sd"], statistics.stdev([.034, .064, .094]))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_reports(models, {"held_out_families": ["x"]}, ["a", "b"], seeds, root, "cpu")
            with (root / "summary.csv").open() as handle:
                overall = list(csv.DictReader(handle))[0]
            for target, scale in [("time", 1), ("energy", 2)]:
                values = [scale * value for value in [.034, .064, .094]]
                self.assertEqual(overall[f"{target}_mape_mean"], f"{statistics.mean(values):.2f}")
                self.assertEqual(overall[f"{target}_mape_sd"], f"{statistics.stdev(values):.2f}")
            metadata = json.loads((root / "metadata.json").read_text())
            self.assertIn("no per-case rounding", metadata["prediction_source"])
            self.assertIn("full precision", metadata["precision"])
        self.assertEqual(json.dumps(models["a", 1]), before)

    def test_single_seed_reports_are_separate_from_other_seeds_and_combined_reports(self):
        config = {"datasets": {"a": {}, "b": {}, "c": {}}, "held_out_families": ["mobilenet"]}
        models = {(gpu, seed, method): [{"family": "mobilenet", "time_ape_percent": float(seed + offset),
                                       "energy_ape_percent": float(2 * seed + offset)}]
                  for gpu, offset in [("a", 0), ("b", 10)] for seed in [2024, 2025]
                  for method in ["MLP", "Layer2Seg"]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, combined = save_evaluation_reports(models, config, ["a", "b"], [2024, 2025], root,
                                                   "cpu", ["MLP", "Layer2Seg"])
            self.assertEqual(combined, root / "multiseed/seed_2024_2025/gpus_a_b")
            for seed in [2024, 2025]:
                for gpu in ["a", "b"]:
                    folder = root / f"seed_{seed}" / gpu
                    self.assertTrue((folder / "summary.csv").exists())
                    self.assertTrue((folder / "summary.txt").exists())
                    self.assertEqual(json.loads((folder / "metadata.json").read_text())["seeds"], [seed])
                self.assertTrue((root / f"seed_{seed}/gpus_a_b/summary.csv").exists())
            self.assertTrue((root / "multiseed/seed_2024_2025/a/summary.csv").exists())
            before = {path: path.read_bytes() for path in [combined / "summary.csv",
                       root / "seed_2025/a/summary.csv", root / "seed_2024/gpus_a_b/summary.csv"]}
            _, single = save_evaluation_reports(models, config, ["a"], [2024], root, "cpu", ["MLP", "Layer2Seg"])
            self.assertEqual(single, root / "seed_2024/a")
            self.assertNotIn("±", (single / "summary.txt").read_text())
            self.assertFalse((root / "summary.csv").exists())
            for path, contents in before.items():
                self.assertEqual(path.read_bytes(), contents)
            all_config = {**config, "datasets": {"a": {}, "b": {}}}
            _, all_gpus = save_evaluation_reports(models, all_config, ["a", "b"], [2024, 2025], root,
                                                  "cpu", ["MLP", "Layer2Seg"])
            self.assertEqual(all_gpus, root / "multiseed/seed_2024_2025")
            self.assertTrue((root / "seed_2024/summary.csv").exists())


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        raw = {}
        for index, name in enumerate(["model_0", "model_1", "model_2", "mobilenet_fixture"]):
            layers = {}
            for position in range(14):
                kind = "conv2d" if position % 2 == 0 else "relu"
                parameters = [3, 3, 1, 1, 0, 1, 1, 1] if kind == "conv2d" else [1]
                layers[str(position)] = [kind, [[1, 3, 4, 4], [3, 4, 4]], parameters,
                                         [.001] * 4, [50.], [.05] * 4]
            raw[str(index)] = {"model name": name, "input size": [4], "layers": layers,
                               "total": [.014], "total_energy": [.7]}
        path = cls.root / "fixture/all_logs.json"
        path.parent.mkdir()
        path.write_text(json.dumps(raw))
        cls.config = {"datasets": {"fixture": {"file": "fixture/all_logs.json", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
                      "held_out_families": ["mobilenet"], "sequence_length": 7, "training_ratio": .9,
                      "mlp_epochs": 1, "rnn_epochs": 1, "mlp_learning_rate": .001,
                      "rnn_learning_rate": .01, "rnn_weight_decay": 1e-5, "rnn_batch_size": 512}
        cls.dataset = Dataset("fixture", cls.config, cls.root)
        cls.path = layer_checkpoint_path(cls.root / "main", "fixture", 2024)
        cls.refinement = method_path(cls.root / "main", "fixture", 2024, "Layer2Seg")
        with redirect_stdout(io.StringIO()):
            train_methods(cls.dataset, cls.config, 2024, "cpu", cls.root / "main", ["MLP", "Layer2Seg"])
        cls.baselines = cls.root / "baselines"
        bank = layer_checkpoint_path(cls.baselines, "fixture", 2024)
        bank.parent.mkdir(parents=True)
        shutil.copyfile(cls.path, bank)
        baseline_config = {**cls.config, "mlp2_epochs": 1, "transformer_epochs": 1}
        with redirect_stdout(io.StringIO()):
            train_methods(cls.dataset, baseline_config, 2024, "cpu", cls.baselines,
                          ["LR", "MLP", "MLP2", "Transformer", "Layer2Seg"])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_two_stage_training_portable_checkpoint_and_family_holdout(self):
        payload = torch.load(self.path, weights_only=True)
        head = torch.load(self.refinement, weights_only=True)
        self.assertEqual(head["metadata"]["training_samples"], 24)
        self.assertEqual(len(payload["history"]["layers"]["conv2d"]["train"]), 1)
        self.assertEqual(payload["history"]["layers"]["conv2d"]["num_samples"], 21)
        self.assertEqual(len(head["history"]["train"]), 1)
        self.assertNotIn("rnn", payload)
        self.assertNotIn("mlps", head)
        predictor = Layer2Seg(self.path, self.refinement, device="cpu")
        record = self.dataset.find("mobilenet_fixture", [4])
        values = predictor.predict(record.features)
        self.assertEqual(values.shape, (14, 2))
        self.assertTrue(np.isfinite(values).all())
        mlps, scalers, metadata = load_layer_checkpoint(self.path, device="cpu")
        rnn, _ = load_method(self.refinement, "Layer2Seg", "cpu")
        self.assertEqual(len(mlps), 36)
        self.assertIsNone(scalers[1])
        self.assertEqual(rnn.rnn.input_size, 25)

    def test_inference_ignores_measurement_labels(self):
        predictor = Layer2Seg(self.path, self.refinement, device="cpu")
        record = self.dataset.find("model_0", [4])
        before = predictor.predict(record.features)
        # Even malformed label contents must not enter the feature-only forward path.
        layers = json.loads(json.dumps(record.layers))
        for layer in layers:
            layer[3] = ["unused"] * 4
            layer[5] = ["unused"] * 4
        from layer2seg.features import layer_features
        after = predictor.predict([layer_features(layer) for layer in layers])
        np.testing.assert_array_equal(before, after)

    def test_dataset_mutation_is_detected_and_evaluation_uses_total(self):
        predictor = Layer2Seg(self.path, self.refinement, device="cpu")
        rows = evaluate(self.dataset, predictor, self.config, 2024, self.root / "evaluation")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model"], "mobilenet_fixture")
        self.assertEqual(rows[0]["measured_time_ms"], .014)
        self.assertEqual(summarize(rows)["num_cases"], 1)
        path = self.dataset.path
        before = path.read_bytes()
        try:
            path.write_bytes(before + b" ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                Dataset("fixture", self.config, self.root)
        finally:
            path.write_bytes(before)

    def test_checkpoint_outputs_are_repeatable_and_layer_training_overwrites_existing_bank(self):
        first = Layer2Seg(self.path, self.refinement, device="cpu").predict(self.dataset.records[0].features)
        second = Layer2Seg(self.path, self.refinement, device="cpu").predict(self.dataset.records[0].features)
        np.testing.assert_array_equal(first, second)
        destination = layer_checkpoint_path(self.root / "overwrite", "fixture", 2024)
        destination.parent.mkdir(parents=True)
        shutil.copyfile(self.path, destination)
        with redirect_stdout(io.StringIO()):
            train_layer_bank(self.dataset, {**self.config, "mlp_epochs": 2}, 2024, "cpu", destination)
        payload = torch.load(destination, weights_only=True)
        self.assertEqual(len(payload["history"]["layers"]["conv2d"]["train"]), 2)
        self.assertEqual(payload["metadata"]["config"]["mlp_epochs"], 2)

    def test_all_five_methods_train_and_predict_from_portable_checkpoints(self):
        record = self.dataset.find("mobilenet_fixture", [4])
        for method in ["LR", "MLP", "MLP2", "Transformer", "Layer2Seg"]:
            with self.subTest(method=method):
                predictor = Predictor(self.baselines, "fixture", 2024, method, "cpu")
                dense = predictor.predict(record.features)
                self.assertEqual(dense.shape, (14, 2))
                self.assertTrue(np.isfinite(dense).all())
                if method == "LR":
                    np.testing.assert_allclose(dense, [[1., .05]] * 14, atol=1e-7)
                if method == "MLP":
                    np.testing.assert_array_equal(dense, np.asarray(predictor.layer_priors(record.features))[:, [2, 6]])

    def test_cli_saves_training_text_logs_checkpoints_and_evaluation_under_runs(self):
        root = self.root / "cli"
        config_path = self.root / "cli_config.json"
        config_path.write_text(json.dumps(self.config))
        common = ["--gpu", "fixture", "--seeds", "2024", "--device", "cpu",
                  "--config", str(config_path), "--data_root", str(self.root)]
        training = ["train"] + common + ["--mlp_epochs", "1", "--rnn_epochs", "1",
                                         "--mlp2_epochs", "1", "--transformer_epochs", "1"]
        with patch("layer2seg.train.ROOT", root), patch.object(sys, "argv", training), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            train_main()
        log_path = root / "runs/logs/seed_2024/fixture_train.txt"
        first = log_path.read_text()
        self.assertIn("GPU: fixture; seed: 2024; device: cpu", first)
        self.assertIn("train=", first)
        self.assertIn("validation=", first)
        self.assertIn("Transformer: epoch 1/1", first)
        self.assertIn("Training complete.", first)
        for stage in ["LR", "Layer MLPs (MLP)", "MLP2", "Transformer", "Layer2Seg"]:
            self.assertIn(f"[Stage: {stage}] started at", first)
            self.assertRegex(first, rf"\[Stage: {re.escape(stage)}\] completed in \d{{2}}:\d{{2}}:\d{{2}}")
        self.assertIn("Layer MLP (conv2d): epoch 1/1", first)
        self.assertRegex(first, r"epoch_time=\d+\.\d{2}s, elapsed=\d{2}:\d{2}:\d{2}")
        self.assertRegex(first, r"Training complete\. Total elapsed: \d{2}:\d{2}:\d{2}")
        checkpoints = root / "runs/checkpoints"
        for method in ["LR", "MLP", "MLP2", "Transformer", "Layer2Seg"]:
            predictor = Predictor(checkpoints, "fixture", 2024, method, "cpu")
            self.assertEqual(predictor.predict(self.dataset.records[0].features).shape, (14, 2))
        paths = [layer_checkpoint_path(checkpoints, "fixture", 2024)] + [
            method_path(checkpoints, "fixture", 2024, method)
            for method in ["LR", "MLP2", "Transformer", "Layer2Seg"]]
        for path in paths:
            path.write_bytes(b"incomplete checkpoint")
            path.with_suffix(".pt.tmp").write_bytes(b"interrupted save")
        log_path.write_text("previous run log\n")
        with patch("layer2seg.train.ROOT", root), patch.object(sys, "argv", training), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            train_main()
        replaced = log_path.read_text()
        self.assertNotIn("previous run log", replaced)
        self.assertEqual(replaced.count("Started:"), 1)
        self.assertIn("Training complete.", replaced)
        self.assertTrue(all(not path.with_suffix(".pt.tmp").exists() for path in paths))
        with patch("layer2seg.train.ROOT", root), patch.object(sys, "argv", training), \
             patch("layer2seg.training_baselines.train_head", side_effect=KeyboardInterrupt("interrupted training")), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                train_main()
        self.assertIn("interrupted training", log_path.read_text())
        self.assertIn("[Stage: MLP2] stopped after", log_path.read_text())
        self.assertNotIn("Training complete.", log_path.read_text())
        with patch("layer2seg.train.ROOT", root), patch.object(sys, "argv", training), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            train_main()
        self.assertNotIn("interrupted training", log_path.read_text())
        self.assertIn("Training complete.", log_path.read_text())
        for method in ["LR", "MLP", "MLP2", "Transformer", "Layer2Seg"]:
            predictor = Predictor(checkpoints, "fixture", 2024, method, "cpu")
            self.assertTrue(np.isfinite(predictor.predict(self.dataset.records[0].features)).all())
        evaluation_output = io.StringIO()
        with patch("layer2seg.evaluate.ROOT", root), \
             patch.object(sys, "argv", ["evaluate"] + common + ["--checkpoint_root", str(checkpoints)]), \
             redirect_stdout(evaluation_output), redirect_stderr(io.StringIO()):
            evaluate_main()
        summary_folder = root / "runs/main/seed_2024/fixture"
        self.assertTrue((summary_folder / "summary.csv").exists())
        self.assertTrue((root / "runs/evaluation/seed_2024/fixture/model_predictions.csv").exists())
        self.assertTrue((root / "runs/evaluation/seed_2024/fixture/layer_predictions.csv").exists())
        self.assertFalse((root / "runs/evaluation/seed_2024/fixture/summary.csv").exists())
        self.assertFalse((root / "runs/evaluation/summary.csv").exists())
        comparison = (summary_folder / "summary.txt").read_text()
        self.assertIn(comparison.strip(), evaluation_output.getvalue())
        for method in ["LR", "MLP", "MLP²", "Transformer", "Layer2Seg"]:
            self.assertTrue(any(line.startswith(method + " ") for line in comparison.splitlines()))
        self.assertIn("Reports saved to:", evaluation_output.getvalue())
        self.assertTrue((root / "runs/evaluation/seed_2024/fixture/Transformer/model_predictions.csv").exists())
        before = {path: path.read_bytes() for path in (root / 'runs/evaluation').rglob('*.csv')}
        with patch('layer2seg.evaluate.ROOT', root), \
             patch.object(sys, 'argv', ['evaluate'] + common + ['--report_only']), \
             patch('layer2seg.evaluate.Predictor', side_effect=AssertionError('must not load checkpoints')), \
             patch('layer2seg.evaluate.Dataset', side_effect=AssertionError('must not load datasets')), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            evaluate_main()
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual(comparison, (summary_folder / 'summary.txt').read_text())

    def test_mlp2_training_uses_previous_measured_targets_only_in_training(self):
        mlps, scalers, _ = load_layer_checkpoint(self.path, "cpu")
        features, targets = cumulative_samples(self.dataset, mlps, scalers, self.config, "cpu")
        self.assertEqual(len(features), 42)
        self.assertEqual(len(features[0]), 28)
        self.assertEqual(features[0][25:27], [0., 0.])
        self.assertEqual(features[1][25:27], targets[0])
        self.assertEqual(features[14][25:27], [0., 0.])

    def test_lr_only_mlp_only_and_later_rnn_training_have_no_hidden_dependencies(self):
        lr_root, mlp_root = self.root / "lr_only", self.root / "mlp_only"
        with redirect_stdout(io.StringIO()):
            train_methods(self.dataset, self.config, 2024, "cpu", lr_root, ["LR"])
            train_methods(self.dataset, self.config, 2024, "cpu", mlp_root, ["MLP"])
        self.assertFalse(layer_checkpoint_path(lr_root, "fixture", 2024).exists())
        self.assertNotIn("rnn", torch.load(layer_checkpoint_path(mlp_root, "fixture", 2024), weights_only=True))
        self.assertFalse(method_path(mlp_root, "fixture", 2024, "Layer2Seg").exists())
        record = self.dataset.find("mobilenet_fixture", [4])
        self.assertEqual(Predictor(lr_root, "fixture", 2024, "LR", "cpu").predict(record.features).shape, (14, 2))
        self.assertEqual(Predictor(mlp_root, "fixture", 2024, "MLP", "cpu").predict(record.features).shape, (14, 2))
        with redirect_stdout(io.StringIO()):
            train_methods(self.dataset, self.config, 2024, "cpu", mlp_root, ["Layer2Seg"])
        self.assertEqual(Predictor(mlp_root, "fixture", 2024, "Layer2Seg", "cpu").predict(record.features).shape, (14, 2))
        bank = layer_checkpoint_path(mlp_root, "fixture", 2024)
        before = bank.read_bytes()
        with redirect_stdout(io.StringIO()):
            train_methods(self.dataset, {**self.config, "rnn_epochs": 2}, 2024, "cpu", mlp_root, ["Layer2Seg"])
        self.assertEqual(bank.read_bytes(), before)
        head = torch.load(method_path(mlp_root, "fixture", 2024, "Layer2Seg"), weights_only=True)
        self.assertEqual(len(head["history"]["train"]), 2)
        self.assertEqual(Predictor(mlp_root, "fixture", 2024, "Layer2Seg", "cpu").predict(record.features).shape, (14, 2))

    def test_other_neural_methods_do_not_require_layer2seg_refinement(self):
        head = method_path(self.baselines, "fixture", 2024, "Layer2Seg")
        hidden = head.with_suffix(".hidden")
        head.rename(hidden)
        try:
            record = self.dataset.find("mobilenet_fixture", [4])
            for method in ["MLP", "MLP2", "Transformer"]:
                predictor = Predictor(self.baselines, "fixture", 2024, method, "cpu")
                self.assertEqual(predictor.predict(record.features).shape, (14, 2))
            with self.assertRaises(FileNotFoundError):
                Predictor(self.baselines, "fixture", 2024, "Layer2Seg", "cpu")
        finally:
            hidden.rename(head)

    def test_refinement_rejects_a_different_shared_layer_bank(self):
        head = method_path(self.baselines, "fixture", 2024, "Layer2Seg")
        original = head.read_bytes()
        try:
            payload = torch.load(head, weights_only=True)
            payload["metadata"]["layer_bank_sha256"] = "different-bank"
            torch.save(payload, head)
            with self.assertRaisesRegex(ValueError, "different layer bank"):
                Predictor(self.baselines, "fixture", 2024, "Layer2Seg", "cpu")
        finally:
            head.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
