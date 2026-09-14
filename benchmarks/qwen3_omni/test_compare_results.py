# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Synthetic-data validation only: these tests do not measure real performance."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from compare_results import compare_results, markdown_report


def fixture(variant, pair_id="r1", difference=10):
    metadata = {
        "environment": {
            "python": "3.12",
            "torch": "test",
            "vllm": "test",
            "vllm_omni": "test",
            "cuda": "test",
            "gpu_name": "synthetic GPU",
            "gpu_uuid": "synthetic-uuid",
            "gpu_total_memory": 1000,
            "gpu_capability": [9, 0],
            "platform": "test",
            "device": "cuda:0",
            "mask_device": "cpu",
            "embedding_backend": "synthetic",
            "tp_size": 1,
            "dtype": "bfloat16",
        },
        "model_shape": {
            "vocab_size": 128,
            "hidden_size": 8,
            "deepstack_levels": 2,
            "image_token_id": 125,
            "video_token_id": 126,
            "audio_token_id": 127,
        },
        "config_sha256": None,
        "seed": 42,
        "script_sha256": "script",
        "harness_sha256": "harness",
        "vllm_sources_sha256": {
            "interfaces.py": "interfaces-source",
            "utils.py": "utils-source",
            "qwen2_5_omni_thinker.py": "upstream-thinker-source",
            "vocab_parallel_embedding.py": "embedding-source",
        },
        "source_sha256": variant,
        "git_head": variant,
        "git_dirty": variant == "patched",
        "warmup": 10,
        "iterations": 10,
        "blocks": 4,
        "input_pool_size": 4,
    }
    return {
        "schema_version": "qwen3-omni-embedding-v1",
        "variant": variant,
        "mode": "timing",
        "pair_id": pair_id,
        "status": "complete",
        "metadata": metadata,
        "results": [
            {
                "case_id": "image:n32:m8",
                "scenario": "image",
                "num_tokens": 32,
                "num_mm_tokens": 8,
                "case_sha256": "same-case",
                "deepstack": True,
                "oracle_passed": True,
                "lm_calls_per_invocation": [2 if variant == "baseline" else 1] * 4,
                "measurements": [
                    {
                        "block_id": i,
                        "wall_us": 100 + i - (difference if variant == "patched" else 0),
                        "cuda_interval_us": 90 + i - (difference if variant == "patched" else 0),
                    }
                    for i in range(4)
                ],
            }
        ],
    }


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.baseline = fixture("baseline")
        self.patched = fixture("patched")

    def write(self, name, data):
        path = self.directory / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def compare(self):
        return compare_results([self.write("b.json", self.baseline)], [self.write("p.json", self.patched)], 100, 42)

    def test_positive_negative_and_zero_savings(self):
        for difference, status in [(10, "improvement"), (-10, "regression"), (0, "no_stable_difference")]:
            with self.subTest(difference=difference):
                baseline = [self.write(f"b{index}.json", fixture("baseline", f"r{index}")) for index in range(5)]
                patched = [
                    self.write(f"p{index}.json", fixture("patched", f"r{index}", difference)) for index in range(5)
                ]
                result = compare_results(baseline, patched, 100, 42)["results"][0]
                self.assertEqual(result["status"], status)
                metric = result["metrics"]["wall_us"]
                self.assertEqual(metric["paired_saving_median_us"], difference)
                self.assertEqual(metric["paired_saving_ci95_us"], [difference, difference])
                self.assertEqual(metric["baseline"]["median"], 101.5)
                self.assertEqual(metric["baseline"]["iqr"], 1.5)

    def test_single_pair_has_no_stability_claim(self):
        summary = self.compare()
        result = summary["results"][0]
        self.assertEqual(result["status"], "insufficient_independent_pairs")
        self.assertIsNone(result["metrics"]["wall_us"]["paired_saving_ci95_us"])
        self.assertTrue(summary["warnings"])
        self.assertIn("N/A", markdown_report(summary))

    def test_pairing_uses_block_ids_not_list_order(self):
        self.patched["results"][0]["measurements"].reverse()
        self.assertEqual(self.compare()["results"][0]["metrics"]["wall_us"]["paired_saving_median_us"], 10)

    def test_multiple_pairs_and_deterministic_bootstrap(self):
        b2, p2 = fixture("baseline", "r2"), fixture("patched", "r2", difference=-2)
        baseline = [self.write("b2.json", b2), self.write("b1.json", self.baseline)]
        patched = [self.write("p1.json", self.patched), self.write("p2.json", p2)]
        result = compare_results(baseline, patched, 100, 42)
        self.assertEqual(result, compare_results(baseline[::-1], patched[::-1], 100, 42))
        self.assertEqual(result["results"][0]["num_paired_blocks"], 8)
        self.assertEqual(result["results"][0]["num_pairs"], 2)
        self.assertEqual(result["results"][0]["status"], "no_stable_difference")
        self.assertEqual(result["results"][0]["metrics"]["wall_us"]["paired_saving_median_us"], 4)
        self.assertIn("run-pair cluster bootstrap", markdown_report(result))
        self.assertIn(str(baseline[0]), markdown_report(result))

    def test_saving_uses_pair_medians_instead_of_pooled_blocks(self):
        baseline, patched = [], []
        for index, savings in enumerate(([-100, -100, 200, 200], [-100, -100, 200, 200], [0, 0, 0, 0])):
            before, after = fixture("baseline", f"r{index}"), fixture("patched", f"r{index}")
            for block_id, saving in enumerate(savings):
                before["results"][0]["measurements"][block_id].update(wall_us=1000, cuda_interval_us=900)
                after["results"][0]["measurements"][block_id].update(
                    wall_us=1000 - saving, cuda_interval_us=900 - saving
                )
            baseline.append(self.write(f"b{index}.json", before))
            patched.append(self.write(f"p{index}.json", after))
        metric = compare_results(baseline, patched, 100, 42)["results"][0]["metrics"]["wall_us"]
        self.assertEqual(metric["paired_saving_median_us"], 50)
        self.assertEqual(metric["paired_reduction_median_pct"], 5)
        self.assertEqual(metric["ratio_of_medians_reduction_pct"], 0)

    def test_rejects_metadata_mismatches_and_missing_fields(self):
        changes = [
            lambda data: data["metadata"]["environment"].update(gpu_uuid="different GPU"),
            lambda data: data["metadata"].update(seed=43),
            lambda data: data["metadata"].pop("harness_sha256"),
            lambda data: data["metadata"]["environment"].pop("gpu_name"),
            lambda data: data["metadata"].update(source_sha256="baseline"),
            lambda data: data.update(mode="profile"),
            lambda data: data.update(schema_version="wrong"),
            lambda data: data.update(variant="baseline"),
            lambda data: data.update(pair_id="unpaired"),
            lambda data: data.update(status="incomplete"),
            lambda data: data.pop("status"),
            lambda data: data["metadata"]["vllm_sources_sha256"].update({"interfaces.py": "changed"}),
            lambda data: data["metadata"]["vllm_sources_sha256"].pop("utils.py"),
        ]
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                self.patched = fixture("patched")
                change(self.patched)
                with self.assertRaises(ValueError):
                    self.compare()

    def test_rejects_case_and_block_mismatches(self):
        changes = [
            lambda case: case.update(case_id="different"),
            lambda case: case.update(case_sha256="different"),
            lambda case: case.update(num_tokens=33),
            lambda case: case["measurements"].pop(),
            lambda case: case["measurements"][0].update(block_id=9),
            lambda case: case["measurements"][0].update(block_id=1),
            lambda case: case.update(oracle_passed=False),
            lambda case: case.pop("oracle_passed"),
            lambda case: case.update(lm_calls_per_invocation=[2] * 4),
            lambda case: case.update(lm_calls_per_invocation=[1] * 3),
            lambda case: case.update(lm_calls_per_invocation=[True] * 4),
        ]
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                self.patched = fixture("patched")
                change(self.patched["results"][0])
                with self.assertRaises(ValueError):
                    self.compare()

    def test_rejects_nonfinite_nonpositive_or_boolean_timings(self):
        for metric in ("wall_us", "cuda_interval_us"):
            for value in (float("nan"), float("inf"), -float("inf"), 0, -1, True):
                with self.subTest(metric=metric, value=value):
                    self.patched = fixture("patched")
                    self.patched["results"][0]["measurements"][0][metric] = value
                    with self.assertRaises(ValueError):
                        self.compare()

    def test_rejects_duplicate_pairs_and_inconsistent_source(self):
        b = self.write("b.json", self.baseline)
        p = self.write("p.json", self.patched)
        with self.assertRaisesRegex(ValueError, "duplicate pair_id"):
            compare_results([b, b], [p], 100)
        b2 = copy.deepcopy(self.baseline)
        b2["pair_id"] = "r2"
        b2["metadata"]["source_sha256"] = "another-baseline-source"
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            compare_results([b, self.write("b2.json", b2)], [p], 100)

    def test_cli_writes_markdown_and_json(self):
        baseline = self.write("b.json", self.baseline)
        patched = self.write("p.json", self.patched)
        output_md, output_json = self.directory / "report.md", self.directory / "report.json"
        process = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("compare_results.py")),
                "--baseline",
                str(baseline),
                "--patched",
                str(patched),
                "--output-md",
                str(output_md),
                "--output-json",
                str(output_json),
                "--bootstrap-samples",
                "100",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(output_json.read_text())["results"][0]["status"], "insufficient_independent_pairs")
        self.assertIn("synthetic GPU", output_md.read_text())


if __name__ == "__main__":
    unittest.main()
