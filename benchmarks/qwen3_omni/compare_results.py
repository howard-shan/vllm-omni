#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Compare matched embedding-stage runs; requires only the Python standard library."""

import argparse
import json
import math
import random
import statistics
from pathlib import Path

SCHEMA = "qwen3-omni-embedding-v1"
ENV_FIELDS = (
    "python torch vllm vllm_omni cuda gpu_name gpu_uuid gpu_total_memory "
    "gpu_capability platform device mask_device embedding_backend tp_size dtype"
).split()
SHAPE_FIELDS = ("vocab_size hidden_size deepstack_levels image_token_id video_token_id audio_token_id").split()
MATCH_FIELDS = (
    "environment model_shape config_sha256 seed script_sha256 harness_sha256 "
    "vllm_sources_sha256 warmup iterations blocks input_pool_size"
).split()
VLLM_SOURCE_FILES = "interfaces.py utils.py qwen2_5_omni_thinker.py vocab_parallel_embedding.py".split()
PROVENANCE_FIELDS = "source_sha256 git_head git_dirty".split()
CASE_FIELDS = "case_id scenario num_tokens num_mm_tokens case_sha256 deepstack".split()
METRICS = ("wall_us", "cuda_interval_us")


def require_fields(value, fields, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected an object")
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(f"{label}: missing fields: {', '.join(missing)}")


def require_int(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label}: expected an integer >= {minimum}")


def require_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected a nonempty string")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_run(path, variant):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        run = json.load(handle, object_pairs_hook=unique_object)
    label = str(path)
    require_fields(run, ["schema_version", "variant", "mode", "pair_id", "metadata", "results", "status"], label)
    if run["schema_version"] != SCHEMA or run["variant"] != variant or run["mode"] != "timing":
        raise ValueError(f"{label}: requires schema={SCHEMA}, variant={variant}, mode=timing")
    if run["status"] != "complete":
        raise ValueError(f"{label}: status must be complete")
    require_text(run["pair_id"], f"{label}: pair_id")
    meta = run["metadata"]
    require_fields(meta, MATCH_FIELDS + PROVENANCE_FIELDS, f"{label}: metadata")
    require_fields(meta["environment"], ENV_FIELDS, f"{label}: environment")
    require_fields(meta["model_shape"], SHAPE_FIELDS, f"{label}: model_shape")
    require_fields(meta["vllm_sources_sha256"], VLLM_SOURCE_FILES, f"{label}: vllm_sources_sha256")
    for source_file in VLLM_SOURCE_FILES:
        require_text(meta["vllm_sources_sha256"][source_file], f"{label}: vllm_sources_sha256.{source_file}")
    if type(meta["environment"]["tp_size"]) is not int or meta["environment"]["tp_size"] != 1:
        raise ValueError(f"{label}: this comparator requires tp_size=1")
    for field in SHAPE_FIELDS:
        require_int(meta["model_shape"][field], f"{label}: model_shape.{field}")
    for field in ("vocab_size", "hidden_size"):
        require_int(meta["model_shape"][field], f"{label}: model_shape.{field}", 1)
    for field in ("seed", "warmup"):
        require_int(meta[field], f"{label}: {field}")
    for field in ("iterations", "blocks", "input_pool_size"):
        require_int(meta[field], f"{label}: {field}", 1)
    for field in ("script_sha256", "harness_sha256", "source_sha256", "git_head"):
        require_text(meta[field], f"{label}: {field}")
    if meta["config_sha256"] is not None:
        require_text(meta["config_sha256"], f"{label}: config_sha256")
    if type(meta["git_dirty"]) is not bool:
        raise ValueError(f"{label}: git_dirty must be a boolean")
    if not isinstance(run["results"], list) or not run["results"]:
        raise ValueError(f"{label}: results must be a nonempty list")
    cases = {}
    for case in run["results"]:
        require_fields(
            case, CASE_FIELDS + ["measurements", "oracle_passed", "lm_calls_per_invocation"], f"{label}: case"
        )
        for field in ("case_id", "scenario", "case_sha256"):
            require_text(case[field], f"{label}: {field}")
        case_id = case["case_id"]
        if case["oracle_passed"] is not True:
            raise ValueError(f"{label}: {case_id} oracle_passed must be true")
        counts = case["lm_calls_per_invocation"]
        expected_count = 2 if variant == "baseline" and case["scenario"] not in {"text", "empty", "interleaved"} else 1
        if (
            not isinstance(counts, list)
            or len(counts) != meta["input_pool_size"]
            or any(type(count) is not int or count != expected_count for count in counts)
        ):
            raise ValueError(
                f"{label}: {case_id} requires {meta['input_pool_size']} LM call counts, each equal to {expected_count}"
            )
        if case_id in cases:
            raise ValueError(f"{label}: duplicate case_id {case_id}")
        require_int(case["num_tokens"], f"{label}: {case_id}.num_tokens", 1)
        require_int(case["num_mm_tokens"], f"{label}: {case_id}.num_mm_tokens")
        if case["num_mm_tokens"] > case["num_tokens"] or type(case["deepstack"]) is not bool:
            raise ValueError(f"{label}: invalid dimensions/deepstack for {case_id}")
        measurements = case["measurements"]
        if not isinstance(measurements, list) or len(measurements) != meta["blocks"]:
            raise ValueError(f"{label}: {case_id} measurement count must equal metadata.blocks")
        blocks = {}
        for measurement in measurements:
            require_fields(measurement, ["block_id", *METRICS], f"{label}: {case_id} measurement")
            block_id = measurement["block_id"]
            require_int(block_id, f"{label}: {case_id}.block_id")
            if block_id in blocks:
                raise ValueError(f"{label}: {case_id} duplicate block_id {block_id}")
            for metric in METRICS:
                value = measurement[metric]
                if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{label}: {case_id}.{metric} must be finite and positive")
            blocks[block_id] = measurement
        cases[case_id] = {"spec": {key: case[key] for key in CASE_FIELDS}, "blocks": blocks}
    return {"path": str(path.resolve()), "pair_id": run["pair_id"], "metadata": meta, "cases": cases}


def load_group(paths, variant):
    group = {}
    for path in paths:
        run = load_run(path, variant)
        if run["pair_id"] in group:
            raise ValueError(f"{variant}: duplicate pair_id {run['pair_id']}")
        group[run["pair_id"]] = run
    if not group:
        raise ValueError(f"{variant}: no input files")
    if len({run["metadata"]["source_sha256"] for run in group.values()}) != 1:
        raise ValueError(f"{variant}: source_sha256 must be consistent across the group")
    return group


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def distribution(values):
    q1, q3 = percentile(values, 0.25), percentile(values, 0.75)
    return {"median": statistics.median(values), "q1": q1, "q3": q3, "iqr": q3 - q1}


def metric_summary(baseline, patched, bootstrap_samples, rng):
    pairs = []
    for pair_id in sorted(baseline):
        before, after = baseline[pair_id], patched[pair_id]
        pairs.append(
            {
                "pair_id": pair_id,
                "num_blocks": len(before),
                "paired_saving_median_us": statistics.median([b - p for b, p in zip(before, after)]),
                "paired_reduction_median_pct": statistics.median([(b - p) / b * 100 for b, p in zip(before, after)]),
            }
        )
    # Each entire run pair has one vote; repeated blocks cannot create new pairs.
    savings = [pair["paired_saving_median_us"] for pair in pairs]
    if len(pairs) < 2:
        interval, status = None, "insufficient_independent_pairs"
    else:
        samples = [statistics.median(rng.choices(savings, k=len(savings))) for _ in range(bootstrap_samples)]
        low, high = percentile(samples, 0.025), percentile(samples, 0.975)
        interval = [low, high]
        status = "improvement" if low > 0 else "regression" if high < 0 else "no_stable_difference"
    before = distribution([value for values in baseline.values() for value in values])
    after = distribution([value for values in patched.values() for value in values])
    return {
        "baseline": before,
        "patched": after,
        "paired_saving_median_us": statistics.median(savings),
        "paired_saving_ci95_us": interval,
        "paired_reduction_median_pct": statistics.median([pair["paired_reduction_median_pct"] for pair in pairs]),
        "ratio_of_medians_reduction_pct": (1 - after["median"] / before["median"]) * 100,
        "pairs": pairs,
        "status": status,
    }


def compare_results(baseline_paths, patched_paths, bootstrap_samples=2000, seed=42):
    require_int(bootstrap_samples, "bootstrap_samples", 1)
    require_int(seed, "bootstrap seed")
    groups = {"baseline": load_group(baseline_paths, "baseline"), "patched": load_group(patched_paths, "patched")}
    if groups["baseline"].keys() != groups["patched"].keys():
        raise ValueError("baseline and patched must have exactly the same pair_id set")
    pair_ids = sorted(groups["baseline"])
    reference = groups["baseline"][pair_ids[0]]
    if reference["metadata"]["source_sha256"] == groups["patched"][pair_ids[0]]["metadata"]["source_sha256"]:
        raise ValueError("baseline and patched source_sha256 are identical; refusing a same-source comparison")
    for variant, group in groups.items():
        for run in group.values():
            for field in MATCH_FIELDS:
                if run["metadata"][field] != reference["metadata"][field]:
                    raise ValueError(f"{run['path']}: metadata.{field} mismatch")
            if run["cases"].keys() != reference["cases"].keys():
                raise ValueError(f"{run['path']}: case set mismatch")
            for case_id, case in run["cases"].items():
                if case["spec"] != reference["cases"][case_id]["spec"]:
                    raise ValueError(f"{run['path']}: {case_id} case metadata/hash mismatch")
    for pair_id in pair_ids:
        before, after = groups["baseline"][pair_id], groups["patched"][pair_id]
        for case_id in reference["cases"]:
            if before["cases"][case_id]["blocks"].keys() != after["cases"][case_id]["blocks"].keys():
                raise ValueError(f"pair {pair_id}: {case_id} block_id set mismatch")
    rng = random.Random(seed)
    summaries = []
    for case_id in sorted(reference["cases"]):
        keys = [
            (pair_id, block_id)
            for pair_id in pair_ids
            for block_id in sorted(groups["baseline"][pair_id]["cases"][case_id]["blocks"])
        ]
        metrics = {}
        for metric in METRICS:
            values = {
                variant: {
                    pair_id: [
                        group[pair_id]["cases"][case_id]["blocks"][block_id][metric]
                        for block_id in sorted(group[pair_id]["cases"][case_id]["blocks"])
                    ]
                    for pair_id in pair_ids
                }
                for variant, group in groups.items()
            }
            metrics[metric] = metric_summary(values["baseline"], values["patched"], bootstrap_samples, rng)
        summaries.append(
            {
                **reference["cases"][case_id]["spec"],
                "num_pairs": len(pair_ids),
                "num_paired_blocks": len(keys),
                "paired_blocks": [
                    {
                        "pair_id": pair_id,
                        "block_id": block_id,
                        **{
                            variant: {
                                metric: group[pair_id]["cases"][case_id]["blocks"][block_id][metric]
                                for metric in METRICS
                            }
                            for variant, group in groups.items()
                        },
                    }
                    for pair_id, block_id in keys
                ],
                "status": metrics["wall_us"]["status"],
                "metrics": metrics,
            }
        )
    return {
        "schema_version": "qwen3-omni-embedding-comparison-v1",
        "input_schema_version": SCHEMA,
        "method": {
            "primary_metric": "wall_us",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "confidence_interval": "95% percentile bootstrap of the median of run-pair saving medians",
            "point_estimate": "Median across pairs of each pair's median baseline-minus-patched block difference",
            "percentage_estimate": "Median across pairs of each pair's median paired percentage reduction",
            "resampling_unit": "pair_id (whole run-pair cluster)",
            "distribution_scope": "Baseline/patched median, IQR and ratio-of-medians use all paired blocks",
            "limitation": (
                "Blocks within a run pair may be correlated and are not resampled independently. "
                "Run pairs should be independently repeated; this is not end-to-end performance evidence."
            ),
            "positive_saving_means": "improvement",
        },
        "warnings": (
            [
                f"Only {len(pair_ids)} run pair(s); fewer than 5 pairs gives limited uncertainty evidence. "
                "Repeated blocks do not increase the independent pair count."
            ]
            if len(pair_ids) < 5
            else []
        ),
        "matched_metadata": {key: reference["metadata"][key] for key in MATCH_FIELDS},
        "inputs": {
            variant: [
                {
                    "path": group[pair_id]["path"],
                    "pair_id": pair_id,
                    **{key: group[pair_id]["metadata"][key] for key in PROVENANCE_FIELDS},
                }
                for pair_id in pair_ids
            ]
            for variant, group in groups.items()
        },
        "results": summaries,
    }


def markdown_report(summary):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "# Qwen3-Omni embedding-stage comparison",
        "",
        "Positive savings mean faster patched execution; negative savings are retained.",
        "Primary metric: synchronized host wall time per invocation. "
        "CUDA interval is auxiliary and is not a sum of kernel durations.",
        "",
        "Savings are the median of each run pair's median paired block difference; percentage reductions use "
        "the same two-stage median. Baseline/patched median, IQR and ratio-of-medians describe all blocks.",
        "95% CIs use a **run-pair cluster bootstrap**, resampling whole `pair_id` clusters via their medians. "
        "Blocks within a pair are not independent samples; a single pair has no CI or stability claim.",
        "Run pairs should be independently repeated. These embedding-stage results do not establish end-to-end gains.",
        f"Bootstrap samples: {summary['method']['bootstrap_samples']}; seed: {summary['method']['bootstrap_seed']}.",
        "",
    ]
    for warning in summary["warnings"]:
        lines.extend([f"**Limited evidence:** {warning}", ""])
    for metric in METRICS:
        lines.extend(
            [
                f"## {metric}",
                "",
                "| Case | Run pairs / blocks | Baseline median (IQR), us | Patched median (IQR), us "
                "| Paired saving median [95% CI], us | Median paired reduction, % "
                "| Ratio-of-medians reduction, % | Status |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for case in summary["results"]:
            result = case["metrics"][metric]
            before, after = result["baseline"], result["patched"]
            interval = result["paired_saving_ci95_us"]
            interval_text = "N/A" if interval is None else f"{interval[0]:.3f}, {interval[1]:.3f}"
            lines.append(
                f"| {cell(case['case_id'])} | {case['num_pairs']} / {case['num_paired_blocks']} "
                f"| {before['median']:.3f} ({before['iqr']:.3f}) | {after['median']:.3f} ({after['iqr']:.3f}) "
                f"| {result['paired_saving_median_us']:.3f} [{interval_text}] "
                f"| {result['paired_reduction_median_pct']:.3f} | {result['ratio_of_medians_reduction_pct']:.3f} "
                f"| {result['status']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Matched environment and configuration",
            "",
            "```json",
            json.dumps(summary["matched_metadata"], indent=2, ensure_ascii=False),
            "```",
            "",
            "## Provenance and raw input paths",
            "",
        ]
    )
    for variant, inputs in summary["inputs"].items():
        for run in inputs:
            lines.append(f"- {variant}, pair `{cell(run['pair_id'])}`: `{cell(run['path'])}`")
            lines.append(
                f"  source SHA256 `{cell(run['source_sha256'])}`; "
                f"git HEAD `{cell(run['git_head'])}`; dirty `{run['git_dirty']}`."
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", required=True, type=Path)
    parser.add_argument("--patched", nargs="+", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        output_paths = [args.output_md] + ([args.output_json] if args.output_json else [])
        input_paths = {path.resolve() for path in args.baseline + args.patched}
        if len({path.resolve() for path in output_paths}) != len(output_paths):
            raise ValueError("Markdown and JSON outputs must have different paths")
        if any(path.resolve() in input_paths for path in output_paths):
            raise ValueError("an output path would overwrite a raw input file")
        summary = compare_results(args.baseline, args.patched, args.bootstrap_samples, args.seed)
        args.output_md.write_text(markdown_report(summary), encoding="utf-8")
        if args.output_json:
            args.output_json.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
