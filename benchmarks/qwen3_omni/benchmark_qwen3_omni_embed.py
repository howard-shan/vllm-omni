# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Linux/CUDA benchmark of the real thinker embedding path, without a checkpoint.

Run the SAME script with --source-root selecting baseline or patched sources.
Timing, call-count and profiler runs are separate. No CUDA means a hard error,
never a silent CPU fallback. See README.md for the paired A/B commands.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

SCENARIOS = ("text", "empty", "audio", "image", "video", "mixed", "interleaved", "vision_no_deepstack")
SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA = "qwen3-omni-embedding-v1"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=SCRIPT_DIR.parents[1])
    parser.add_argument("--variant", choices=("baseline", "patched"), required=True)
    parser.add_argument("--pair-id", required=True, help="Same ID for one baseline/patched pair, e.g. r1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("timing", "count", "profile"), default="timing")
    parser.add_argument(
        "--device", default="cuda:0", help="One CUDA device per run; paired A/B runs must use the same GPU"
    )
    parser.add_argument("--mask-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--embedding-backend", choices=("vllm", "torch"), default="vllm")
    parser.add_argument("--config", type=Path, help="Pinned HF config.json; otherwise use the documented Qwen3 preset")
    parser.add_argument("--tokens", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--mm-fraction", type=float, default=0.25)
    parser.add_argument("--scenarios", choices=SCENARIOS, nargs="+", default=list(SCENARIOS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--input-pool-size", type=int, default=4)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    if not args.pair_id.strip():
        parser.error("--pair-id must not be empty")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not math.isfinite(args.mm_fraction) or not 0 < args.mm_fraction < 1:
        parser.error("--mm-fraction must be between 0 and 1")
    for name in ("iterations", "blocks", "input_pool_size", "profile_steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0 or min(args.tokens) <= 0:
        parser.error("warmup must be nonnegative and tokens positive")
    if len(set(args.tokens)) != len(args.tokens) or len(set(args.scenarios)) != len(args.scenarios):
        parser.error("duplicate token lengths or scenarios are not allowed")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; select a new run filename")
    return args


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_harness():
    # Load this script's companion even when the baseline tree predates it.
    spec = importlib.util.spec_from_file_location(
        "qwen3_embedding_benchmark_harness", SCRIPT_DIR / "embedding_harness.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_shape(harness, config_path):
    if config_path is None:
        return harness.ModelShape()
    config = json.loads(config_path.read_text())
    thinker = config.get("thinker_config", config)
    text_config, vision_config = thinker["text_config"], thinker["vision_config"]
    if vision_config.get("out_hidden_size", text_config["hidden_size"]) != text_config["hidden_size"]:
        raise ValueError("This harness requires visual output width to equal thinker text hidden_size")
    return harness.ModelShape(
        vocab_size=text_config["vocab_size"],
        hidden_size=text_config["hidden_size"],
        deepstack_levels=len(vision_config["deepstack_visual_indexes"]),
        image_token_id=thinker["image_token_id"],
        video_token_id=thinker["video_token_id"],
        audio_token_id=thinker["audio_token_id"],
    )


def git_value(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def metadata(args, shape, model, torch):
    source = Path(inspect.getfile(type(model))).resolve()
    if not source.is_relative_to(args.source_root):
        raise RuntimeError(f"Wrong thinker imported: {source}; expected it under {args.source_root}")
    import vllm

    import vllm_omni

    properties = torch.cuda.get_device_properties(args.device)
    gpu_uuid = getattr(properties, "uuid", None)
    if gpu_uuid is None:
        raise RuntimeError("PyTorch must expose the GPU UUID to identify the physical A/B device")
    return {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "vllm_omni": vllm_omni.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": properties.name,
            "gpu_uuid": str(gpu_uuid),
            "gpu_total_memory": properties.total_memory,
            "gpu_capability": [properties.major, properties.minor],
            "platform": platform.platform(),
            "device": str(args.device),
            "mask_device": args.mask_device,
            "embedding_backend": args.embedding_backend,
            "tp_size": 1,
            "dtype": args.dtype,
        },
        "model_shape": asdict(shape),
        "vllm_sources_sha256": {
            name + ".py": sha256(importlib.import_module(module).__file__)
            for name, module in {
                "interfaces": "vllm.model_executor.models.interfaces",
                "utils": "vllm.model_executor.models.utils",
                "qwen2_5_omni_thinker": "vllm.model_executor.models.qwen2_5_omni_thinker",
                "vocab_parallel_embedding": "vllm.model_executor.layers.vocab_parallel_embedding",
            }.items()
        },
        "config_sha256": sha256(args.config) if args.config else None,
        "seed": args.seed,
        "script_sha256": sha256(__file__),
        "harness_sha256": sha256(SCRIPT_DIR / "embedding_harness.py"),
        "source_sha256": sha256(source),
        "source_path": str(source),
        "vllm_path": vllm.__file__,
        "omni_path": vllm_omni.__file__,
        "git_head": git_value(args.source_root, "rev-parse", "HEAD"),
        "git_dirty": bool(git_value(args.source_root, "status", "--porcelain")),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "blocks": args.blocks,
        "input_pool_size": args.input_pool_size,
        "measurement_protocol": "separate synchronized wall and CUDA-event passes for each block; per-call us",
        "scope": "real thinker method; TP1 embedding; random weights; synthetic encoder outputs; no model forward",
    }


def invoke(model, case, embeddings):
    return model.embed_input_ids(case.input_ids, multimodal_embeddings=embeddings, is_multimodal=case.is_multimodal)


def prepare(pool, iterations, offset=0):
    return [
        (pool[(offset + index) % len(pool)], pool[(offset + index) % len(pool)].fresh_embeddings())
        for index in range(iterations)
    ]


def run_prepared(model, prepared):
    for case, embeddings in prepared:
        invoke(model, case, embeddings)


def preflight(model, pool, harness, variant):
    counts = []
    for case in pool:
        model._clear_deepstack_input_embeds(case.input_ids.numel())
        with patch.object(model.language_model, "embed_input_ids", wraps=model.language_model.embed_input_ids) as spy:
            result = invoke(model, case, case.fresh_embeddings())
            counts.append(spy.call_count)
        expected_count = 2 if variant == "baseline" and case.name not in {"text", "empty", "interleaved"} else 1
        if counts[-1] != expected_count:
            raise AssertionError(f"{variant}/{case.name}: expected {expected_count} LM calls, got {counts[-1]}")
        harness.assert_case_output(model, case, result)
    return counts


def time_blocks(model, pool, args, torch):
    measurements = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()
    for block in range(args.blocks):
        prepared = prepare(pool, args.iterations, block)
        torch.accelerator.synchronize()
        begin = time.perf_counter_ns()
        run_prepared(model, prepared)
        torch.accelerator.synchronize()
        wall_us = (time.perf_counter_ns() - begin) / 1000 / args.iterations

        # The first pass mutated its lists; prepare NEW lists for this pass.
        prepared = prepare(pool, args.iterations, block)
        torch.accelerator.synchronize()
        start.record()
        run_prepared(model, prepared)
        end.record()
        end.synchronize()
        cuda_interval_us = start.elapsed_time(end) * 1000 / args.iterations
        measurements.append({"block_id": block, "wall_us": wall_us, "cuda_interval_us": cuda_interval_us})
    return measurements


def profile_case(model, pool, args, case_id, torch):
    profile_dir = args.profile_dir or args.output.parent / f"{args.output.stem}-traces"
    profile_dir.mkdir(parents=True, exist_ok=True)
    path = profile_dir / f"{case_id.replace(':', '-')}.json"
    if path.exists():
        raise FileExistsError(path)
    original = model.language_model.embed_input_ids

    def traced(ids):
        with torch.profiler.record_function("qwen3_lm_embedding"):
            return original(ids)

    prepared = prepare(pool, args.profile_steps)
    with patch.object(model.language_model, "embed_input_ids", new=traced):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as profiler:
            for case, embeddings in prepared:
                with torch.profiler.record_function("qwen3_thinker_embed_input_ids"):
                    invoke(model, case, embeddings)
                profiler.step()
            torch.accelerator.synchronize()
    profiler.export_chrome_trace(str(path))
    return str(path.resolve())


def main():
    args = arguments()
    if sys.platform != "linux":
        raise RuntimeError("Run this experiment in a Linux/CUDA environment")
    if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") not in {"", "0"}:
        raise RuntimeError("Unset CUDA_LAUNCH_BLOCKING for this benchmark")
    args.source_root = args.source_root.resolve()
    if not (args.source_root / "vllm_omni").is_dir():
        raise ValueError("--source-root must be a vllm-omni repository checkout")
    sys.path.insert(0, str(args.source_root))
    import torch

    if torch.version.cuda is None or not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("A working CUDA GPU is required; CPU fallback is disabled")

    # Import the selected source tree only after pinning custom-op registration
    # order, just as make_thinker does before importing the model.
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

    bootstrap_vllm_layer_custom_op_modules()
    from vllm_omni.platforms import current_omni_platform

    args.device = torch.device(args.device)
    current_omni_platform.set_device(args.device)
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("The selected CUDA device must support native BF16 for --dtype bfloat16")
    harness = load_harness()
    shape = load_shape(harness, args.config)
    dtype = getattr(torch, args.dtype)
    result = {
        "schema_version": SCHEMA,
        "variant": args.variant,
        "mode": args.mode,
        "pair_id": args.pair_id,
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for scenario in args.scenarios:
            for num_tokens in args.tokens:
                mm_tokens = 0 if scenario in {"text", "empty"} else int(num_tokens * args.mm_fraction) // 2 * 2
                model = harness.make_thinker(
                    shape,
                    device=args.device,
                    dtype=dtype,
                    seed=args.seed,
                    backend=args.embedding_backend,
                    deepstack=scenario != "vision_no_deepstack",
                    buffer_capacity=max(args.tokens),
                )
                if "metadata" not in result:
                    result["metadata"] = metadata(args, shape, model, torch)
                    print(json.dumps(result["metadata"], indent=2), flush=True)
                pool = [
                    harness.build_case(
                        scenario,
                        num_tokens,
                        mm_tokens,
                        shape,
                        device=args.device,
                        dtype=dtype,
                        mask_device=args.device if args.mask_device == "cuda" else "cpu",
                        seed=args.seed + index,
                    )
                    for index in range(args.input_pool_size)
                ]
                counts = preflight(model, pool, harness, args.variant)
                case_id = f"{scenario}:n{num_tokens}:m{mm_tokens}"
                row = {
                    "case_id": case_id,
                    "scenario": scenario,
                    "num_tokens": num_tokens,
                    "num_mm_tokens": mm_tokens,
                    "modality_tokens": {
                        modality: sum(
                            len(feature.positions) for feature in pool[0].features if feature.modality == modality
                        )
                        for modality in ("audio", "image", "video")
                    },
                    "case_sha256": hashlib.sha256("".join(case.fingerprint for case in pool).encode()).hexdigest(),
                    "deepstack": any(feature.deepstack is not None for feature in pool[0].features),
                    "lm_calls_per_invocation": counts,
                    "oracle_passed": True,
                }
                run_prepared(model, prepare(pool, args.warmup))
                torch.accelerator.synchronize()
                if args.mode == "timing":
                    row["measurements"] = time_blocks(model, pool, args, torch)
                    print(
                        f"{case_id}: median wall {statistics.median(m['wall_us'] for m in row['measurements']):.3f} us",
                        flush=True,
                    )
                elif args.mode == "profile":
                    row["trace_path"] = profile_case(model, pool, args, case_id, torch)
                else:
                    print(f"{case_id}: LM calls={counts}; numerical oracle passed", flush=True)
                result["results"].append(row)
    # Only a completely successful run is eligible for comparison.
    result["status"] = "complete"
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
