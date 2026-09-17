# 第三层：RTX PRO 6000 双卡服务性能验证

本方案在 **Linux、单机两张 RTX PRO 6000 Blackwell 96GB** 上执行。先完成
[主 README](README.md) 的环境、baseline 和局部验证，再完成真实模型功能验证。
下面的命令未在 GPU 上实测；启动成功、请求全部成功之后才能填写 PR 的结果。

复用仓库现有 `vllm bench serve --omni`、`openai-chat-omni` 后端和
`random` / `random-mm` 数据生成器，不引入另一个压测框架。现有
`tests/dfx/perf/scripts/run_benchmark.py` 适合 CI 的固定硬件阈值检查；其中
Qwen3 JSON 的 H100 阈值不适用于 RTX，因此这里直接调用相同 bench CLI，保存
RTX 上的 baseline/patched 成对数据。

## 1. 实验约束和指标

服务器始终运行完整三阶段配置：Thinker 在 GPU 0，Talker/Code2Wav 在 GPU 1。
`modalities=["text"]` 的请求只执行文本输出所需阶段；因此另有独立的
`image-audio` 扩展，真正触发语音生成链路。

| 场景 | 输入 | 输出 | 并发上限 | 每次正式请求数 |
| --- | --- | --- | --- | --- |
| `text` | 合成文本，目标 256 tokens | 文本 32 tokens | 1、4 | 100 |
| `image` | 相同文本长度 + 一张 512×512 合成图片 | 文本 32 tokens | 1、4 | 100 |
| `audio` | 相同文本长度 + 10 秒 48kHz 单声道合成音频 | 文本 32 tokens | 1、4 | 100 |
| `image-audio` | 同 `image` | 文本 + 语音 | 1 | 100 |

这些是固定形状的**合成性能负载**，不是语义准确率数据。媒体由已有数据生成器
在机器上生成，不需要另下图片/音频数据集。功能层使用真实模型和可解释的媒体检查模型行为。
实际 prompt token 数还包括 chat template 和多模态展开；不能把 256 写成完整
prefill 长度。保留详细结果中的 `input_lens`，并结合服务日志解释 token 统计口径。

- **TTFT**：客户端开始发送请求到收到首个文本 token 事件的时间，包含服务排队、
  预处理、编码、prefill 和本机 HTTP 开销；不是单独的 embedding 耗时。
- **E2EL**：该后端记录的请求开始到最后一个有效 SSE 数据事件的时间。语音场景
  包含音频生成，不等同于文本生成完成时间，也不等同于播放器播放结束时间。
- **request throughput**：本轮成功请求数 / 正式测量总时长，单位 req/s。
  **output throughput**：本轮生成的文本 token 数 / 正式测量总时长，单位 tok/s。
  `C=1` 主要看延迟；`C=4` 是固定并发下的吞吐，不宣称是机器最大吞吐。
- `audio_ttfp`：现有后端支持该指标，但打点在首个 `modality=audio` SSE 事件，
  检查非空音频之前。报告名为“首音频事件延迟”；**不把它写成首个可播放音频
  延迟**。后者需要另行验证非空、可解码音频事件后才能声称测得。

每个版本独立启动服务器，再用 20 条请求预热每个负载。正式测量无 profiler、无
调用计数补丁。共 5 组独立 run pairs，顺序为 AB、BA、AB、BA、AB。A 是 baseline，
B 是 patched；两者顺序使用相同两张 GPU、同一 Python 环境、模型 revision、部署
YAML 和 patched 版压测客户端。服务器重启不清理磁盘上的编译缓存；这里的独立
启动是进程生命周期独立，测量目标是预热后的服务性能，不是模型冷启动耗时。

关闭 prefix caching，防止重复前缀直接跳过需要验证的 prefill 路径；关闭 Thinker
的 MM processor cache，并为预热和正式请求使用不同种子。正式运行关闭 bench
的初始请求探测，避免它提前重放第一条正式输入。不同正式请求的合成媒体内容
不同。MM processor cache 与 GPU encoder cache 并非同一个缓存，不声称此设置
关闭了所有 encoder cache；使用不同媒体避免跨请求复用，prefix cache 关闭则
避免整个 embedding/prefill 被命中跳过。

## 2. 固定部署和留档

在 GPU 机器的 **Bash** 中执行。继承主 README 设置的绝对路径：
`PATCHED_ROOT`、`BASELINE_ROOT`、`RESULTS`、`PYTHON`、`MODEL_DIR`、`MODEL_ID`。
`MODEL_REVISION` 应是下载时记录的 ModelScope 模型 commit SHA，来源保存在
`$RESULTS/model-source.json`，主 README 将 SHA 另存于
`$RESULTS/model-revision.txt`；缺失时先核实本地模型来源，不能用当天 Hub 最新 SHA
代替已经下载的版本。

先用 `SERVING_PAIRS=1`、单独的 pilot 结果目录完整排练一对，核对请求长度、结果
字段和服务退出。之后在新目录跑正式五对。主实验是 `SERVING_SUITE=text`，共六个
case；语音扩展是 `SERVING_SUITE=audio`，只有 `image-audio-c1`，不要混入主实验
统计。默认五对主实验会发送 6,000 条正式请求和 1,200 条预热请求，启动十次服务器；
语音扩展另需十次启动、1,000 条正式和 200 条预热请求。先 pilot 再估计租赁时长。

```bash
# 第一次排练：设好后执行下文的准备和运行代码块。
export SERVING_SUITE=text SERVING_PAIRS=1
export SERVING_RESULTS="$RESULTS/serving-rtx-pro6000-text-pilot"

# 排练通过后：改为下面三个变量，重新执行准备和运行代码块。
# export SERVING_SUITE=text SERVING_PAIRS=5
# export SERVING_RESULTS="$RESULTS/serving-rtx-pro6000-text"

# 主实验完成后：完整链路语音扩展独立执行，最后独立汇总。
# export SERVING_SUITE=audio SERVING_PAIRS=5
# export SERVING_RESULTS="$RESULTS/serving-rtx-pro6000-audio"
```

```bash
export SERVING_DEPLOY="$SERVING_RESULTS/deploy.yaml"
export SERVING_PORT=18000
export MODEL_REVISION="${MODEL_REVISION:-$(cat "$RESULTS/model-revision.txt")}"
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=0
(
set -euo pipefail
# mkdir 无 -p：旧目录存在即失败，不覆盖旧协议、配置或结果。
mkdir "$SERVING_RESULTS"
test -n "$MODEL_REVISION"
test "$(uname -s)" = Linux
command -v setsid

# 保存同一份配置给 A/B 使用，不修改生产部署文件。
"$PYTHON" - <<'PY'
import os
from pathlib import Path
import yaml

source = Path(os.environ["PATCHED_ROOT"]) / "vllm_omni/deploy/qwen3_omni_moe.yaml"
config = yaml.safe_load(source.read_text())
config["dtype"] = "bfloat16"
for stage in config["stages"]:
    stage["enable_prefix_caching"] = False
    stage["max_num_seqs"] = 4
    stage.setdefault("engine_extras", {})["seed"] = 7451
    if stage["stage_id"] in (0, 1):
        stage["max_model_len"] = 8192
        stage["max_num_batched_tokens"] = 8192
    if stage["stage_id"] == 0:
        stage["mm_processor_cache_gb"] = 0
Path(os.environ["SERVING_DEPLOY"]).write_text(yaml.safe_dump(config, sort_keys=False))
PY

"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

suite = os.environ["SERVING_SUITE"]
pairs = int(os.environ["SERVING_PAIRS"])
assert suite in ("text", "audio") and pairs > 0
Path(os.environ["SERVING_RESULTS"], "protocol.json").write_text(
    json.dumps({"suite": suite, "pairs": pairs, "warmup": 20, "measured": 100}) + "\n"
)
PY

uv pip freeze --python "$PYTHON" > "$SERVING_RESULTS/packages.txt"
nvidia-smi -q > "$SERVING_RESULTS/nvidia-smi.txt"
nvidia-smi topo -m > "$SERVING_RESULTS/topology.txt"
git -C "$PATCHED_ROOT" diff --binary HEAD > "$SERVING_RESULTS/patched-uncommitted.diff"
git -C "$BASELINE_ROOT" diff --binary HEAD > "$SERVING_RESULTS/baseline-uncommitted.diff"
sha256sum "$SERVING_DEPLOY" "$SERVING_RESULTS/packages.txt" \
  > "$SERVING_RESULTS/config-and-packages.sha256"
)
```

保留默认 CUDA Graph 和 CUDA 平台的 rotary 配置，避免把 eager/graph 切换混进优化
收益。若 Blackwell 环境出现与本修改无关的启动问题，先修复或另设 eager 排障配置；
**A/B 必须使用同一份最终 YAML 重新跑完**。任一端 OOM、启动失败、输入超限都不算
有效 pair。机器上不得同时有另一个实验占 GPU；记录实际 GPU 名称、UUID 和功耗配置。

下面将已有生成器的正式输入保存为 JSONL，并记录内容 hash。这不是下载动作。
压测 CLI 用同一生成器、版本、tokenizer、种子和参数再次生成输入；此处留档用于
审查及复现。**不要在跑完一端后修改客户端源码、模型、依赖或输入参数。**

```bash
cd "$SERVING_RESULTS"
PYTHONPATH="$PATCHED_ROOT" "$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from vllm.benchmarks.datasets import RandomDataset
from vllm.tokenizers import get_tokenizer
from vllm_omni.benchmarks.data_modules.random_multi_modal_dataset import OmniRandomMultiModalDataset

root = Path(os.environ["SERVING_RESULTS"])
tokenizer = get_tokenizer(os.environ["MODEL_DIR"], trust_remote_code=True)
manifest = {}
for kind, seed, bucket in (
    ("text", 7451, None),
    ("image", 7452, {(512, 512, 1): 1.0}),
    ("audio", 7453, {(0, 10, 1): 1.0}),
):
    cls = RandomDataset if bucket is None else OmniRandomMultiModalDataset
    extra = {} if bucket is None else dict(
        base_items_per_request=1,
        limit_mm_per_prompt={kind: 1},
        num_mm_items_range_ratio=0.0,
        bucket_config=bucket,
    )
    requests = cls(random_seed=seed).sample(
        tokenizer=tokenizer, num_requests=100, input_len=256, output_len=32,
        prefix_len=0, range_ratio=0.0, **extra,
    )
    path = root / f"inputs-{kind}.jsonl"
    with path.open("w") as stream:
        for request in requests:
            stream.write(json.dumps({
                "prompt": request.prompt, "prompt_len": request.prompt_len,
                "output_len": request.expected_output_len,
                "multi_modal_data": request.multi_modal_data,
            }, sort_keys=True) + "\n")
    manifest[kind] = {"seed": seed, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
(root / "input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
```

## 3. 跑 5 组独立 A/B

整块复制到同一个 Bash 会话。先完成第二层 smoke，避免十次模型加载后才发现硬件
兼容问题。监听地址只使用 `127.0.0.1:18000`；若已有服务占用此端口，先更换端口
或结束自己之前启动的服务，不要对未知 PID 执行 `pkill`。

```bash
(
set -euo pipefail
cd "$SERVING_RESULTS"
port="$SERVING_PORT"
server_pid=""

stop_server() {
    if [[ -n "$server_pid" ]]; then
        # 只向本轮启动的服务发中断，让其清理 worker。
        kill -INT "$server_pid" 2>/dev/null || true
        for _ in $(seq 1 60); do
            if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
            sleep 1
        done
        # 超时仅处理本轮 setsid 创建的进程组，不匹配进程名称。
        if kill -0 -- "-$server_pid" 2>/dev/null; then
            kill -TERM -- "-$server_pid" 2>/dev/null || true
            for _ in $(seq 1 30); do
                if ! kill -0 -- "-$server_pid" 2>/dev/null; then break; fi
                sleep 1
            done
        fi
        if kill -0 -- "-$server_pid" 2>/dev/null; then
            kill -KILL -- "-$server_pid" 2>/dev/null || true
            for _ in $(seq 1 10); do
                if ! kill -0 -- "-$server_pid" 2>/dev/null; then break; fi
                sleep 1
            done
            printf 'Forced cleanup of this run: process group %s; inspect logs before rerunning.\n' "$server_pid" >&2
            server_pid=""
            return 1
        fi
        wait "$server_pid" || true
        server_pid=""
    fi
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

bench_case() {
    local kind="$1" concurrency="$2" run_dir="$3" phase="$4"
    local seed=7451 dataset=random count=100 modalities='["text"]'
    local extra=()
    case "$kind" in
        image|image-audio)
            seed=7452; dataset=random-mm
            extra=(--random-mm-base-items-per-request 1
                   --random-mm-num-mm-items-range-ratio 0
                   --random-mm-limit-mm-per-prompt '{"image":1}'
                   --random-mm-bucket-config '{"(512,512,1)":1.0}') ;;
        audio)
            seed=7453; dataset=random-mm
            extra=(--random-mm-base-items-per-request 1
                   --random-mm-num-mm-items-range-ratio 0
                   --random-mm-limit-mm-per-prompt '{"audio":1}'
                   --random-mm-bucket-config '{"(0,10,1)":1.0}') ;;
    esac
    if [[ "$kind" = image-audio ]]; then modalities='["text","audio"]'; fi
    if [[ "$phase" = warmup ]]; then seed=$((seed + 10000)); count=20; fi
    local cmd=("$PYTHON" -m vllm_omni.entrypoints.cli.main bench serve --omni
        --host 127.0.0.1 --port "$port"
        --model "$MODEL_DIR" --tokenizer "$MODEL_DIR" --trust-remote-code
        --backend openai-chat-omni --endpoint /v1/chat/completions
        --dataset-name "$dataset" --seed "$seed"
        --random-input-len 256 --random-prefix-len 0
        --random-output-len 32 --random-range-ratio 0
        --ignore-eos --num-prompts "$count" --num-warmups 0
        --ready-check-timeout-sec 0
        --request-rate inf --max-concurrency "$concurrency"
        --extra-body "{\"modalities\":$modalities,\"temperature\":0,\"seed\":7451}"
        --percentile-metrics ttft,e2el,audio_ttfp,audio_duration
        --metric-percentiles 50,95,99
        --save-result --save-detailed --result-dir "$run_dir"
        --result-filename "$kind-c$concurrency-$phase.json"
        "${extra[@]}")
    printf '%q ' "${cmd[@]}" > "$run_dir/$kind-c$concurrency-$phase.command"
    printf '\n' >> "$run_dir/$kind-c$concurrency-$phase.command"
    PYTHONPATH="$PATCHED_ROOT" "${cmd[@]}" \
        > "$run_dir/$kind-c$concurrency-$phase.log" 2>&1
    "$PYTHON" - "$run_dir/$kind-c$concurrency-$phase.json" "$count" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
result = json.loads(path.read_text())
count = int(sys.argv[2])
assert result["completed"] == count and result["failed"] == 0, path
assert len(result["errors"]) == count and not any(result["errors"]), path
assert result["output_lens"] == [32] * count, (path, result["output_lens"])
assert len(result["ttfts"]) == count, path
if path.name.startswith("image-audio-"):
    assert result["mean_audio_duration_s"] > 0, path
PY
}

for pair in $(seq 1 "$SERVING_PAIRS"); do
    order=(baseline patched)
    if (( pair % 2 == 0 )); then order=(patched baseline); fi
    for variant in "${order[@]}"; do
        source_root="$BASELINE_ROOT"
        if [[ "$variant" = patched ]]; then source_root="$PATCHED_ROOT"; fi
        run_dir="$SERVING_RESULTS/pair-$pair/$variant"
        # 避免把失败的旧结果和新结果混在一起，也避免覆盖实验留档。
        if [[ -e "$run_dir" ]]; then
            printf 'Result directory already exists: %s\n' "$run_dir" >&2
            exit 1
        fi
        mkdir -p "$run_dir"
        export RUN_DIR="$run_dir" SOURCE_ROOT="$source_root"
        PYTHONPATH="$source_root" "$PYTHON" - <<'PY'
import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
from pathlib import Path
import vllm_omni

root = Path(os.environ["SOURCE_ROOT"]).resolve()
import_path = Path(vllm_omni.__file__).resolve()
assert import_path.is_relative_to(root), (root, import_path)
with socket.socket() as sock:
    sock.bind(("127.0.0.1", int(os.environ["SERVING_PORT"])))
thinker = root / "vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py"
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
metadata = {
    "source_root": str(root), "omni_import": str(import_path),
    "commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "thinker_sha256": digest(thinker), "model_id": os.environ["MODEL_ID"],
    "model_revision": os.environ["MODEL_REVISION"], "model_dir": os.environ["MODEL_DIR"],
    "deploy_sha256": digest(os.environ["SERVING_DEPLOY"]),
    "inputs_sha256": digest(Path(os.environ["SERVING_RESULTS"]) / "input-manifest.json"),
    "protocol_sha256": digest(Path(os.environ["SERVING_RESULTS"]) / "protocol.json"),
    "packages_sha256": hashlib.sha256(subprocess.check_output(["uv", "pip", "freeze", "--python", os.environ["PYTHON"]])).hexdigest(),
    "packages": {name: importlib.metadata.version(name) for name in ("torch", "vllm", "transformers", "numpy")},
    "client_commit": subprocess.check_output(["git", "-C", os.environ["PATCHED_ROOT"], "rev-parse", "HEAD"], text=True).strip(),
    "client_sources": {str(path.relative_to(os.environ["PATCHED_ROOT"])): digest(path)
                       for path in sorted((Path(os.environ["PATCHED_ROOT"]) / "vllm_omni/benchmarks").rglob("*.py"))},
    "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,power.limit", "--format=csv,noheader"], text=True),
}
Path(os.environ["RUN_DIR"], "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY
        # 新进程内再次断言实际导入路径，然后进入正常 Omni CLI。
        setsid env PYTHONPATH="$source_root" "$PYTHON" -c '
import os, runpy
from pathlib import Path
import vllm_omni
assert Path(vllm_omni.__file__).resolve().is_relative_to(Path(os.environ["SOURCE_ROOT"]).resolve())
print("Validated server import:", vllm_omni.__file__, flush=True)
runpy.run_module("vllm_omni.entrypoints.cli.main", run_name="__main__")
' serve "$MODEL_DIR" --omni --host 127.0.0.1 --port "$port" \
            --deploy-config "$SERVING_DEPLOY" \
            --stage-init-timeout 1200 --init-timeout 1800 \
            > "$run_dir/server.log" 2>&1 &
        server_pid=$!
        export SERVER_PID="$server_pid"
        "$PYTHON" - <<'PY'
import os
import time
import urllib.error
import urllib.request

pid = int(os.environ["SERVER_PID"])
deadline = time.monotonic() + 2400
while time.monotonic() < deadline:
    os.kill(pid, 0)
    try:
        url = f"http://127.0.0.1:{os.environ['SERVING_PORT']}/health"
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                assert os.getpgid(pid) == pid, "Server is not leader of its own process group"
                break
    except (urllib.error.URLError, TimeoutError):
        time.sleep(2)
else:
    raise RuntimeError("Server readiness timed out; inspect server.log")
PY
        kinds=(text image audio)
        if [[ "$SERVING_SUITE" = audio ]]; then kinds=(image-audio); fi
        for kind in "${kinds[@]}"; do
            concurrencies=(1 4)
            if [[ "$kind" = image-audio ]]; then concurrencies=(1); fi
            for concurrency in "${concurrencies[@]}"; do
                bench_case "$kind" "$concurrency" "$run_dir" warmup
                bench_case "$kind" "$concurrency" "$run_dir" measured
            done
        done
        nvidia-smi > "$run_dir/gpu-after.txt"
        stop_server
    done
done
)
```

`image-audio` 的 `max_tokens=32` / `ignore_eos` 主要控制文本阶段，不能假设它使
Talker 和 Code2Wav 产生相同长度的音频。报告音频时同时给出输出音频时长/帧数；
若 A/B 语音长度明显不同，不将 E2EL 或音频吞吐差异归因于 embedding 优化。

预热日志和结果也要检查：任何失败、重试、worker 重启、OOM 或异常都使该轮不适合
直接比较。现有后端可能重试请求并重置该请求计时，`failed=0` 不能独自证明从未
重试。检查客户端和服务日志，并保留它们；发现异常时记录原因，修复后换一个新的
`SERVING_RESULTS` 目录重跑成对实验，不能只保留快的一端。

## 4. 校验结果并汇总

局部实验的 `compare_results.py` 有不同的 schema，**不能用它读取 serving JSON**。
下面使用当前 bench 已保存的指标，要求每个场景有 5 个完整 pairs、请求全部成功、
输出长度一致且为 32、GPU/模型/部署/依赖/输入相同。汇总中的改善百分比先在每对
内部计算，再取五对中位数；置信区间对 **run pairs** 做 bootstrap，不把 500 条请求
当成 500 次独立实验。五对只是起点，区间宽或跨零时需要增加独立 pairs。

```bash
"$PYTHON" - <<'PY'
import csv
import json
import math
import os
from pathlib import Path
import numpy as np

root = Path(os.environ["SERVING_RESULTS"])
protocol = json.loads((root / "protocol.json").read_text())
num_pairs = protocol["pairs"]
assert num_pairs >= 5, "Pilot only: fewer than 5 independent pairs"
cases = [(kind, c) for kind in ("text", "image", "audio") for c in (1, 4)]
if protocol["suite"] == "audio":
    cases = [("image-audio", 1)]
metrics = {"median_ttft_ms": -1, "p95_ttft_ms": -1, "median_e2el_ms": -1,
           "p95_e2el_ms": -1, "request_throughput": 1, "output_throughput": 1}
match_keys = ("model_id", "model_revision", "model_dir", "deploy_sha256", "protocol_sha256",
              "inputs_sha256", "packages", "packages_sha256", "client_commit", "client_sources", "gpu")
source_keys = ("source_root", "omni_import", "commit", "thinker_sha256")
all_rows = []
raw_rows = []
audio_rows = []
reference = None
source_reference = {}
for kind, concurrency in cases:
    pairs = []
    for pair in range(1, num_pairs + 1):
        versions = {}
        for variant in ("baseline", "patched"):
            directory = root / f"pair-{pair}" / variant
            meta = json.loads((directory / "metadata.json").read_text())
            common = {key: meta[key] for key in match_keys}
            if reference is None:
                reference = common
            assert reference == common, f"Environment changed: {directory}"
            source = {key: meta[key] for key in source_keys}
            assert source_reference.setdefault(variant, source) == source, directory
            for phase, expected in (("warmup", 20), ("measured", 100)):
                result = json.loads((directory / f"{kind}-c{concurrency}-{phase}.json").read_text())
                assert result["completed"] == expected and result["failed"] == 0, directory
                assert len(result["errors"]) == expected and not any(result["errors"]), directory
                assert result["output_lens"] == [32] * expected, (directory, result["output_lens"])
                assert len(result["ttfts"]) == expected, directory
                assert all(math.isfinite(x) and x > 0 for x in result["ttfts"]), directory
                if kind == "image-audio":
                    assert result["mean_audio_duration_s"] > 0, directory
                    assert result["median_audio_ttfp_ms"] > 0, directory
            versions[variant] = result
        assert versions["baseline"]["input_lens"] == versions["patched"]["input_lens"], (kind, pair)
        pairs.append(versions)
    case_metrics = dict(metrics)
    interpretation = "equal_text_output_length"
    if kind == "image-audio":
        case_metrics["median_audio_ttfp_ms"] = -1
        interpretation = "audio_request_lengths_not_verified_individually"
        for pair, versions in enumerate(pairs, 1):
            a, b = versions["baseline"], versions["patched"]
            aggregate_match = (a["total_audio_frames"] == b["total_audio_frames"]
                               and a["total_audio_duration_s"] == b["total_audio_duration_s"])
            if not aggregate_match:
                interpretation = "audio_lengths_differ_do_not_attribute_to_embedding"
            audio_rows.append({"pair": pair,
                "baseline_mean_audio_s": a["mean_audio_duration_s"],
                "patched_mean_audio_s": b["mean_audio_duration_s"],
                "baseline_total_audio_s": a["total_audio_duration_s"],
                "patched_total_audio_s": b["total_audio_duration_s"],
                "baseline_audio_frames": a["total_audio_frames"],
                "patched_audio_frames": b["total_audio_frames"],
                "aggregate_lengths_match": aggregate_match})
    for metric, direction in case_metrics.items():
        baseline = np.array([p["baseline"][metric] for p in pairs], dtype=float)
        patched = np.array([p["patched"][metric] for p in pairs], dtype=float)
        assert np.all(np.isfinite(baseline)) and np.all(baseline > 0), metric
        assert np.all(np.isfinite(patched)) and np.all(patched > 0), metric
        change = direction * (patched / baseline - 1) * 100
        bootstrap = np.random.default_rng(7451).choice(change, size=(10000, num_pairs), replace=True)
        low, high = np.percentile(np.median(bootstrap, axis=1), [2.5, 97.5])
        all_rows.append({"case": kind, "concurrency": concurrency, "metric": metric,
                         "baseline_median": float(np.median(baseline)),
                         "patched_median": float(np.median(patched)),
                         "paired_improvement_pct": float(np.median(change)),
                         "ci95_low_pct": float(low), "ci95_high_pct": float(high),
                         "independent_pairs": num_pairs, "interpretation": interpretation})
        for pair, (a, b, delta) in enumerate(zip(baseline, patched, change), 1):
            raw_rows.append({"case": kind, "concurrency": concurrency, "metric": metric,
                             "pair": pair, "baseline": float(a), "patched": float(b),
                             "improvement_pct": float(delta)})
assert source_reference["baseline"]["thinker_sha256"] != source_reference["patched"]["thinker_sha256"]
for filename, rows in (("summary.csv", all_rows), ("paired-metrics.csv", raw_rows),
                       ("audio-output-lengths.csv", audio_rows)):
    if not rows:
        continue
    with (root / filename).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
print(root / "summary.csv")
PY
```

语音扩展另输出 `audio-output-lengths.csv`。音频总时长/帧数不同，会在结果表标记
`audio_lengths_differ_do_not_attribute_to_embedding`。当前 serving JSON 不保存逐请求
音频长度，聚合长度相同也无法证明每条请求等长，因此始终保留这一限制；语音结果
用于观察完整链路性能，不独自作为 embedding 贡献的定量归因证据。

输出 token 校验若失败，先查请求参数是否传到 Thinker、usage 是否可靠、是否有特殊
停止条件；不要删除断言后继续宣称是等长 A/B。源代码路径通过进程内断言和日志确认，
commit/hash 证明代码来源；若本地有未提交的修改，要连同补丁留档，不能只报 commit。

PR 中分别放：第一层输出一致和 embedding 次数/耗时；第二层真实多模态与语音功能
结果；第三层六个文本输出场景和独立语音扩展的结果表、完整命令与原始数据。纯文本是控制组，本次
修复不应被描述为它必然提速。某个指标 CI 跨零就写“本负载下未检出稳定改善”；不能
把局部 kernel 收益相加推导 TTFT，也不能把 RTX 结果换算成 H100/H800 收益。
