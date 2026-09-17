# Qwen3-Omni：RTX PRO 6000 单机双卡验证

目标平台为 **Linux x86_64 + 两张 RTX PRO 6000 Blackwell 96GB + 原生 Python/CUDA 环境**。全部安装、下载、pytest、benchmark 和结果汇总都在这台机器执行；Mac 仅编辑和静态检查。以下流程尚未在 RTX 上实测，不能把脚本完成写成测试通过。

验证范围是 issue #7451 的 **Qwen3 thinker 重复 embedding** 改动。Qwen2.5 不在本次范围内。实验“三层”是证据划分，与仓库 CI 的 L1–L4 级别不同。

| 层次 | 入口 | 产物与结论 |
| --- | --- | --- |
| 第一层：局部正确性和性能 | 本文第 3 节；`test_qwen3_omni_embed.py`、`benchmark_qwen3_omni_embed.py`、`compare_results.py` | 独立 oracle、查表次数、方法 wall 时间、CUDA interval、trace；无需完整权重 |
| 第二层：完整模型功能 | [functional-validation.md](functional-validation.md) | 两卡三阶段的真实权重 E2E；文本、音频、图像、视频、交错输入及文本/音频输出 |
| 第三层：服务性能 A/B | [serving-validation.md](serving-validation.md) | 同一双卡服务配置下的请求延迟和吞吐；与局部结果分别报告 |

默认部署将 Thinker 放在 GPU 0，Talker + Code2Wav 放在 GPU 1，阶段间使用 SharedMemoryConnector。完整三阶段部署不需要为了使用双卡而把 Thinker 改成 TP2。每张卡的显存预算独立，不能把两张卡当作一个 192GB 分配空间。先以短输入、低并发启动，再增加负载；原部署 YAML 的“两张 H100 已验证”不代表 RTX 已验证。

## 1. 源码、baseline 和唯一 Python 环境

### 1.1 获取当前实验分支

先把本次修改提交并推送到自己的 GitHub 分支，然后在 GPU 主机操作。以下用现有 fork/分支举例；若分支名改变，修改 `FIX_BRANCH`。不要在 A/B 运行过程中 pull、升级依赖或改变生产代码。

```bash
bash
set -euo pipefail
export FORK_URL=https://github.com/howard-shan/vllm-omni.git
export FIX_BRANCH=fix/qwen3-omni-redundant-embedding
mkdir -p /work/opensource
cd /work/opensource
# 首次克隆；已有目录时跳过这一行，执行下面的更新步骤。
git clone --branch "$FIX_BRANCH" "$FORK_URL" vllm-omni
cd vllm-omni
```

已有克隆的更新步骤（工作树须干净）：

```bash
cd /work/opensource/vllm-omni
test -z "$(git status --porcelain)"
git switch "$FIX_BRANCH"
git pull --ff-only "$FORK_URL" "$FIX_BRANCH"
```

定义变量，后续三个文档在同一 Bash 会话按顺序执行。重开 shell 时重新设置这些变量，`RESULTS` 应指向已有实验目录；新实验使用新目录，不覆盖旧结果。

```bash
export PATCHED_ROOT="$(pwd -P)"
export BASELINE_COMMIT=01a2f93256975c7ff9565c5414b0fe0225bc4765
export BASELINE_ROOT="$(dirname "$PATCHED_ROOT")/vllm-omni-baseline-7451"
export RESULTS="$(dirname "$PATCHED_ROOT")/results-qwen3-7451-rtx-$(date -u +%Y%m%dT%H%M%SZ)"
export BENCHMARK="$PATCHED_ROOT/benchmarks/qwen3_omni/benchmark_qwen3_omni_embed.py"
export PYTHON="$PATCHED_ROOT/.venv/bin/python"
export MODEL_ID=Qwen/Qwen3-Omni-30B-A3B-Instruct
export MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen3-Omni-30B-A3B-Instruct}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/root/autodl-tmp/cache/modelscope}"
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
unset CUDA_LAUNCH_BLOCKING
mkdir "$RESULTS"
```

### 1.2 准备固定 baseline，避免 invalid reference

`invalid reference` 意味着本地无法解析该 Git 对象，可能是浅克隆或没有获取对应历史。先补对象，再创建 worktree；失败时停止，不继续复制文件，也不要静默换成 `HEAD^`。

```bash
if ! git cat-file -e "${BASELINE_COMMIT}^{commit}" 2>/dev/null; then
  if [ "$(git rev-parse --is-shallow-repository)" = true ]; then
    git fetch --no-tags --unshallow "$FORK_URL" "$FIX_BRANCH"
  else
    git fetch --no-tags "$FORK_URL" "$FIX_BRANCH"
  fi
fi
git cat-file -e "${BASELINE_COMMIT}^{commit}"
if [ ! -e "$BASELINE_ROOT" ]; then
  git worktree add --detach "$BASELINE_ROOT" "$BASELINE_COMMIT"
fi
test "$(git -C "$BASELINE_ROOT" rev-parse HEAD)" = "$BASELINE_COMMIT"
test -z "$(git -C "$BASELINE_ROOT" status --porcelain)"
git rev-parse HEAD > "$RESULTS/patched-commit.txt"
git -C "$BASELINE_ROOT" rev-parse HEAD > "$RESULTS/baseline-commit.txt"
git diff "$BASELINE_COMMIT" HEAD -- vllm_omni > "$RESULTS/production.diff"
```

检查 `production.diff` 应只有本次 thinker 改动。若 fetch 后仍找不到 SHA，通常需要找回包含该提交的分支历史；rebase/squash 后不能假设旧 SHA 仍在分支祖先中。先解决 baseline 身份再实验。

### 1.3 原生安装，无需 Docker

确认机器具有完整两张 GPU、足够主机 RAM/磁盘以及可用的 `/dev/shm`；多阶段共享内存不足也会阻止启动。下面只查询，不更改驱动或系统配置。

```bash
nvidia-smi
nvidia-smi topo -m
free -h
df -h /work /dev/shm
```

当前源码配套 **vLLM 0.29.0**，其默认 wheel 使用 CUDA 13.0，见[仓库安装文档](../../docs/getting_started/installation/gpu/cuda.inc.md)。以下固定 cu130，不混入系统 Python 包；使用 [uv 独立安装器](https://docs.astral.sh/uv/getting-started/installation/) 避免依赖系统 pip。CUDA 13.x 常规运行需要 R580 或更新的兼容驱动，见 [NVIDIA 兼容说明](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-toolkit-release-notes/index.html)。`nvidia-smi` 显示的 CUDA 版本是驱动支持上限，不是 PyTorch 实际运行时版本。驱动不满足时让租赁平台提供匹配镜像/驱动，再进行安装。

```bash
cd "$PATCHED_ROOT"
# 已有 uv 时跳过；只在租用的 Linux 主机执行。
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/qwen3-7451-install-uv.sh
sh /tmp/qwen3-7451-install-uv.sh
export PATH="$HOME/.local/bin:$PATH"
# 已有本实验专用 .venv 时复用；不要覆盖其它项目的环境。
if [ ! -x "$PYTHON" ]; then uv venv --python 3.12 .venv; fi
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
VLLM_OMNI_TARGET_DEVICE=cuda VLLM_OMNI_VERSION_OVERRIDE=0.29.0.dev0 \
  uv pip install --python "$PYTHON" --torch-backend=cu130 \
  'vllm==0.29.0' -e "$PATCHED_ROOT" \
  'pytest==9.1.1' 'pytest-asyncio==1.4.0' 'pytest-xdist==3.8.0' 'pytest-mock==3.15.1' \
  'modelscope-hub==0.4.0'
# baseline 不重装；两组使用同一个环境及生成的包版本标识。
test -f "$PATCHED_ROOT/vllm_omni/_version.py"
cp "$PATCHED_ROOT/vllm_omni/_version.py" "$BASELINE_ROOT/vllm_omni/_version.py"
export PATH="$PATCHED_ROOT/.venv/bin:$PATH"
```

按第二层文档的“依赖准备”补齐功能测试依赖，按第三层文档准备其客户端依赖，**全部安装结束后**再冻结环境并开始正式三层实验。不要给 baseline 和 patched 分别解析一套依赖。

```bash
uv pip check --python "$PYTHON"
uv pip freeze --python "$PYTHON" > "$RESULTS/packages.txt"
nvidia-smi -q > "$RESULTS/nvidia-smi.txt"
nvidia-smi topo -m > "$RESULTS/topology.txt"
"$PYTHON" - <<'PY' | tee "$RESULTS/cuda-smoke.txt"
import platform
import torch
import vllm
from vllm_omni.platforms import current_omni_platform

assert platform.system() == "Linux"
assert torch.cuda.is_available()
assert current_omni_platform.get_device_count() == 2, "Expose exactly two complete GPUs"
print("python", platform.python_version(), "torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__, "compiled_arches", torch.cuda.get_arch_list())
for index in range(2):
    current_omni_platform.set_device(index)
    props = torch.cuda.get_device_properties(index)
    assert "RTX PRO 6000" in props.name.upper() and "BLACKWELL" in props.name.upper(), props.name
    assert (props.major, props.minor) == (12, 0), props
    assert props.total_memory > 90 * 1024**3, "Requires a full 96GB card, not a MIG slice"
    assert torch.cuda.is_bf16_supported(including_emulation=False)
    x = torch.ones((32, 32), device=f"cuda:{index}", dtype=torch.bfloat16)
    assert (x @ x)[0, 0].item() == 32
    print(index, props.name, props.total_memory, str(props.uuid))
PY
```

这只确认 PyTorch/CUDA 基础可用，vLLM attention/MoE/音频 kernels 的兼容性由第二层实际启动验证。不要把所有 `sm_120` 报错归因于本次 Python embedding 改动；环境修正必须同时用于两组，并重新冻结记录。

## 2. 从 ModelScope 下载并固定模型权重

第一层不需要完整模型。第二、三层共用 [ModelScope 的 Qwen3-Omni-30B-A3B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Instruct)，无需下载 Captioner。下载可在 **Ubuntu 22.04 无卡模式**完成，不导入或启动 vLLM，不要求 `nvidia-smi` 成功。环境安装沿用之前的 Python 3.12 / vLLM 0.29.0 / cu130 方案。

下载使用官方轻量 [ModelScope Hub SDK](https://github.com/modelscope/modelscope_hub/tree/v0.4.0)。先查询 ModelScope Git 仓库的提交 SHA，再交给支持 commit SHA 的 `HubApi.download_repo`；`git ls-remote` 只读取版本信息，不克隆权重，也无需 Git LFS。ModelScope 的 SHA 与 Hugging Face 的 SHA 分开记录，不复用旧的 `${MODEL_DIR}.revision`；`master` 本身不能作为固定版本。

如果已经按无卡安装方案保存了环境文件，先执行：

```bash
source /root/autodl-tmp/qwen3-7451.env
export MODELSCOPE_CACHE=/root/autodl-tmp/cache/modelscope
export MODEL_RECORD_DIR="$PREP_DIR"
```

已经从本文第 1 节进入 GPU 验证阶段时，改用 `export MODEL_RECORD_DIR="$RESULTS"`。下面的安装、下载和校验均在租用的 Linux 机器执行。若目标目录已有未登记来源的 Hugging Face 文件，给 `MODEL_DIR` 选一个新的空目录，并同步修改环境文件中的路径；不要混合两个来源的文件。

```bash
set -euo pipefail
mkdir -p "$MODEL_DIR" "$MODELSCOPE_CACHE" "$MODEL_RECORD_DIR"
uv pip install --python "$PYTHON" 'modelscope-hub==0.4.0'
uv pip check --python "$PYTHON"
uv pip freeze --python "$PYTHON" > "$MODEL_RECORD_DIR/packages.txt"

"$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from modelscope_hub import HubApi

model_id = os.environ["MODEL_ID"]
root = Path(os.environ["MODEL_DIR"]).resolve()
records = Path(os.environ["MODEL_RECORD_DIR"])
source_file = Path(str(root) + ".modelscope.json")
if source_file.exists():
    source = json.loads(source_file.read_text())
    assert source["provider"] == "modelscope" and source["model_id"] == model_id
else:
    assert not any(root.iterdir()), "Use an empty MODEL_DIR; do not mix download sources"
    remote = subprocess.check_output(
        ["git", "ls-remote", f"https://www.modelscope.cn/{model_id}.git", "HEAD"],
        text=True, timeout=60,
    ).split()
    assert len(remote) == 2 and remote[1] == "HEAD", remote
    source = {"provider": "modelscope", "model_id": model_id, "revision": remote[0]}
    assert re.fullmatch(r"[0-9a-f]{40}", source["revision"]), source
    source_file.write_text(json.dumps(source, indent=2) + "\n")

revision = source["revision"]
assert re.fullmatch(r"[0-9a-f]{40}", revision), revision
print(f"Downloading {model_id} @ {revision} from ModelScope", flush=True)
api = HubApi(endpoint="https://modelscope.cn")
files = [f for f in api.list_repo_files(model_id, "model", revision=revision) if f.type != "tree"]
assert files, "ModelScope returned an empty file list"
api.download_repo(
    model_id, "model", revision=revision, local_dir=root,
    cache_dir=os.environ["MODELSCOPE_CACHE"], max_workers=4,
)

# Check every remote file: a download progress bar alone is not proof of completeness.
manifest = []
for item in sorted(files, key=lambda f: f.path):
    local = root / item.path
    assert local.is_file() and local.stat().st_size == item.size, f"Missing/truncated: {local}"
    with local.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if item.sha256:
        assert digest == item.sha256, f"SHA256 mismatch: {local}"
    manifest.append(f"{digest}  {item.path}\n")

index = json.loads((root / "model.safetensors.index.json").read_text())
shards = set(index["weight_map"].values())
assert shards and all((root / name).is_file() for name in shards)
assert (root / "config.json").is_file()
(records / "model-source.json").write_text(json.dumps(source, indent=2) + "\n")
(records / "model-revision.txt").write_text(revision + "\n")
(records / "model-sha256.txt").write_text("".join(manifest))
print(f"Verified {len(files)} files, {len(shards)} weight shards: {root}")
PY
export MODEL_REVISION="$(cat "$MODEL_RECORD_DIR/model-revision.txt")"
du -sh "$MODEL_DIR"
```

中断或校验失败后重跑同一段：先复用 `.modelscope.json` 中的 SHA，再检查并补齐文件。若同一文件持续报 `Missing/truncated`，先将报错文件移出模型目录，再重跑；远端未提供 SHA256 时，SDK 可能跳过已经存在但不完整的文件。保留版本记录，不在 A/B 中间重新选择最新版本。成功后两组服务、tokenizer 和测试仍读取 `$MODEL_DIR`，无需设置 `VLLM_USE_MODELSCOPE`，也不需要改生产代码。

从无卡准备切换到正式实验时，将准备记录复制到本次 `RESULTS`，使第三层继续使用相同 SHA：

```bash
cp "$PREP_DIR/model-source.json" "$PREP_DIR/model-revision.txt" \
  "$PREP_DIR/model-sha256.txt" "$RESULTS/"
export MODEL_REVISION="$(cat "$RESULTS/model-revision.txt")"
```

全量 hash 会读取完整 checkpoint，应在计时前完成。Whisper small 是第二层音频转写用的辅助权重，仍按 [functional-validation.md](functional-validation.md) 从 OpenAI 官方下载原生 `small.pt`；ModelScope 上的 Transformers / faster-whisper 格式不能直接替代它。第三层素材准备见 [serving-validation.md](serving-validation.md)。下载期间不跑性能实验。

## 3. 第一层：局部回归、计数、性能与 trace

`embedding_harness.py` 构造真实 thinker 方法和真实 vLLM `VocabParallelEmbedding`，TP1、随机权重、合成 encoder 输出；不会运行整个模型 forward。默认 `V=152064`、`H=2048`、DeepStack 3 层，BF16 表约 594 MiB。也可在两组相同的 `COMMON` 参数中加 `--config "$MODEL_DIR/config.json"`，记录配置 hash。

局部 benchmark 不再按 H100 名称拒绝其它 CUDA GPU；检查 Linux/CUDA/native BF16 能力，并记录实际 GPU 名称、UUID、架构与显存。回归用例同样不锁定某个 SKU。此实验仍固定使用同一张 RTX GPU 0，第二张卡空闲，整机不要同时运行其它工作。

### 3.1 patched 回归和两组计数

```bash
cd "$PATCHED_ROOT"
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH="$PATCHED_ROOT" "$PYTHON" -m pytest \
  tests/model_executor/models/qwen3_omni/test_qwen3_omni_embed.py -q \
  --junitxml="$RESULTS/regression.xml"

# 同一回归的 CI 风格入口；上面通过后无需重复执行。
# PYTHONPATH="$PATCHED_ROOT" "$PYTHON" -m pytest \
#   tests/model_executor/models/qwen3_omni/test_qwen3_omni_embed.py \
#   -m 'core_model and gpu' --run-level=core_model -q

for variant in baseline patched; do
  source_root="$BASELINE_ROOT"
  if [ "$variant" = patched ]; then source_root="$PATCHED_ROOT"; fi
  "$PYTHON" "$BENCHMARK" --source-root "$source_root" \
    --variant "$variant" --pair-id smoke --mode count --tokens 512 --warmup 0 \
    --output "$RESULTS/count-$variant.json"
done
```

预期 25 个 patched 回归通过，覆盖主 embedding、DeepStack、OOV、padding/clear、重复调用以及 CPU/CUDA mask。baseline 普通多模态每次方法调用查表 2 次，patched 1 次；text/empty/interleaved 两组均为 1 次。计数基于当前选定源码及依赖，不沿用 issue 中未经核对的调用数。

八个场景：text、empty、audio、image、video、mixed（非交错 audio+image）、interleaved（video+audio）、vision_no_deepstack。每个输入先通过独立数值 oracle；任何失败都应停止正式计时。`N` 是单次调度调用的 token 数，不是整个请求长度。

### 3.2 五对独立进程，交替 AB/BA

```bash
COMMON=(--tokens 512 2048 8192 --seed 42 --warmup 50 --iterations 50 --blocks 10 --input-pool-size 4)
for run in 1 2 3 4 5; do
  variants=(baseline patched)
  if (( run % 2 == 0 )); then variants=(patched baseline); fi
  for variant in "${variants[@]}"; do
    source_root="$BASELINE_ROOT"
    if [ "$variant" = patched ]; then source_root="$PATCHED_ROOT"; fi
    "$PYTHON" "$BENCHMARK" --source-root "$source_root" \
      --variant "$variant" --pair-id "r$run" --mode timing "${COMMON[@]}" \
      --output "$RESULTS/timing-$variant-r$run.json" \
      > "$RESULTS/timing-$variant-r$run.log" 2>&1
  done
done
```

两组从同一个 `BENCHMARK` 路径启动，仅切换 `--source-root`，脚本检查真实 thinker 导入位置并记录源码 hash。正式 timing 不含 spy/profiler；wall 与 CUDA event 分开测量。主指标为同步后的 wall µs/调用；CUDA event interval 不是 kernel 时间之和。交错路径保留原有同步，不能要求其耗时严格不变。

### 3.3 单独采 trace，再汇总

```bash
for variant in baseline patched; do
  source_root="$BASELINE_ROOT"
  if [ "$variant" = patched ]; then source_root="$PATCHED_ROOT"; fi
  "$PYTHON" "$BENCHMARK" --source-root "$source_root" \
    --variant "$variant" --pair-id trace --mode profile \
    --scenarios image --tokens 8192 --warmup 50 --profile-steps 5 \
    --output "$RESULTS/profile-$variant.json"
done

"$PYTHON" -m unittest discover \
  -s "$PATCHED_ROOT/benchmarks/qwen3_omni" -p test_compare_results.py -v
"$PYTHON" "$PATCHED_ROOT/benchmarks/qwen3_omni/compare_results.py" \
  --baseline "$RESULTS"/timing-baseline-r*.json \
  --patched "$RESULTS"/timing-patched-r*.json \
  --output-md "$RESULTS/comparison.md" --output-json "$RESULTS/comparison.json"
```

在 `profile-*-traces/` 检查 `qwen3_thinker_embed_input_ids` 内 `qwen3_lm_embedding` 次数和对应查表 kernel。trace 不参与延迟汇总。

比较器拒绝缺 pair/case/block、失败 oracle/计数、不同 GPU UUID/环境/输入/hash、相同生产源码以及无效耗时。报告 median、IQR、绝对节省、下降比例及以独立 run pair 为单位的 bootstrap 区间，保留负收益；五对运行仍是有限样本。换卡、改参数或改软件后创建新实验组，不拼接旧结果。

## 4. 第二、三层执行顺序

恢复两卡可见性，然后依次执行两个文档；它们使用本文定义的路径和唯一 Python 环境。

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

1. [完整模型功能验证](functional-validation.md)：先完成真实权重 smoke，再扩展独立模态、混合/交错输入、流式和音频输出；保存两组 JUnit、日志及输出。仓库 `--run-level=core_model` 会使用 dummy 权重，不能作为真实权重通过证据；按文档选择实际级别。
2. [真实服务 A/B](serving-validation.md)：第二层通过后执行；使用固定模型、素材、部署配置和请求参数，服务启动/预热排除在正式计时之外，同一组双卡顺序跑 baseline/patched。

功能测试与性能测试不要同时运行，语音转写检查也不要与计时并行。若需降低上下文/并发或切换 attention/MoE/eager 设置，两组一起变更，记录新配置，再重新跑对应组。

## 5. PR 的证据与验收

| 证据 | 必须区分的口径 |
| --- | --- |
| 回归、count、trace | 输出/DeepStack 一致；普通多模态查表 2→1；每次方法调用为单位 |
| 局部 A/B 总表 | wall 和 CUDA interval 分开；不把查表减半写成方法快 50% |
| 完整模型功能结果 | 已执行的输入/输出分支、真实权重、失败/skip；不能用 dummy smoke 替代 |
| 服务 A/B 总表 | 请求延迟、吞吐、实际输出长度与错误数；不从局部耗时推算 TTFT 收益 |
| 可复现信息 | GPU/驱动/UUID、源码提交与 hash、包版本、权重 revision/hash、素材和部署配置、完整命令 |

先要求数值回归、调用数和完整功能通过，再根据实测报告收益。端到端区间包含零或负收益时如实说明，不挑选最好的一次。RTX 的结果只代表该平台与负载；提交前运行项目 pre-commit 与 precheck-pr，维护者要求的其它硬件覆盖仍通过项目 CI/补测完成。本次仅准备租用平台的验证流程，不修改项目 CI 资源调度，也尚未执行 GPU 实验。
