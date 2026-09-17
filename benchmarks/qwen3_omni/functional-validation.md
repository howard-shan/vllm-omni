# 第 2 层：双卡真实模型功能验证

本节在 **Linux、同一台机器的两张 RTX PRO 6000 Blackwell 96GB** 上执行。沿用主 README 已准备的绝对路径变量：`PATCHED_ROOT`、`BASELINE_ROOT`、`RESULTS`、`PYTHON`、`MODEL_DIR`、`MODEL_ID`。`PYTHON` 指向两组共用的 `.venv/bin/python`；`MODEL_DIR` 是固定 revision 的完整 Instruct 权重，`MODEL_ID=Qwen/Qwen3-Omni-30B-A3B-Instruct`。

目标是验证真实输入经过预处理、Thinker、Talker、Code2Wav 后仍能正常输出。查表次数、embedding 精确等价和局部耗时由第 1 层负责；本节的 pytest 总耗时包含模型加载、媒体生成及音频转写，不能作为端到端性能收益。

这里的“第 2 层”是本实验的功能层，与仓库的 L1–L4 分级不同。**不要把下方 `--run-level` 改成 `core_model`**：该级别会向部署配置加入 `load_format: dummy`。基础在线和离线测试使用 `advanced_model`，扩展测试使用 `full_model`，才能加载真实权重。

## 1. 在所有 A/B 实验之前补齐依赖

先完成本节的依赖准备，再统一冻结环境并开始第 1～3 层。两组之间不升级或重新安装 vLLM、Omni、PyTorch 或测试依赖。以下假定主 README 中的四个 pytest 包及 Omni 运行依赖已经安装。

```bash
cd "$PATCHED_ROOT"
mkdir -p "$RESULTS"

# Ubuntu/Debian；已有这些系统包时跳过。root 用户去掉 sudo。
sudo apt-get update
sudo apt-get install -y ffmpeg espeak-ng libsndfile1 libgl1 libglib2.0-0

uv pip install --python "$PYTHON" \
  "pyttsx3==2.99" "opencc==1.4.1" \
  "openai-whisper==20250625"
uv pip check --python "$PYTHON"
uv pip freeze --python "$PYTHON" > "$RESULTS/packages-final.txt"
```

这些用例通过 `pyttsx3`/eSpeak 生成输入语音，通过 OpenCV/FFmpeg 构造视频；使用 vLLM 已依赖的 `opencv-python-headless`，不额外安装同样提供 `cv2` 的 `opencv-python`。Omni 的运行依赖已经包含 PyAV、SoundFile、ImageIO 等。无需为这几个用例安装整个 `.[dev]`，也不需要下载 Captioner 权重。

在线 `advanced_model`/`full_model` 的文本+音频断言会额外使用 **Whisper small** 转写生成的音频，并与输出文本比较。它不使用 `MODEL_PREFIX`，在同一个 Linux 账号的 Whisper 缓存中下载。Qwen3 主模型已改用 ModelScope；此处仍使用 OpenAI 官方的原生 `small.pt`，不能换成 Transformers、CTranslate2 或 ONNX 格式的同名模型。先在实验机器完成下载（无卡模式也可执行）：

```bash
"$PYTHON" - <<'PY'
import whisper

# 这里只预下载并检查评测权重；不启动 Qwen3 或运行性能实验。
whisper.load_model("small", device="cpu")
print("Whisper small is cached")
PY

sha256sum "${XDG_CACHE_HOME:-$HOME/.cache}/whisper/small.pt" \
  > "$RESULTS/whisper-small.sha256"
```

现有转写 helper 会选择空闲显存足够的 GPU，否则回退 CPU；它没有固定转写到 CPU 的环境变量。预下载时指定 `device="cpu"` 不会改变之后 pytest 的选择。因此功能测试与性能测试必须分开运行，待所有 pytest/转写进程退出后再计时。Whisper 误识别也可能造成断言失败，需保留实际文本、音频转写和错误，不能直接归为 embedding 回归。

## 2. 为两个工作树固定同一份本地模型和测试代码

`MODEL_PREFIX` 是 fixture 读取的**环境变量**，不是 pytest 的 `--model-prefix` 参数。fixture 将它与完整模型 ID 拼接。下面创建一个只含符号链接的目录，将两组都指向 `MODEL_DIR`：

```bash
export CUDA_VISIBLE_DEVICES=0,1
unset VLLM_TEST_PD_MODE

FUNCTIONAL_RESULTS="$RESULTS/functional-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$FUNCTIONAL_RESULTS"
export MODEL_PREFIX="$FUNCTIONAL_RESULTS/model-prefix"

"$PYTHON" - "$MODEL_PREFIX" "$MODEL_ID" "$MODEL_DIR" <<'PY'
from pathlib import Path
import sys

prefix, model_id, model_dir = sys.argv[1:]
assert model_id == "Qwen/Qwen3-Omni-30B-A3B-Instruct", model_id
target = Path(model_dir).resolve(strict=True)
assert (target / "config.json").is_file(), target
assert (target / "model.safetensors.index.json").is_file(), "Need full weights"
link = Path(prefix) / model_id
link.parent.mkdir(parents=True, exist_ok=True)
if link.exists() or link.is_symlink():
    assert link.resolve() == target, f"Existing link points elsewhere: {link}"
else:
    link.symlink_to(target, target_is_directory=True)
print(f"{link} -> {target}")
PY
```

两边分别执行各自工作树中的现成 E2E 测试，但先检查测试、fixture 和部署配置的实际文件内容一致。这样每个 fixture 启动服务时，`cwd` 也会落在对应工作树。**不要在 patched 工作树执行 patched 的测试文件、只把 `PYTHONPATH` 改成 baseline**：现有服务 helper 会把子进程 `cwd` 设回 helper 所在仓库，可能导致服务仍加载 patched 代码。

```bash
"$PYTHON" - "$BASELINE_ROOT" "$PATCHED_ROOT" "$FUNCTIONAL_RESULTS" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

baseline, patched, output = map(Path, sys.argv[1:])
selected = [
    "tests/helpers",
    "tests/model_tests/diffusion",
    "tests/model_executor/helpers.py",
    "tests/conftest.py",
    "tests/e2e/online_serving/test_qwen3_omni.py",
    "tests/e2e/online_serving/test_qwen3_omni_expansion.py",
    "tests/e2e/offline_inference/test_qwen3_omni.py",
    "vllm_omni/deploy/qwen3_omni_moe.yaml",
    "vllm_omni/deploy/qwen3_omni_moe_thinking.yaml",
    "pyproject.toml",
]
paths = set()
for root in (baseline, patched):
    for rel in selected:
        path = root / rel
        assert path.exists(), path
        files = path.rglob("*.py") if path.is_dir() else [path]
        paths.update(str(file.relative_to(root)) for file in files)
    # 包初始化及沿途 conftest 也属于测试执行上下文。
    for rel in list(paths):
        for parent in (root / rel).parents:
            if parent == root:
                break
            for name in ("__init__.py", "conftest.py"):
                if (parent / name).is_file():
                    paths.add(str((parent / name).relative_to(root)))
manifest = {}
for rel in sorted(paths):
    a, b = baseline / rel, patched / rel
    assert a.is_file() and b.is_file(), f"Missing test dependency: {rel}"
    assert a.read_bytes() == b.read_bytes(), f"A/B test code differs: {rel}"
    manifest[rel] = hashlib.sha256(a.read_bytes()).hexdigest()
(output / "test-code-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Verified {len(manifest)} identical test/config files")
PY
```

若一致性检查失败，先检查差异；不要忽略错误或临时给某一组修改测试。主 README 的生产代码检查也应通过：baseline 与 patched 的生产差异只能是本次修复。

## 3. 按相同用例顺序串行运行两组

部署使用现有双卡布局：GPU 0 为 Thinker，GPU 1 为 Talker + Code2Wav，stage 内 TP=1；阶段间使用共享内存 connector。基础在线测试使用 CI overlay，扩展测试参数化运行普通输出和 `async_chunk` 两种配置；其中纯文本→文本+音频、音频→文本+音频还各有一项 `batch_token_2048` 配置，将 Thinker/Talker 的 `max_num_batched_tokens` 降至 2048。两张 96GB 的容量提供了合理运行条件，但这份配置在 RTX 上仍需要实际启动确认，不能把 H100 的验证注释当作 RTX 的通过记录。

所选用例覆盖：

| 组 | 输入及输出 | 预期用例数/版本 |
| --- | --- | --- |
| online | 纯文本→文本、图像→文本与 prefix cache、混合模态→文本+音频、文本→文本+音频与 prefix cache | 4 |
| expansion | 独立音频、图像、视频、纯文本，以及 `use_audio_in_video=True` 的音视频交错；普通/异步分块配置，另含两项 `batch_token_2048` | 16 |
| offline | 视频→音频的离线调用路径 | 1 |

离线 helper 当前内部只调用 `core_model` 深度的响应断言，虽然 `--run-level=advanced_model` 让它加载真实权重。这一项用于补充离线路径 smoke，不能单独证明音频内容质量；音频非空和转写一致性由上面的在线用例检查。

```bash
(
  set -euo pipefail

  ONLINE=tests/e2e/online_serving/test_qwen3_omni.py
  EXPANSION=tests/e2e/online_serving/test_qwen3_omni_expansion.py
  OFFLINE=tests/e2e/offline_inference/test_qwen3_omni.py

  for variant in baseline patched; do
    source_root="$BASELINE_ROOT"
    if [ "$variant" = patched ]; then
      source_root="$PATCHED_ROOT"
    fi
    cd "$source_root"
    export PYTHONPATH="$source_root"

    # 当前解释器、pytest 和服务子进程共用此源码与环境。
    "$PYTHON" - "$source_root" <<'PY'
import importlib.util
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
for package in ("vllm_omni", "tests"):
    spec = importlib.util.find_spec(package)
    assert spec is not None and spec.origin is not None, package
    origin = Path(spec.origin).resolve()
    assert origin.is_relative_to(root), (package, origin, root)
    print(package, origin)
PY
    git rev-parse HEAD > "$FUNCTIONAL_RESULTS/$variant-commit.txt"
    sha256sum vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py \
      > "$FUNCTIONAL_RESULTS/$variant-source.sha256"

    "$PYTHON" -m pytest -s -v -n 0 \
      "$ONLINE::test_text_to_text_001" \
      "$ONLINE::test_mix_to_text_audio_001" \
      "$ONLINE::test_thinker_prefix_caching_text_output" \
      "$ONLINE::test_thinker_prefix_caching_audio_output" \
      -m 'advanced_model and cuda' --run-level advanced_model \
      --junitxml="$FUNCTIONAL_RESULTS/$variant-online.xml" \
      2>&1 | tee "$FUNCTIONAL_RESULTS/$variant-online.log"

    "$PYTHON" -m pytest -s -v -n 0 \
      "$EXPANSION::test_text_to_text_audio_001" \
      "$EXPANSION::test_text_audio_to_text_audio_002" \
      "$EXPANSION::test_large_image_to_text_audio_001" \
      "$EXPANSION::test_text_video_to_text_001" \
      "$EXPANSION::test_text_video_to_text_audio_001" \
      "$EXPANSION::test_audio_in_video_001" \
      "$EXPANSION::test_audio_in_video_002" \
      -m 'full_model and cuda' --run-level full_model \
      --junitxml="$FUNCTIONAL_RESULTS/$variant-expansion.xml" \
      2>&1 | tee "$FUNCTIONAL_RESULTS/$variant-expansion.log"

    "$PYTHON" -m pytest -s -v -n 0 \
      "$OFFLINE::test_video_to_audio" \
      -m 'advanced_model and cuda' --run-level advanced_model \
      --junitxml="$FUNCTIONAL_RESULTS/$variant-offline.xml" \
      2>&1 | tee "$FUNCTIONAL_RESULTS/$variant-offline.log"
  done
)
```

`-n 0` 禁用 pytest 并行，避免两组或多个模型实例抢卡。不要一边保留服务进程，一边运行下一层 benchmark。标准 stage helper 会用同一个 `sys.executable` 启动服务，继承 `PYTHONPATH`、`MODEL_PREFIX` 和可见 GPU；日志中的 `Launching OmniServerStageCli` 应展示本地模型链接及该版本的部署配置路径。

如果某个命令失败，代码块会停止并保留该组日志。重新测试时新建结果目录；不要覆盖失败结果或只保留某次成功。遇到 OOM/不支持的 kernel，先定位环境或部署配置；如需调整 eager、token 预算或并发，必须对两组使用同样配置并重新运行。默认配置中出现的 H100/B200 pytest 标记是 CI 资源标签，不会按 GPU 名称拒绝 RTX；本节不把通过这些标签选出的用例称为“H100 已验证”。

## 4. 核对实际执行数量，保留未通过项

```bash
"$PYTHON" - "$FUNCTIONAL_RESULTS" <<'PY'
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

root = Path(sys.argv[1])
for variant in ("baseline", "patched"):
    for group, expected in (("online", 4), ("expansion", 16), ("offline", 1)):
        path = root / f"{variant}-{group}.xml"
        cases = ET.parse(path).getroot().findall(".//testcase")
        failed = [case for case in cases if any(case.find(tag) is not None
                  for tag in ("failure", "error", "skipped"))]
        assert len(cases) == expected, (path, len(cases), expected)
        assert not failed, (path, [(case.get("name"), list(case)) for case in failed])
        print(f"{variant} {group}: {len(cases)} passed, no failures/errors/skips")
PY
```

这里既检查 pytest 退出码，也检查 JUnit 的实际用例数、失败和跳过。两组都通过只说明所列真实模型功能检查通过，不说明每个最终生成 token 或波形逐位一致，也不说明整个 issue 的其他模型已经修复。PR 应同时附第 1 层精确数值回归和计数结果；端到端收益另附第 3 层结果。

本方案尚未在双 RTX 上执行。检查结果、显存峰值、Blackwell kernel 兼容性和输出质量均以服务器实际日志为准。
