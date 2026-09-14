# Qwen3-Omni embedding validation on H100

所有执行命令均面向 **Linux + 独占 H100 + CUDA**。Mac 只用于编辑和静态检查，不安装实验依赖，不执行这些测试。当前脚本尚未在 H100 上运行，不能将代码完成当作回归通过或实测收益。

三个入口：

| 入口 | 用途 |
| --- | --- |
| `tests/model_executor/models/qwen3_omni/test_qwen3_omni_embed.py` | 25 个 GPU 回归用例：主输出、单次 LM lookup、DeepStack、OOV、padding/clear、重复调用、CPU/CUDA mask |
| `benchmarks/qwen3_omni/benchmark_qwen3_omni_embed.py` | 同一脚本测 baseline/patched；`count`、`timing`、`profile` 三种模式 |
| `validation/vllm-omni-7451/compare_results.py`（工作区目录） | 校验 A/B 可比性，输出 Markdown 总表和 JSON 统计 |

`embedding_harness.py` 是前两个入口共用的输入构造、最小模型构造和独立数值 oracle。测试默认 CUDA/BF16，另有 FP32 对照；benchmark 默认真实 vLLM `VocabParallelEmbedding`、BF16、TP1。没有 CUDA 时明确报错，不回退 CPU。

## 1. 在 H100 主机准备源码和唯一环境

以下约定主机目录为 `/work`。将工作区生成的 `validation/vllm-omni-7451/h100-validation.tar.gz` 上传到 H100 的 `/work/`；归档包含脚本和补丁，不含权重、虚拟环境或 Git 对象。在 H100 主机执行：

```bash
cd /work
mkdir -p opensource
git clone https://github.com/vllm-project/vllm-omni.git opensource/vllm-omni
git -C opensource/vllm-omni checkout --detach 01a2f93256975c7ff9565c5414b0fe0225bc4765
tar -xzf h100-validation.tar.gz
docker build -f opensource/vllm-omni/docker/Dockerfile.ci \
  -t qwen3-7451-validation opensource/vllm-omni
docker image inspect qwen3-7451-validation > validation/vllm-omni-7451/image-inspect.json
docker run --rm -it --gpus device=0 --ipc=host \
  -v /work:/work -w /work/opensource/vllm-omni \
  qwen3-7451-validation bash
```

镜像构建使用该 commit 的 CI Dockerfile，目标 vLLM 为 `v0.29.0`。只构建一次，A/B 共用容器环境；不要给两组分别解析依赖。已有同一版本的完整 Omni CUDA 环境时可复用，但只装 torch 不足以导入真实 thinker。首次构建需要联网下载依赖。

以下命令全部在这个 H100 容器中执行。保留同一个 shell：

```bash
set -euo pipefail
cd /work/opensource/vllm-omni
git worktree add --detach /work/baseline 01a2f93256975c7ff9565c5414b0fe0225bc4765
git apply --check /work/validation/vllm-omni-7451/qwen3-single-embedding.patch
git apply /work/validation/vllm-omni-7451/qwen3-single-embedding.patch
uv venv --system-site-packages .venv
mkdir -p /work/results/qwen3-7451
nvidia-smi -q > /work/results/qwen3-7451/nvidia-smi.txt
uv pip freeze --python .venv/bin/python > /work/results/qwen3-7451/packages.txt

PATCHED_ROOT=/work/opensource/vllm-omni
BASELINE_ROOT=/work/baseline
BENCHMARK="$PATCHED_ROOT/benchmarks/qwen3_omni/benchmark_qwen3_omni_embed.py"
RESULTS=/work/results/qwen3-7451
COMMON=(--tokens 512 2048 8192 --seed 42 --warmup 50 --iterations 50 --blocks 10 --input-pool-size 4)
```

如已同步了一份打过补丁的源码，跳过 `git apply`，先用 `snapshot.json` 的 SHA-256 核对生产文件。不要复制 Mac 的 `.venv` 到 H100。上述 `.venv` 继承构建镜像的包，不重新安装另一套 torch/vLLM。

局部实验不需要 30B checkpoint。默认明确使用配置预设：`V=152064`、`H=2048`、DeepStack 3 层。也可将固定模型 revision 的 `config.json` 放到 H100，并在 `COMMON` 中添加 `--config /work/model/config.json`；脚本会记录其 hash。两组必须选同一种配置来源。

## 2. 先回归与计数

```bash
PYTHONPATH="$PATCHED_ROOT" .venv/bin/python -m pytest \
  tests/model_executor/models/qwen3_omni/test_qwen3_omni_embed.py -q \
  --junitxml="$RESULTS/regression.xml"

for variant in baseline patched; do
  source_root="$BASELINE_ROOT"
  if [ "$variant" = patched ]; then source_root="$PATCHED_ROOT"; fi
  .venv/bin/python "$BENCHMARK" --source-root "$source_root" \
    --variant "$variant" --pair-id smoke --mode count --tokens 512 --warmup 0 \
    --output "$RESULTS/count-$variant.json"
done
```

pytest 针对 patched，普通多模态必须只查表一次。`count` 针对两组分别要求：baseline 普通多模态两次、patched 一次；text/empty/interleaved 两组均一次。每个输入都先与独立布局 oracle 精确比较主 embedding 和 DeepStack；任何不符都报错，不能进入正式汇总。

八个场景是 text、empty、audio、image、video、mixed（非交错 audio+image）、interleaved（video+audio）、vision_no_deepstack。非空场景默认 `M=N/4`，混合场景两种模态各占一半。`N` 是单次调度调用的 token 数，不是完整请求长度。

## 3. 五对独立进程，交替顺序测量

```bash
for run in 1 2 3 4 5; do
  variants=(baseline patched)
  if (( run % 2 == 0 )); then variants=(patched baseline); fi
  for variant in "${variants[@]}"; do
    source_root="$BASELINE_ROOT"
    if [ "$variant" = patched ]; then source_root="$PATCHED_ROOT"; fi
    .venv/bin/python "$BENCHMARK" --source-root "$source_root" \
      --variant "$variant" --pair-id "r$run" --mode timing "${COMMON[@]}" \
      --output "$RESULTS/timing-$variant-r$run.json" \
      > "$RESULTS/timing-$variant-r$run.log" 2>&1
  done
done
```

两组必须从**同一个** `BENCHMARK` 路径启动，只切换 `--source-root`。脚本前插源码目录，并检查真实 thinker 的导入位置；记录生产文件及关键 vLLM 依赖 hash。不要通过安装、卸载两份包切换 A/B。

默认 8 场景 × 3 个 N，每个条件 50 次预热、10 块 × 50 次调用，4 组输入轮换。正式计时没有 spy/profiler；wall 和 CUDA event 分两次执行，输入与 fresh list 在计时外准备，不累积 GPU 输出。每次复制外层 list，以保留下一次调用的原始多尺度输入。

主指标是前后同步的 wall µs/调用。CUDA event interval 单独列出，不能解释成 kernel 时间之和。交错分支保留原检测的 `.item()` 同步，可能由检测耗时主导；调用数不变并不要求测得耗时严格不变。GPU 上不要同时运行其他任务；保留 `nvidia-smi.txt` 中的型号、MIG、驱动等信息。

## 4. 独立采 trace

```bash
for variant in baseline patched; do
  source_root="$BASELINE_ROOT"
  if [ "$variant" = patched ]; then source_root="$PATCHED_ROOT"; fi
  .venv/bin/python "$BENCHMARK" --source-root "$source_root" \
    --variant "$variant" --pair-id trace --mode profile \
    --scenarios image --tokens 8192 --warmup 50 --profile-steps 5 \
    --output "$RESULTS/profile-$variant.json"
done
```

Chrome trace 写入各自的 `profile-*-traces/`。检查 `qwen3_thinker_embed_input_ids` 范围内的 `qwen3_lm_embedding` 次数，以及相关查表 kernel；DeepStack 正确性由 oracle 校验。trace 数据不参与 latency 汇总。

## 5. 在同一 H100 环境汇总

```bash
.venv/bin/python -m unittest discover \
  -s /work/validation/vllm-omni-7451 -p test_compare_results.py -v
.venv/bin/python /work/validation/vllm-omni-7451/compare_results.py \
  --baseline "$RESULTS"/timing-baseline-r*.json \
  --patched "$RESULTS"/timing-patched-r*.json \
  --output-md "$RESULTS/comparison.md" \
  --output-json "$RESULTS/comparison.json"
```

比较器只接收成功且 oracle/调用数通过的 timing JSON。缺 pair、缺 case、缺 block、输入/hash/环境不一致、不同组使用相同生产源码、组内混用源码、NaN/无效耗时都会报错。新实验用新结果目录，benchmark 拒绝覆盖旧 JSON。

总表分别报告 wall/CUDA interval 的 median、IQR、绝对节省和下降比例，保留负收益。95% bootstrap 区间以独立运行 `pair_id` 为单位，对各 pair 内块差值的中位数重采样；只有一对运行时不判断稳定收益。五对仍是有限样本，区间不保证覆盖所有机器漂移。

这里只量化“真实 thinker 方法 + TP1 embedding 层 + 随机权重 + 合成 encoder 输出”。调用数 2→1 不等于方法快 50%，局部耗时也不等于服务 TTFT 或吞吐提升。完整模型与服务 A/B 继续按 `/work/validation/vllm-omni-7451/validation-plan.md` 执行。
