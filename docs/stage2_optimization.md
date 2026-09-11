# CPU_V3 第二阶段：BF16 decode 优化

验收对象是 Nmoe、GPU 0 RTX A6000、CPU_CORES=3（原有 2 compute + 1 shared 放置）、WiC 前五条、batch=1、output_len=100、GPU_MEM=43、prefetch=false。基线为 CPU_V3 `06eb7e11881a53f1a3c3d4b93b72449f973d201d`。

环境实测：Python 3.13.2 free-threading、PyTorch 2.7.1+cu126、CUDA runtime 12.6、transformers 4.50.3、NVIDIA driver 570.211.01；CPU 为 Xeon Gold 6444Y。

最终默认配置两次完整五条复测通过：均值 **0.1459004 / 0.1443840 s/token**，均低于 0.16 s。相对穿插重测的原始 CPU_V3（0.1850600 s），延迟降低 **21.16% / 21.98%**。数值测试及源代码/日志验收审计通过；轻量逐条数据随代码提交于 [stage2_results.json](stage2_results.json)。

## 计时与数据

指标沿用 `utils.patcher` 包装的 inner model forward wall time：每条剔除 prefill，99 个 decode forward 的累计时间 / 99，再对完整五条做算术平均。它包括 GPU、CPU offload 和等待，不是只累加 CUDA kernel 的时间；LM head、采样和 tokenizer 原本不属于该 inner forward 指标。没有修改计时起止位置，只将最终日志精度提高到小数点后六位。

所有成功、失败假设和异常数据均保留。五条中不剔除任何 prompt，不取 best run，不把 profiler 运行作为验收数据。原始日志、源码快照、GPU0 monitor、SHA256、逐条 JSON 和总表位于仓库外层的 `阶段2优化/过程` 与 `阶段2优化/results.json`。较早基线 .25208 s 有明显 CPU 波动，不能单独用它夸大加速比。

| 运行 | prompt 0 | prompt 1 | prompt 2 | prompt 3 | prompt 4 | 五条均值 | 样本标准差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_02 | .182300 | .189100 | .183800 | .189700 | .180400 | .1850600 | .0041464 |
| final_01 | .147758 | .145764 | .143588 | .148519 | .143873 | .1459004 | .0022240 |
| final_02 | .144169 | .144897 | .142580 | .147586 | .142688 | .1443840 | .0020429 |

上述每条均为 99 个 decode token，各运行退出码 0。执行顺序为 final_01 → baseline_02 → final_02，各自独立进程；最终两轮十条合并均值 .1451422 s。dma_01 首条异常快（.116115 s）同样保留，但不用它作为最终结论，随后完整重测了两轮。异常慢的 baseline_01 与 arena_01 也保留在总表。

沿用原 CPU 动态放置：baseline_02/final_01 使用 compute=[1,2]、shared=3；final_02 使用 compute=[5,6]、shared=7，均 socket 0 的 2+1 物理核。未人为固定不同核或改变选核策略。GPU0 约每 5 秒采样一次，三个验收/参考运行均未采到其他计算进程；这并不排除采样间隙的系统波动。逐轮 CPU expert 均值、affinity 和监控采样数见 JSON。

## 保留的实现

| 改动 | 机制与不变量 | 回退开关（设为 0） |
|---|---|---|
| 单 token dispatch/combine | 每个唯一 expert 都读同一输入行，使用输入/权重视图；同顺序 add_ 替换单目的地 index_add_，保留每次 BF16 舍入 | `SMOE_DECODE_DIRECT` |
| GPU gate/up 合并 | 两组权重本已在缓存 storage 中相邻，建立零拷贝视图，只对单 token 合并独立行的线性投影；BF16 不变 | `SMOE_GPU_BF16_FUSED_GATE_UP` |
| 局部 CUDA Graph | 模型构建时捕获 515 个 GPU slot 和 28 个 shared expert 的固定形状计算；权重地址稳定，换槽通过原 storage DMA 更新内容 | `SMOE_EXPERT_GRAPH` |
| pinned arena | 每层 64,512 B：一行输入和最多八行输出；D2H 排在 shared expert 前，CPU 只等输入拷贝 event；H2D 源重用前等上次拷贝完成 | `SMOE_PINNED_DECODE` |
| 合并冗余 fence | 单 token、prefetch=false、默认 stream、非末层时，利用下一层 B0 `.tolist()` 的既有同步；末层、prefill、非默认 stream 与 prefetch=true 保留 fence | `SMOE_DECODE_STREAM_CHAIN` |
| 完整 DMA 成本 | load stream 记录开始/结束 event，在原 B8 drain 后消费；修正用“CPU 算完后的残余等待”低估完整搬运成本的问题 | `SMOE_MEASURE_DMA` |

六个开关默认开启。另外，后台 GPU 专家线程显式进入 `torch.no_grad()`，与主线程的推理用途一致。

没有量化、降精度、减少专家、剪 token、改变输入、缩短生成、启用 prefetch 或改变 CPU 核数。`CPU_load_management` 的贪心公式及 cache router/替换策略源码不变；CPU/GPU 分配使用更准确的负载测量，因此缓存运行轨迹可以随调度改变，如原实现在时延波动下也会改变。验证数值等价时固定输入、权重和选中专家，不能把不同缓存轨迹的生成文本要求为 bitwise 相等。

Graph 各自持有私有 pool 和输出缓冲；整个推理仍是现有单请求执行协议，slot 被 pin 到输出消费完毕。调用者不能把 graph 的借用输出保留到该 slot 下次调用后。多 token、不同 dtype/shape 和 grad-enabled 调用回退 eager。跨设备动态迁移和并发请求未作为新增支持。

## 正确性与资源

测试覆盖真实 dispatch/combine 的全 GPU、混合 CPU/GPU、全 CPU；norm on/off；H=17/3584，token=0/1/2/7；48 项输出和 router logits exact（包括空输入回退）。GPU gate/up 16 项包括真正非连续输入、cache storage 内容替换、prefill fallback，实际 max_abs=0。fused projection + graph 的 8 项激活更新/换槽/replay 测试 exact。三次原始 BF16 storage DMA 全量比较通过，event 消费及正耗时通过。

GPU gate/up 的多 token 初版曾在 128 行输入上超出预设 atol=.002/rtol=.02，已缩小到单 token；未放宽门禁保留该长输入路径。最终不依赖 bitwise 保证，允许同精度浮点运算的常规舍入差异。

28 层 pinned arena 总计 1,806,336 B，未复制专家权重。Graph 增加显存 pool 占用和启动捕获成本。两轮最终运行的 PyTorch 峰值 allocated=46,263,630,336 B、reserved=48,289,021,952 B；driver-visible 采样峰值分别 46,628 / 46,630 MiB，原始 baseline_02 为 46,496 MiB。原 GPU_MEM=43 是缓存容量计算参数，并非整个进程显存硬上限；缓存仍是原配置算出的 515 个 GPU slot，未靠增加缓存容量达标。两轮日志确认所有 515 个 slot 及 28 个 shared graph 均实际 replay。

## Profiler 与负结果

原生 profiler 采集 10 个 warmup token 后的 5 个 decode forward。优化中间态 `profile_arena` 显示 H2D 累计 567.386 ms，占累计 GPU 活动 72.49%；CPU aten::mm self 366.555 ms。累计活动可以互相 overlap，不能直接等同关键路径百分比。由此优先校准搬运成本，没有无依据重写已经运行 Flash Attention 的 attention 路径。

最终同一诊断窗口 `profile_final`：H2D 累计 438.702 ms（67.33%，317 次，含专家权重及小输出拷贝），中间态为 383 次；CPU aten::mm self 增至 410.421 ms。该变化与更准确的搬运代价使一部分 miss expert 留在 CPU 计算相符，但缓存轨迹与 profiler 开销会影响调度，不能将两份 trace 当成固定专家集合的算子加速比。最终窗口 140 次 Flash Attention 共 1.150 ms；搬运仍是主要 GPU 活动。带 profiler 的单条平均 .166454 s（19 个 decode token）仅作诊断，不用于性能验收。两份原生 trace 和 operator 表均在 `过程/profile_{arena,final}/`。

GIL 切换间隔 .5 ms 与 OpenMP 逐线程固定核两个假设均没有稳定收益，已撤销代码，保留日志和源快照。Nmoe 的 tokenizer 导入已启用 GIL；未强制禁用 GIL。未引入 Triton/CUTLASS 依赖。

按用户指定目录参考 gpu-operator-optimization-engineer、indexed-pack-reduce-fusion-engineer 的契约、舍入和逐层验收方法，以及 ops 的融合约束、work/flash-attention 的数值约束。llm-torch-profiler-analysis 统一入口只支持四个服务框架，实际分析 SMoE trace 报无法识别 framework；没有把 SMoE 伪装成 SGLang。使用原生 torch.profiler trace 和 key_averages 作为此环境的诊断回退，并记录该限制。

## 复现

在本 CPU_V3 工作树中运行（模型路径按本机设置）：

```bash
cd /home/guoying/SMoE/阶段2优化/CPU_V3
CUDA_VISIBLE_DEVICES=0 CONDA_ENV=Nmoe MODEL_NAME=qwenmoe \
MODEL_PATH=/mnt/data/zgy/qwen2_moe \
CONFIG_PATH=configs/qwen2moe_prefetch_false.json \
DATASET_PATH=/home/guoying/SMoE/datasets/SuperGLUE/WiC/val.jsonl \
INPUT_NUM=5 BATCH_SIZE=1 OUTPUT_LEN=100 \
GPU_MEM=43 CPU_CORES=3 LOG_LEVEL=INFO bash run.sh
```

正确性测试（激活 Nmoe 后）：

```bash
PYTHONPATH=. python tests/test_decode_direct.py
PYTHONPATH=. python tests/test_gpu_gate_up.py
PYTHONPATH=. python tests/test_decode_graph.py
PYTHONPATH=. python tests/test_dma_timing.py
```

诊断采集用 `tests/profile_decode.py`，通过 `SMOE_PROFILE_TRACE` 指定输出，传入相同 main.py 参数但一条、输出20；带 profiler 的时间不能用于 .16 s 验收。完整实验记录由外层 `run_experiment.py` 与 `summarize_experiments.py` 生成。

外层 `audit_acceptance.py` 检查两轮最终生产源码 SHA 与运行快照一致、完整五条日志 SHA 和 token 数、固定环境、配置、graph 命中和 GPU0 采样，再导出 `acceptance.json`；不含自身 commit hash 的便携副本随代码提交于 [stage2_validation.json](stage2_validation.json)。仅 CPU_V3 独立工作树的实现、配置、测试、报告和轻量数据纳入 commit，主目录 main 的已有改动未修改。
