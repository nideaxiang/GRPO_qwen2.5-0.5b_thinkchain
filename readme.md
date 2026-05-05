# 🧠 ThinkChain - 基于TRL的GRPO训练框架

一个使用 **TRL (Transformer Reinforcement Learning)** 库进行 **GRPO (Group Relative Policy Optimization)** 训练的项目，专门用于训练模型的思维链（Chain-of-Thought）能力。

## 📋 项目概述

本项目实现了：
- ✅ 基于 TRL 的 `GRPOConfig` 和 `GRPOTrainer` 完整配置
- 🎯 使用 GSM8K 数学问题数据集进行训练
- 🏆 多维度奖励函数设计
- ⚡ vLLM 加速生成支持
- 🔄 SFT 监督微调和 GRPO 强化学习两阶段训练流程

## 🚀 快速开始

### 安装依赖

```bash
pip install torch trl modelscope datasets swanlab
```

### 训练流程

#### 1. SFT 监督微调（第一阶段）

```bash
python main.py \
    --task sft_train \
    --checkpoint_dir ./outputs/sft_checkpoint \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --split_half first_half \
    --epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --bf16
```
swanlab记录了训练过程：https://swanlab.cn/@duan_daniel/thinkchain/runs/p1ucmpon738406aypna7y/chart
<img width="2132" height="761" alt="image" src="https://github.com/user-attachments/assets/1f37032b-a40a-4f1b-be4b-a89080dc9fc3" />





#### 2. GRPO 强化学习训练（第二阶段）

```bash
python main.py \
    --task grpo_train \
    --checkpoint_dir ./outputs/grpo_checkpoint \
    --model_name_or_path ./outputs/sft_checkpoint \
    --split_half second_half \
    --epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_generations 8 \
    --max_prompt_length 256 \
    --max_completion_length 256 \
    --learning_rate 5e-6 \
    --bf16 \
    --use_vllm
```

## ⚙️ 核心配置详解

### 1. GRPOConfig 配置

`GRPOConfig` 继承自 `transformers.TrainingArguments`，新增了 GRPO 特有的参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `output_dir` | 模型检查点保存目录 | 必填 |
| `learning_rate` | 学习率 | `5e-6` |
| `per_device_train_batch_size` | 每个设备的训练批次大小 | `2` |
| `gradient_accumulation_steps` | 梯度累积步数 | `4` |
| `num_generations` | 每个提示的生成样本数 | `8` |
| `max_prompt_length` | 提示的最大长度 | `256` |
| `max_completion_length` | 生成补全的最大长度 | `256` |
| `num_train_epochs` | 训练轮数 | `1` |
| `bf16` | 是否使用 bf16 混合精度训练 | `False` |
| `use_vllm` | 是否使用 vLLM 加速生成 | `False` |
| `vllm_gpu_memory_utilization` | vLLM 的 GPU 内存利用率 | `0.2` |
| `adam_beta1` | AdamW 优化器的 beta1 | `0.9` |
| `adam_beta2` | AdamW 优化器的 beta2 | `0.99` |
| `weight_decay` | 权重衰减 | `0.1` |
| `warmup_ratio` | 预热比例 | `0.1` |
| `lr_scheduler_type` | 学习率调度器类型 | `cosine` |
| `max_grad_norm` | 梯度裁剪最大范数 | `0.1` |
| `logging_steps` | 日志记录步数 | `10` |
| `save_strategy` | 检查点保存策略 | `steps` |
| `save_steps` | 检查点保存步数 | `100` |

**关键配置说明（grpo_train.py:10-33）：**

```python
training_args = GRPOConfig(
    output_dir=args.checkpoint_dir,
    learning_rate=args.learning_rate,
    num_generations=args.num_generations,           # 每个样本生成多少个候选
    max_prompt_length=args.max_prompt_length,       # 提示长度限制
    max_completion_length=args.max_completion_length, # 生成长度限制
    use_vllm=args.use_vllm,                         # 是否使用vLLM加速
    vllm_gpu_memory_utilization=args.vllm_gpu_ratio, # vLLM显存占用率
    # ... 其他标准训练参数
)
```

### 2. GRPOTrainer 配置

`GRPOTrainer` 是 TRL 提供的强化学习训练器，核心配置如下：

| 参数 | 说明 |
|------|------|
| `model` | 要训练的模型 |
| `processing_class` | 分词器 |
| `reward_funcs` | 奖励函数列表 |
| `args` | `GRPOConfig` 实例 |
| `train_dataset` | 训练数据集 |

**关键代码（grpo_train.py:48-57）：**

```python
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_funcs,  # 奖励函数列表
    args=training_args,
    train_dataset=get_gsm8k_dataset(...),
)
trainer.train()
```

## 🏆 奖励函数设计

项目包含 5 个精心设计的奖励函数（reward.py）：

| 奖励函数 | 权重 | 功能 |
|----------|------|------|
| `correctness_reward_func` | 2.0 | 答案正确性检查 |
| `int_reward_func` | 0.5 | 答案是否为整数 |
| `strict_format_reward_func` | 0.5 | 严格格式检查 |
| `soft_format_reward_func` | 0.5 | 宽松格式检查 |
| `xmlcount_reward_func` | 0.5 | XML 标签完整性 |

### 思维链格式要求

模型必须按照以下格式输出：

```xml
<think>
... 思考过程 ...
</think>
<answer>
... 最终答案 ...
</answer>
```

## 📁 项目结构

```
thinkchain/
├── main.py              # 主入口文件，包含命令行参数解析
├── grpo_train.py        # GRPO 训练核心逻辑
├── sft_train.py         # SFT 监督微调逻辑
├── reward.py            # 奖励函数定义
├── utils.py             # 工具函数（数据集加载等）
└── outputs/             # 模型检查点输出目录
```

## 📊 数据集

使用 **GSM8K** 数学问题数据集，支持：
- 数据集切分为两半（前半用于 SFT，后半用于 GRPO）
- 自动处理为思维链格式

## 🔧 核心参数调优建议

### 批次大小设置
- `num_generations` 必须能被全局批次大小整除
- 建议设置为 4、8 或 16

### 学习率
- GRPO 通常使用比 SFT 更小的学习率
- 推荐范围：1e-6 ~ 1e-5

### vLLM 加速
- 设置 `--use_vllm` 开启
- 调整 `vllm_gpu_ratio` 控制显存占用（建议 0.2-0.5）

## 📝 参考文献

- [TRL Library Documentation](https://huggingface.co/docs/trl/index)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [GSM8K Dataset](https://arxiv.org/abs/2110.14168)

## 📄 许可证

MIT License
