# DrMAS 项目工作区地图 & 快速复现指南

> **DrMAS** = Dynamic Reinforcement learning of Multi-Agent Systems  
> 基于 veRL 框架，支持多 LLM Agent 端到端 RL 后训练的系统。

---

## 1. 工作区结构地图

```
DrMAS/
├── agent_system/                  # ★ 多Agent系统核心
│   ├── agent/
│   │   ├── registry.py            # @AgentRegistry.register 装饰器注册机制
│   │   ├── utils.py               # BaseAgent 基类 (build_prompt, _generate_with_llm, call)
│   │   ├── agents/
│   │   │   ├── math/              # Solver Agent + Verifier Agent (迭代式)
│   │   │   └── search/            # Verifier + Search Agent + Answer Agent (层级式)
│   │   └── orchestra/
│   │       ├── math/              # MathMultiAgentOrchestra (solver↔verifier 循环)
│   │       └── search/            # SearchMultiAgentOrchestra (条件分支)
│   ├── environments/
│   │   ├── base.py                # EnvironmentManagerBase (reset/step/success_evaluator)
│   │   ├── env_manager.py         # 环境管理器工厂
│   │   ├── env_package/           # 具体环境实现 (math/search)
│   │   └── prompts/               # 环境 prompt 模板
│   ├── memory/
│   │   ├── base.py                # BaseMemory 抽象接口
│   │   └── memory.py              # SimpleMemory + SearchMemory (滑动窗口历史)
│   ├── multi_turn_rollout/
│   │   ├── rollout_loop.py        # ★ TrajectoryCollector + 核心rollout循环
│   │   └── utils.py               # 数据处理工具
│   └── reward_manager/
│       └── episode.py             # ★ EpisodeRewardManager (outcome-based奖励)
│
├── verl/                          # veRL 框架 (训练基础设施)
│   ├── trainer/
│   │   ├── main_ppo.py            # 训练入口
│   │   ├── ppo/
│   │   │   ├── ray_trainer.py     # ★ RayPPOTrainer.fit() 主训练循环
│   │   │   └── core_algos.py      # ★ Advantage 计算 (GRPO/GiGPO/GAE/RLOO)
│   │   └── config/
│   │       └── ppo_trainer.yaml   # 默认训练配置
│   ├── workers/                   # Ray Worker 实现 (Actor/Critic/Ref/Reward)
│   ├── single_controller/         # Ray 单控制器分布式调度
│   └── protocol.py                # DataProto 数据协议
│
├── examples/
│   ├── drmas_trainer/
│   │   ├── run_math.sh            # ★ Math训练脚本 (2-agent: Solver+Verifier, 4B)
│   │   └── run_search.sh          # ★ Search训练脚本 (3-agent: 7B)
│   ├── grpo_trainer/              # 单Agent GRPO 参考
│   ├── ppo_trainer/               # PPO 参考
│   └── dapo_trainer/              # DAPO 参考
│
├── recipe/                        # 复现方案 (dapo/prime/r1/spin/sppo)
├── data/drmas_math/               # 数学数据集
└── docs/drmas/                    # DrMAS 文档
```

---

## 2. Rollout 机制

### 2.1 核心流程 (`agent_system/multi_turn_rollout/rollout_loop.py`)

```
vanilla_multi_turn_loop():
  FOR step in range(max_steps):
    1. preprocess → input_ids + attention_mask
    2. orchestra.run() → 各 Agent 按编排逻辑生成 text_actions
    3. env.step(text_actions) → rewards, dones, infos
    4. 记录 batch 数据 (responses, rewards, active_masks, traj_uid, agent_id)
    5. 若全部 done → break
  RETURN batch_list, episode_rewards, episode_lengths, success, traj_uid
```

### 2.2 两种 Rollout 模式

| 模式 | 触发条件 | 特点 |
|------|---------|------|
| `vanilla_multi_turn_loop` | 默认 | 标准 multi-turn 交互 |
| `dynamic_multi_turn_loop` | `filter_groups.enable=True` & training | DAPO 风格：过采样 → 过滤零方差组 |

### 2.3 关键参数

| 参数 | 作用 | Math 示例 | Search 示例 |
|------|------|-----------|-------------|
| `env.max_steps` | 最大交互轮数 | 默认50 | `max_turn=4` |
| `env.rollout.n` | GRPO group size | 8 | 5 |
| `env.rollout.val_n` | 验证 pass@k 采样数 | 1/16 | 1/16 |
| `rollout.temperature` | 采样温度 (train) | 1.0 | 1.0 |
| `rollout.val_kwargs.temperature` | 采样温度 (val) | 0.6 | 0.6 |
| `rollout.val_kwargs.top_p` | val top-p | 0.95 | 0.95 |

### 2.4 多 Agent 编排

**Math (迭代式):**
```
for loop in range(max_loop_num):
    Solver 生成解答 (active & ~approved)
    → Verifier 审核 (approve/reject)
    → 更新 approved_vector
    → 全部 approved → 提前退出
```

**Search (层级式):**
```
Verifier 判断信息充分性
  → 不够: Search Agent 检索
  → 够了: Answer Agent 回答
```

---

## 3. Reward 机制

### 3.1 EpisodeRewardManager (`agent_system/reward_manager/episode.py`)

```python
# 核心设计: Outcome-based, 奖励只放在最后一个有效 token 上
reward_tensor[i, valid_response_length - 1] = score  # 其他 token 为 0
```

### 3.2 奖励链路 (在 `ray_trainer.py` 的 fit() 中)

```
1. episode_rewards ← env.success_evaluator() → info['won']
2. token_level_scores ← EpisodeRewardManager(batch) → 放在末尾 token
3. invalid_action_penalty ← token_level_scores[i, -1] -= coef * is_invalid[i]
4. KL penalty (可选) ← token_level_rewards = scores - beta * kld
5. 进入 advantage 计算
```

### 3.3 关键配置

| 参数 | Math | Search |
|------|------|--------|
| `use_invalid_action_penalty` | True | True |
| `invalid_action_penalty_coef` | **0.1** | **0.01** |
| `use_kl_loss` | False | False |
| `entropy_coeff` | 0.0 | — |
| `normalize_by_length` | 默认 False | 默认 False |

> **注意**: Math 场景 penalty 系数 (0.1) 远大于 Search (0.01)，因为 Math agent 格式违规影响更大。

---

## 4. Advantage 计算

### 4.1 调度器 (`verl/trainer/ppo/core_algos.py`)

```python
compute_advantage(data, adv_estimator, ...):
    GRPO    → compute_grpo_outcome_advantage()
    GiGPO   → compute_gigpo_outcome_advantage()
    GAE     → compute_gae_advantage_return()
    RLOO    → compute_rloo_outcome_advantage()
    GRPO_PASSK → compute_grpo_passk_outcome_advantage()
```

### 4.2 GRPO (DrMAS 默认算法)

```python
compute_grpo_outcome_advantage(
    token_level_rewards,    # (bs, resp_len)
    response_mask,          # (bs, resp_len)
    index,                  # prompt group IDs
    traj_index,             # trajectory IDs
    group_by_agent_id,      # ★ DrMAS 核心开关
):
    scores = token_level_rewards.sum(dim=-1)      # 聚合为标量
    # 按 group 归一化:
    #   group_by_agent_id=True  → key = f"{uid}_{agent_id}" (每 agent 独立归一化)
    #   group_by_agent_id=False → key = uid (prompt group)
    advantages = (score - group_mean) / (group_std + eps)
    return advantages * response_mask
```

### 4.3 GiGPO (Episode + Step 双层 Advantage)

```python
# Episode 级别: (episode_reward - mean) / std
# Step 级别: 通过 anchor_obs 对相同状态分组，(step_reward - step_mean) / step_std
# 合并: episode_adv + step_advantage_w * step_adv
```

### 4.4 ★★★ 关键开关: `group_by_agent_id`

```yaml
algorithm.group_by_agent_id: True   # 必须为 True 才能启用 Dr.MAS
```

- **True**: 各 Agent 在自己的组内归一化 advantage（Solver 和 Verifier 分别归一化）
- **False**: 所有 Agent 混在同一个 prompt group 里归一化（行为异质性导致归一化不合理）

---

## 5. 训练数据流完整链路

```
┌─────────────────────────────────────────────────────────────┐
│  RayPPOTrainer.fit() 主循环 (verl/trainer/ppo/ray_trainer.py) │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 采样 batch → DataLoader                                  │
│  2. multi_turn_loop() → rollout 生成 trajectories            │
│  3. split_batch_by_wg_ids() → 按 worker group 拆分           │
│  4. adjust_batch() → 应用每个 agent 的独立配置                │
│  5. compute_response_mask() → 计算 loss mask                 │
│  6. combine_batches() → 合并回                               │
│  7. compute_step_discounted_returns() (GiGPO only)           │
│  8. reward_fn(batch) → token_level_scores                    │
│  9. apply_invalid_action_penalty()                           │
│  10. KL penalty / token_level_rewards                        │
│  11. compute_log_prob() → old_log_prob (per agent)           │
│  12. compute_ref_log_prob() → ref_log_prob (if KL loss)      │
│  13. compute_advantage() → advantages & returns              │
│  14. actor_rollout_wg.update_actor() (per agent)             │
│  15. log metrics                                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Agent 配置技巧与注意事项

### 6.1 Per-Agent 参数覆盖

```bash
# 长度必须与 agent 数量一致！2 个 agent = 2 个值
+agent.agent_specific_parameters.actor.optim.lr=[1e-6, 1e-6]
+agent.agent_specific_parameters.actor.ppo_micro_batch_size_per_gpu=[4, 4]
```

### 6.2 ID 体系

| ID | 含义 | 用途 |
|----|------|------|
| `uid` | 环境组 ID（同一 prompt 的多次采样共享） | Advantage 分组 |
| `traj_uid` | 每次 rollout 的唯一 ID | 去重/追踪 |
| `agent_id` | Agent 角色名 (如 "Solver Agent") | Per-agent 归一化 |
| `wg_id` | Worker Group ID (模型分配) | 拆分 batch 做 log_prob / update |

### 6.3 model_sharing

- **False** (默认): 每个 agent 独立模型实例 → 更灵活，GPU 占用更高
- **True**: 多个 agent 共享同一模型 → 节省显存，但 agent 行为趋同

### 6.4 常见陷阱

1. **`actor_rollout_ref.rollout.n=1`**：veRL 内部的 n 要设为 1，实际 GRPO 采样由 `env.rollout.n` 控制
2. **`data.return_raw_chat=True`**：multi-turn 场景必须开启
3. **`data.truncation`**：Math 用 `'middle'`，Search 用 `'left'`
4. **response_mask 在 multi-turn 中使用 `loss_mask`** 而非 `attention_mask`
5. **Memory 长度**：Search 场景 `env.history_length` 控制记忆窗口大小

---

## 7. 3B 模型 A800 GPU 需求估算

### 7.1 显存拆解 (3B FP16 模型)

| 组件 | 显存 | 备注 |
|------|------|------|
| 模型权重 | ~6 GB | 3B × 2 bytes |
| Adam 优化器状态 | ~12 GB | param + momentum + variance |
| 梯度缓冲 | ~6 GB | gradient checkpointing 可降低 |
| 推理 KV cache | ~2-4 GB | 取决于 batch size & seq length |
| Rollout 激活值 | ~4-6 GB | 生成阶段 |
| **总计 (per GPU)** | **~34 GB** | **开启所有优化后** |

### 7.2 推荐配置

| 场景 | 最少 GPU | 推荐 GPU | 说明 |
|------|---------|---------|------|
| 单 3B 模型 (研究) | 2×A800 | **4×A800** | micro_batch=1→4 |
| 2×3B Agent (DrMAS Math) | 4×A800 | **4-8×A800** | agent_micro_batch=[2,2] |
| 3×3B Agent (DrMAS Search) | 8×A800 | **8-16×A800** | agent_micro_batch=[2,2,2] |
| 带 Reward Model | 8×A800 | 12×A800 | 独立 reward model GPU pool |

### 7.3 推荐的 4×A800 配置 (2×3B Agent)

```bash
trainer.n_gpus_per_node=4
trainer.nnodes=1

# Actor 训练
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null
+agent.agent_specific_parameters.actor.ppo_micro_batch_size_per_gpu='[2,2]'
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
actor_rollout_ref.model.enable_gradient_checkpointing=True

# Rollout 推理
actor_rollout_ref.rollout.name=sglang
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
```

### 7.4 OOM 应急降级策略 (优先级排序)

1. **降 micro_batch**: `4→2→1` (立竿见影，~30% 显存)
2. **开 optimizer_offload**: `True` (~10GB 节省)
3. **缩短序列**: `max_response_length 4096→2048` (~15% per 2048 tokens)
4. **开 ref model param_offload**: `ref.fsdp_config.param_offload=True` (~3GB)
5. **降 rollout 显存**: `gpu_memory_utilization 0.5→0.4`
6. **增加 tensor_parallel**: `tensor_model_parallel_size=1→2` (跨 GPU 分摊推理)

---

## 8. 快速启动 Checklist

```bash
# 1. 安装
pip install -e .
pip install -r requirements_sglang.txt

# 2. 准备数据
ls ~/data/drmas_math/  # train.parquet, test.parquet, test_sampled.parquet

# 3. 训练 (Math, 4 GPUs)
cd examples/drmas_trainer
bash run_math.sh train

# 4. 评估
bash run_math.sh eval

# 5. 关键监控指标 (wandb)
# - episode_rewards: 回合奖励趋势
# - success_rate: 各数据源成功率
# - policy_entropy: 策略熵 (不应崩塌)
# - advantage_mean/std: advantage 统计
```

---

## 9. 与标准 veRL 单Agent Multi-turn 的主要差异

| 维度 | 单 Agent Multi-turn | DrMAS Multi-Agent |
|------|---------------------|-------------------|
| Rollout | 直接 Actor → Env | Orchestra 编排多个 Agent → Env |
| Advantage | `group_by_agent_id=False` | `group_by_agent_id=True` ★ |
| 参数 | 全局统一 | Per-agent 独立 (lr, micro_batch) |
| Worker Group | 1 个 | 每个 model_id 1 个 (或共享) |
| Batch 处理 | 直接训练 | split → adjust → combine |
| Memory | 简单历史 | SearchMemory / SimpleMemory |
| 数据字段 | uid, traj_uid | + agent_id, wg_id |
