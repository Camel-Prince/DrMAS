set -x

# =============================================================================
# DrMAS Math 训练 - 2×A800 (80GB) 极简配置
# 
# 关键优化:
#   1. model_sharing=True → Solver+Verifier 共享 1 个模型 (1 个 vllm 实例)
#   2. GRPO → 无 Critic (省 ~6GB/GPU)
#   3. use_kl_loss=False → 无 Reference Model
#   4. optimizer_offload=True → Adam 状态卸载到 CPU
#   5. gradient_checkpointing=True → 降低激活值显存
#
# 显存估算 (Qwen2.5-3B, FP16, 2 GPU FSDP):
#   Rollout 阶段: ~3GB(权重) + ~32GB(KV cache@0.4) ≈ 35GB/GPU
#   Training 阶段: ~1.5GB(FSDP权重) + ~1.5GB(梯度) + ~5GB(激活) ≈ 8GB/GPU
#   峰值 ≈ 35GB/GPU ← 80GB 非常充裕
#
# 如果想用更大的模型 (Qwen3-4B), 修改 model_ids 即可, 2×A800 同样能跑
# =============================================================================

MODE=${1:-train}
if [ "$MODE" == "eval" ] || [ "$MODE" == "evaluation" ]; then
    echo "Running in evaluation mode"
    VAL_ONLY=True
    TRAIN_DATA="$HOME/data/drmas_math/train.parquet"
    VAL_DATA="$HOME/data/drmas_math/test.parquet"
    train_data_size=16
    val_data_size=32
    val_group_size=16
else
    echo "Running in training mode"
    VAL_ONLY=False
    TRAIN_DATA="$HOME/data/drmas_math/train.parquet"
    VAL_DATA="$HOME/data/drmas_math/test_sampled.parquet"
    train_data_size=16          # 比 4GPU 版减半 (32→16)
    val_data_size=110
    val_group_size=1
fi

###################### Algorithm Configurations #################
algorithm=grpo
group_size=8                    # GRPO group size, 8 已足够
group_by_agent_id=True          # ★ DrMAS 核心开关

##################### Agent Configurations #####################
agent_ids='["Solver Agent","Verifier Agent"]'

# ============= 选择模型 (取消注释你想用的) =============
# 方案A: Qwen2.5-3B (稳妥之选, 2×A800 非常轻松)
model_ids='["Qwen/Qwen2.5-3B","Qwen/Qwen2.5-3B"]'

# 方案B: Qwen3-4B (更强, 2×A800 仍然够用)
# model_ids='["Qwen/Qwen3-4B","Qwen/Qwen3-4B"]'

# 方案C: Qwen2.5-1.5B (最省显存, 留最大余量给长序列)
# model_ids='["Qwen/Qwen2.5-1.5B","Qwen/Qwen2.5-1.5B"]'
# ===========================================================

model_sharing=True              # ★ 关键: 2 Agent 共享 1 个模型实例, 省一半显存

orchestra_type=math
max_loop_num=2

# Agent-specific parameter override
# model_sharing=True 时两个 agent 共享权重, 但 lr 和 micro_batch 仍可独立设置
actor_optim_lr='[1e-6,1e-6]'
actor_ppo_micro_batch_size_per_gpu='[2,2]'   # 比 4GPU 版减半 (4→2)

model_name_tag=$(jq -r '.[]' <<< "$model_ids"  | awk -F/ '{print $NF}' | tr '[:upper:]' '[:lower:]' | tr '-' '_' | paste -sd_)
experiment_name="drmas_share${model_sharing}_${model_name_tag}_2gpu"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.truncation='middle' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=null \
    actor_rollout_ref.actor.optim.lr=null \
    +agent.agent_specific_parameters.actor.optim.lr=$actor_optim_lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size=True \
    actor_rollout_ref.actor.ppo_mini_update_num=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    +agent.agent_specific_parameters.actor.ppo_micro_batch_size_per_gpu=$actor_ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.group_by_agent_id=$group_by_agent_id \
    env.env_name=math \
    env.seed=0 \
    env.rollout.n=$group_size \
    env.rollout.val_n=$val_group_size \
    agent.agent_ids="$agent_ids" \
    agent.model_ids="$model_ids" \
    agent.model_sharing=$model_sharing \
    agent.orchestra_type=$orchestra_type \
    agent.orchestra.math.max_loop_num=$max_loop_num \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='DrMAS_math' \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 \
    trainer.val_only=$VAL_ONLY \
    trainer.val_before_train=True
