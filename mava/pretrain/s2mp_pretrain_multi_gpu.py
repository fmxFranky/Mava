import argparse
import math
import os
from functools import partial
from typing import Any, Callable, Dict, Tuple

import chex
import flashbax as fbx
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import plot_smax_env
from flashbax.vault import Vault
from flax import serialization
from flax.training import common_utils
from flax.training.train_state import TrainState
from tqdm import tqdm

from mava.networks.s2mp_network import S2MPNetwork

# 全局 finetune 标志，若为 True 则进入微调模式
FINETUNE = False

# -----------------------------------------------------------------------------
# Multi-GPU Utils
# -----------------------------------------------------------------------------


def create_learning_rate_fn(
    train_ds_size: int,
    train_batch_size: int,
    num_train_epochs: int,
    num_warmup_steps: int,
    learning_rate: float,
) -> Callable[[int], jnp.ndarray]:
    """创建学习率调度函数"""
    # 直接使用传入的参数，不重新计算
    num_train_steps = num_train_epochs * train_ds_size  # 这里train_ds_size实际是steps_per_epoch

    # 确保warmup_steps不超过总步数
    num_warmup_steps = min(num_warmup_steps, num_train_steps - 1)

    # 计算decay步数，确保为正数
    decay_steps = max(num_train_steps - num_warmup_steps, 1)

    print(f"Learning rate schedule:")
    print(f"  Total training steps: {num_train_steps}")
    print(f"  Warmup steps: {num_warmup_steps}")
    print(f"  Decay steps: {decay_steps}")

    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps
    )
    decay_fn = optax.cosine_decay_schedule(
        init_value=learning_rate, decay_steps=decay_steps, alpha=0.1
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps]
    )
    return schedule_fn


def remove_cross_episode_contamination(
    experience: Dict[str, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    """移除跨episode污染"""
    done_flags = experience["done"]  # (B, T, N)
    batch_size, sequence_length = done_flags.shape[:2]
    any_done = done_flags.any(axis=-1)
    cumsum_done = jnp.cumsum(any_done, axis=1)
    max_cumsum = jnp.max(cumsum_done, axis=1, keepdims=True)
    last_done_mask = (cumsum_done == max_cumsum) & any_done
    positions = jnp.arange(sequence_length)[None, :]
    last_done_positions = jnp.sum(last_done_mask * positions, axis=1)
    has_done = jnp.max(any_done, axis=1)
    keep_mask = (positions > last_done_positions[:, None]) | ~has_done[:, None]

    def apply_mask(value):
        expanded = keep_mask
        for _ in range(len(value.shape) - 2):
            expanded = expanded[..., None]
        return value * expanded

    return jax.tree.map(apply_mask, experience)


@partial(jax.jit, static_argnums=(1, 2, 3))
def generate_agent_masks(
    rng: chex.PRNGKey, batch_size: int, n_agents: int, num_random: int
) -> Tuple[chex.Array, chex.PRNGKey]:
    """生成agent masks - 多卡版本"""
    rng, subkey = jax.random.split(rng)

    # 每个智能体单独保留的mask
    eye_masks = jnp.eye(n_agents)[None, :, :]
    eye_masks = jnp.broadcast_to(eye_masks, (batch_size, n_agents, n_agents))

    # 随机mask
    def random_mask(k):
        n_to_mask = jax.random.randint(k, (batch_size,), minval=1, maxval=n_agents - 1)

        def make_single(bs_idx, num_mask, key_inner):
            key_perm, key_choice = jax.random.split(key_inner)
            perm = jax.random.permutation(key_perm, n_agents)
            indices = jnp.arange(n_agents)
            select_mask = indices < num_mask
            masked_idx = jnp.where(select_mask, perm, n_agents)
            mask = jnp.ones((n_agents,))
            valid_indices = jnp.where(masked_idx < n_agents, masked_idx, 0)
            valid_mask = masked_idx < n_agents
            mask = mask.at[valid_indices].multiply(1.0 - valid_mask.astype(jnp.float32))
            return mask

        keys_inner = jax.random.split(k, batch_size)
        masks = jax.vmap(make_single)(jnp.arange(batch_size), n_to_mask, keys_inner)
        return masks

    random_masks = jax.vmap(random_mask)(jax.random.split(subkey, num_random))
    random_masks = random_masks.transpose((1, 0, 2))
    masks = jnp.concatenate([eye_masks, random_masks], axis=1)
    return masks, rng


# -----------------------------------------------------------------------------
# Multi-GPU Training Functions
# -----------------------------------------------------------------------------


def create_train_state(rng, model, learning_rate_fn, grad_clip_norm=1.0):
    """创建训练状态"""
    dummy_obs = jnp.zeros((1, model.n_agents, model.K, model.obs_dim))
    dummy_action = jnp.zeros((1, model.n_agents, model.K, model.action_dim))
    dummy_mask = jnp.ones((1, model.n_agents))
    params = model.init(rng, dummy_obs, dummy_action, dummy_mask)

    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate_fn, weight_decay=1e-5, b1=0.9, b2=0.95, eps=1e-8),
    )
    return TrainState.create(params=params, tx=tx, apply_fn=model.apply)


def loss_fn(params, batch, model, rng):
    """损失函数 - 包含时序权重"""
    if FINETUNE:
        obs_seq, action_seq, _ = batch
        B, N, K, obs_dim = obs_seq.shape
        # 构造仅保留单个智能体的 mask
        eye = jnp.eye(N)
        agent_masks_eye = jnp.broadcast_to(eye[None, ...], (B, N, N))
        agent_masks_flat = agent_masks_eye.reshape(-1, N)
        # 重复序列数据以对应 mask
        obs_seq_rep = jnp.repeat(obs_seq, N, axis=0)
        action_seq_rep = jnp.repeat(action_seq, N, axis=0)
        # 获取预测
        pred_obs = model.apply(params, obs_seq_rep, action_seq_rep, agent_masks_flat, rngs={})
        # 计算最后 timestep 的平方误差
        sq_error_last = jnp.square(pred_obs - obs_seq_rep)[:, :, -1, :]
        mask_for_error = 1.0 - agent_masks_flat
        weighted_error = sq_error_last * mask_for_error[:, :, None]
        total_error = jnp.sum(weighted_error)
        denom = jnp.sum(mask_for_error) * obs_dim
        return total_error / denom
    else:
        obs_seq, action_seq, agent_masks = batch
        B, Aug, N = agent_masks.shape
        K = obs_seq.shape[2]

        obs_seq_rep = jnp.repeat(obs_seq, Aug, axis=0)
        action_seq_rep = jnp.repeat(action_seq, Aug, axis=0)
        agent_masks_flat = agent_masks.reshape(-1, N)

        pred_obs = model.apply(params, obs_seq_rep, action_seq_rep, agent_masks_flat, rngs={})
        sq_error = jnp.square(pred_obs - obs_seq_rep)

        # 智能体权重
        mask_for_loss = 1.0 - agent_masks_flat
        keep_for_loss = agent_masks_flat
        mask_weight = mask_for_loss[:, :, None, None]
        keep_weight = keep_for_loss[:, :, None, None]

        # 时序权重
        temporal_decay = 0.9
        time_indices = jnp.arange(K)
        temporal_weights = temporal_decay ** (K - 1 - time_indices)
        temporal_weights = temporal_weights / jnp.mean(temporal_weights)
        temporal_weights = temporal_weights[None, None, :, None]

        # 组合权重
        agent_weights = mask_weight * 2.0 + keep_weight * 1.0
        combined_weights = agent_weights * temporal_weights
        weighted_error = sq_error * combined_weights

        loss = jnp.sum(weighted_error) / jnp.sum(combined_weights)
        return loss


def train_step(state, batch, model, rng):
    """单步训练 - 将被pmap包装"""
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, batch, model, rng)

    # 跨设备平均梯度
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")

    state = state.apply_gradients(grads=grads)
    return state, loss


# 使用pmap包装训练步骤
p_train_step = jax.pmap(train_step, axis_name="batch", static_broadcasted_argnums=2)


@partial(jax.jit, static_argnums=(0,))
def eval_apply_fn(model, params, obs_seq, action_seq, agent_mask):
    """专门用于评估的apply函数，确保deterministic行为"""
    return model.apply(params, obs_seq, action_seq, agent_mask, rngs={})


def evaluate_s2mp(model, params, test_data_list, n_agents, action_space_type="discrete"):
    """评估S2MP模型性能"""
    test_losses = []

    for obs_seq_test, action_seq_test, test_masks in test_data_list:
        # 为每个agent创建单独的mask进行测试
        batch_size = obs_seq_test.shape[0]

        agent_losses = []
        for agent_idx in range(n_agents):
            # 创建只保留当前agent的mask (S2MP的mask机制：1=keep, 0=mask)
            agent_mask = jnp.zeros((batch_size, n_agents))
            agent_mask = agent_mask.at[:, agent_idx].set(1.0)  # 只保留当前agent

            # 预测其他agents的观测
            pred_obs = eval_apply_fn(model, params, obs_seq_test, action_seq_test, agent_mask)

            # 计算被mask的agents的重建误差 (对除了当前agent之外的所有agents)
            mask_for_loss = 1.0 - agent_mask  # 0=keep, 1=mask (计算这些位置的损失)
            obs_error = jnp.square(pred_obs - obs_seq_test)  # (B, N, K, obs_dim)

            # 应用mask并计算平均误差
            masked_error = obs_error * mask_for_loss[:, :, None, None]
            total_error = jnp.sum(masked_error)
            total_count = (
                jnp.sum(mask_for_loss) * obs_seq_test.shape[-2] * obs_seq_test.shape[-1]
            )  # N * K * obs_dim

            avg_error = total_error / total_count if total_count > 0 else 0.0
            agent_losses.append(float(avg_error))

        test_losses.append(np.mean(agent_losses))

    return np.mean(test_losses)


def prepare_batch_for_devices(batch, num_devices):
    """将batch准备为多设备格式"""
    obs_seq, action_seq, agent_masks = batch

    # 确保batch size能被设备数整除
    batch_size = obs_seq.shape[0]
    if batch_size % num_devices != 0:
        pad_size = num_devices - (batch_size % num_devices)
        obs_seq = jnp.concatenate([obs_seq, obs_seq[:pad_size]], axis=0)
        action_seq = jnp.concatenate([action_seq, action_seq[:pad_size]], axis=0)
        agent_masks = jnp.concatenate([agent_masks, agent_masks[:pad_size]], axis=0)

    # Reshape为 (num_devices, batch_per_device, ...)
    def reshape_for_devices(x):
        return x.reshape((num_devices, -1) + x.shape[1:])

    return (
        reshape_for_devices(obs_seq),
        reshape_for_devices(action_seq),
        reshape_for_devices(agent_masks),
    )


# -----------------------------------------------------------------------------
# Main Training Script
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU S2MP Pretraining")
    parser.add_argument("--vault_dir", type=str, default="/home/mxfeng/aaai25_projects/Mava/vaults")
    parser.add_argument("--vault_name", type=str, default="rec_mappo")
    parser.add_argument("--vault_uid", type=str, default="20250624093826")
    # 单卡视角的参数
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size per GPU (single-GPU perspective)"
    )
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for single GPU")
    parser.add_argument("--grad_clip_norm", type=float, default=10.0)
    parser.add_argument("--num_random_masks", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--steps_per_epoch", type=int, default=1000, help="Steps per epoch (single-GPU perspective)"
    )
    parser.add_argument("--warmup_epochs", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument(
        "--save_interval", type=int, default=1000, help="Save checkpoint every N steps"
    )
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file to load and resume training",
    )
    # 多卡策略参数
    parser.add_argument(
        "--lr_scaling",
        type=str,
        default="linear",
        choices=["linear", "sqrt", "constant"],
        help="Learning rate scaling strategy for multi-GPU",
    )
    # 测试集参数
    parser.add_argument(
        "--test_batches",
        type=int,
        default=128,
        help="Number of random test batches to evaluate (unused when using high-reward episodes)",
    )
    parser.add_argument(
        "--min_reward", type=float, default=1.5, help="Minimum cumulative reward for test episodes"
    )
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=3,
        help="Number of high-reward episodes to collect for testing",
    )
    parser.add_argument(
        "--test_output_folder",
        type=str,
        default="./test_plots",
        help="Directory to save test plots",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Enable fine-tune mode: only compute error for other agents at last timestep",
    )
    args = parser.parse_args()

    # 根据命令行参数设置全局 FINETUNE 标志
    global FINETUNE
    FINETUNE = args.finetune

    # 检查可用设备并自动扩展参数
    num_devices = jax.device_count()
    print(f"Available devices: {num_devices}")
    print(f"Device list: {jax.devices()}")

    # 自动扩展超参数到多卡
    # 1. 总batch size = 单卡batch size × 设备数
    total_batch_size = args.batch_size * num_devices
    per_device_batch_size = args.batch_size

    # 2. 学习率缩放策略
    if args.lr_scaling == "linear":
        # 线性缩放：LR ∝ batch_size
        scaled_lr = args.lr * num_devices
    elif args.lr_scaling == "sqrt":
        # 平方根缩放：LR ∝ √batch_size
        scaled_lr = args.lr * math.sqrt(num_devices)
    else:  # constant
        # 保持不变
        scaled_lr = args.lr

    # 3. 保持单卡视角的有效训练步数
    # 多卡情况下，每步处理更多数据，所以需要调整步数
    effective_steps_per_epoch = args.steps_per_epoch
    total_steps = args.epochs * effective_steps_per_epoch
    warmup_steps = int(args.warmup_epochs * effective_steps_per_epoch)

    print(f"\n=== Training Configuration ===")
    print(f"Multi-GPU scaling:")
    print(f"  Devices: {num_devices}")
    print(f"  Batch size per device: {per_device_batch_size}")
    print(f"  Total batch size: {total_batch_size}")
    print(f"  Single-GPU LR: {args.lr:.2e}")
    print(f"  Scaled LR ({args.lr_scaling}): {scaled_lr:.2e}")
    print(f"  LR scaling factor: {scaled_lr / args.lr:.2f}")
    print(f"\nTraining schedule:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Steps per epoch: {effective_steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Effective data per step: {total_batch_size} samples")

    # 加载数据
    print("\nLoading vault data...")
    vlt = Vault(rel_dir=args.vault_dir, vault_name=args.vault_name, vault_uid=args.vault_uid)
    data = vlt.read()

    # 推断shape信息
    obs_shape = data.experience["observation"].shape
    _, seq_len, n_agents, obs_dim = obs_shape

    action_field = data.experience["action"]
    if action_field.ndim == 4:
        action_dim = action_field.shape[-1]
        discrete_action = False
    else:
        action_dim = int(np.array(action_field).max()) + 1
        discrete_action = True

    print(f"Data info: obs_dim={obs_dim}, action_dim={action_dim}, n_agents={n_agents}")
    print(f"Action type: {'discrete' if discrete_action else 'continuous'}")

    # 创建trajectory buffer - 使用总batch size
    sample_sequence_length = args.K + 1
    buffer = fbx.make_trajectory_buffer(
        sample_batch_size=total_batch_size,  # 使用总batch size
        sample_sequence_length=sample_sequence_length,
        period=1,
        max_length_time_axis=1_000_000,
        min_length_time_axis=sample_sequence_length,
        add_batch_size=1,
    )
    buffer_sample = jax.jit(buffer.sample)

    # 创建模型和训练状态
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    model = S2MPNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        K=args.K,
    )

    # 创建学习率调度 - 使用缩放后的学习率
    learning_rate_fn = create_learning_rate_fn(
        train_ds_size=effective_steps_per_epoch,
        train_batch_size=total_batch_size,
        num_train_epochs=args.epochs,
        num_warmup_steps=warmup_steps,
        learning_rate=scaled_lr,  # 使用缩放后的学习率
    )

    # 初始化训练状态
    state = create_train_state(init_rng, model, learning_rate_fn, args.grad_clip_norm)
    # 若指定检查点路径，则加载参数
    if args.load_checkpoint is not None:
        checkpoint_path = args.load_checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, "rb") as f:
            state = serialization.from_bytes(state, f.read())
        print(f"Loaded checkpoint parameters from {checkpoint_path}")

    # 复制状态到所有设备 - 只在多设备时复制
    if num_devices > 1:
        state = jax.device_put_replicated(state, jax.devices())
        print(f"Replicated training state to {num_devices} devices")
    else:
        print("Using single device training")

    # JIT 编译模型 apply 函数用于测试评估
    apply_fn = jax.jit(model.apply)

    # -------------------------------------------------------------------------
    # 准备固定测试集（高收益episodes）
    # -------------------------------------------------------------------------
    high_reward_episodes = []
    print(
        f"Collecting top {args.num_test_episodes} episodes with cumulative reward > {args.min_reward}..."
    )
    # 转为numpy方便处理
    rewards = np.array(data.experience["reward"])  # (B, T, N)
    dones = np.array(data.experience["done"])  # (B, T, N)
    batch_size_data, seq_len, _ = rewards.shape
    for batch_idx in range(batch_size_data):
        cumulative_reward = 0.0
        episode_start = 0
        for t in range(seq_len):
            cumulative_reward += np.sum(rewards[batch_idx, t, :])
            if np.any(dones[batch_idx, t, :]):
                if cumulative_reward > args.min_reward:
                    high_reward_episodes.append((batch_idx, episode_start, t))
                    print(
                        f"Found high reward episode: batch {batch_idx}, steps {episode_start}-{t}, reward {cumulative_reward:.3f}"
                    )
                    if len(high_reward_episodes) >= args.num_test_episodes:
                        break
                episode_start = t + 1
                cumulative_reward = 0.0
        if len(high_reward_episodes) >= args.num_test_episodes:
            break
    if not high_reward_episodes:
        print(f"No episodes found with cumulative reward > {args.min_reward}")
        fixed_test_data = []
    else:
        print(f"Found {len(high_reward_episodes)} high-reward episodes for testing")

    # 构建K步序列数据 - 为新的评估函数格式化
    fixed_test_data = []
    for batch_idx, start_idx, end_idx in high_reward_episodes:
        for t in range(start_idx + args.K, end_idx + 1):
            seq_start = t - args.K
            seq_end = t
            obs_window = data.experience["observation"][
                batch_idx, seq_start:seq_end, :, :
            ]  # (K, N, obs_dim)
            action_window = data.experience["action"][
                batch_idx, seq_start:seq_end, ...
            ]  # (K, N) or (K, N, action_dim)
            # 转换为 (1, N, K, obs_dim)
            obs_seq_test = jnp.array(obs_window)[None, ...].transpose(0, 2, 1, 3)
            # 转换动作序列
            if discrete_action:
                action_seq_test = jnp.array(action_window).transpose(1, 0)[None, ...]  # (1, N, K)
            else:
                action_seq_test = jnp.array(action_window).transpose(1, 0, 2)[
                    None, ...
                ]  # (1, N, K, action_dim)
            # 构造mask (为兼容性保留，但新的评估函数会重新生成)
            eye = jnp.eye(n_agents)[None, :, :]
            test_masks = jnp.broadcast_to(eye, (1, n_agents, n_agents))
            fixed_test_data.append((obs_seq_test, action_seq_test, test_masks))

    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 训练循环
    print("\nStarting multi-GPU training...")
    print(
        f"Effective training throughput: {total_batch_size * effective_steps_per_epoch * args.epochs:,} samples total"
    )

    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        epoch_pbar = tqdm(range(effective_steps_per_epoch), desc=f"Epoch {epoch + 1}")
        epoch_losses = []

        for step_in_epoch in epoch_pbar:
            global_step += 1

            # 生成多设备的随机数
            rng, sample_key, mask_key = jax.random.split(rng, 3)

            # 采样数据 - 采样总batch size的数据
            sample = buffer_sample(data, sample_key)
            processed = remove_cross_episode_contamination(sample.experience)

            # 准备输入数据
            obs_seq = processed["observation"][:, -args.K :, :, :].transpose(0, 2, 1, 3)
            action_raw = processed["action"][:, -args.K :, :]

            if discrete_action:
                action_seq = action_raw.transpose(0, 2, 1)
            else:
                action_seq = action_raw.transpose(0, 2, 1, 3)

            # 生成agent masks - 使用总batch size
            agent_masks, _ = generate_agent_masks(
                mask_key, total_batch_size, n_agents, args.num_random_masks
            )

            if num_devices > 1:
                # 多设备训练
                device_rngs = jax.random.split(mask_key, num_devices)
                # 准备多设备batch
                batch = prepare_batch_for_devices((obs_seq, action_seq, agent_masks), num_devices)
                # 多设备训练步骤
                state, loss = p_train_step(state, batch, model, device_rngs)
                # 获取第一个设备的损失
                loss_value = float(loss[0])
            else:
                # 单设备训练
                batch = (obs_seq, action_seq, agent_masks)
                state, loss = train_step(state, batch, model, mask_key)
                loss_value = float(loss)

            epoch_losses.append(loss_value)

            # 计算学习率
            current_lr = learning_rate_fn(global_step - 1)

            # 更新进度条 - 显示实际处理的样本数
            samples_processed = global_step * total_batch_size
            epoch_pbar.set_postfix(
                {
                    "loss": f"{loss_value:.6f}",
                    "lr": f"{current_lr:.2e}",
                    "best": f"{best_loss:.6f}",
                    "samples": f"{samples_processed:,}",
                }
            )

            if loss_value < best_loss:
                best_loss = loss_value

            # 详细日志
            if global_step % args.log_interval == 0:
                avg_loss = np.mean(epoch_losses[-args.log_interval :])
                print(
                    f"\nStep {global_step}/{total_steps}, Avg Loss: {avg_loss:.6f}, "
                    f"Current LR: {current_lr:.2e}, Samples: {samples_processed:,}"
                )

            # 保存检查点
            if global_step % args.save_interval == 0:
                # 处理单设备和多设备情况
                if num_devices > 1:
                    # 多设备情况下，提取每个leaf的第一个分片
                    state_to_save = jax.tree.map(lambda x: jax.device_get(x)[0], state)
                else:
                    state_to_save = jax.device_get(state)

                checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{global_step}")
                with open(checkpoint_path, "wb") as f:
                    f.write(serialization.to_bytes(state_to_save))
                print(f"Saved checkpoint to {checkpoint_path}")

        # Epoch结束统计
        epoch_avg_loss = np.mean(epoch_losses)
        epoch_samples = effective_steps_per_epoch * total_batch_size
        print(
            f"Epoch {epoch + 1} completed. Average loss: {epoch_avg_loss:.6f}, "
            f"Samples processed: {epoch_samples:,}"
        )
        # ---------------------------------------------------------------------
        # 本轮测试评估
        # ---------------------------------------------------------------------
        # 提取当前参数
        if num_devices > 1:
            # 第一个设备的参数
            params_for_test = jax.tree.map(lambda x: x[0], state).params
        else:
            params_for_test = state.params

        # 评估所有固定测试样本
        if fixed_test_data:
            # 使用新的评估函数计算总体重建损失
            test_loss = evaluate_s2mp(
                model,
                params_for_test,
                fixed_test_data,
                n_agents,
                "discrete" if discrete_action else "continuous",
            )
            print(f"Epoch {epoch + 1} Test Loss (S2MP reconstruction): {test_loss:.6f}")

            # 额外评估：计算最后时刻的重建误差（兼容原有逻辑）
            test_epoch_mses = []
            test_all_timestep_mses = []  # 新增：所有时刻的MSE

            for obs_seq_test, action_seq_test, test_masks in fixed_test_data:
                # 对每个agent分别进行预测测试
                for agent_idx in range(n_agents):
                    # 创建只保留当前agent的mask
                    agent_mask = jnp.zeros((1, n_agents))
                    agent_mask = agent_mask.at[:, agent_idx].set(1.0)

                    # 预测
                    pred_obs = eval_apply_fn(
                        model, params_for_test, obs_seq_test, action_seq_test, agent_mask
                    )

                    # 计算最后时刻的重建误差（对被mask的agents）
                    mask_for_loss = 1.0 - agent_mask  # 对其他agents计算误差
                    obs_error_last = jnp.square(pred_obs[:, :, -1, :] - obs_seq_test[:, :, -1, :])
                    masked_error_last = obs_error_last * mask_for_loss[:, :, None]

                    if jnp.sum(mask_for_loss) > 0:
                        mse_last = jnp.sum(masked_error_last) / (
                            jnp.sum(mask_for_loss) * obs_seq_test.shape[-1]
                        )
                        test_epoch_mses.append(float(mse_last))

                    # 计算所有时刻的重建误差
                    obs_error_all = jnp.square(pred_obs - obs_seq_test)
                    masked_error_all = obs_error_all * mask_for_loss[:, :, None, None]

                    if jnp.sum(mask_for_loss) > 0:
                        mse_all = jnp.sum(masked_error_all) / (
                            jnp.sum(mask_for_loss) * obs_seq_test.shape[-2] * obs_seq_test.shape[-1]
                        )
                        test_all_timestep_mses.append(float(mse_all))

            # 输出详细的测试结果
            if test_epoch_mses:
                avg_epoch_mse_last = np.mean(test_epoch_mses)
                print(f"Epoch {epoch + 1} Test MSE (last timestep): {avg_epoch_mse_last:.6f}")

            if test_all_timestep_mses:
                avg_all_timestep_mse = np.mean(test_all_timestep_mses)
                print(f"Epoch {epoch + 1} Test MSE (all timesteps): {avg_all_timestep_mse:.6f}")

            # 输出详细统计信息
            if test_epoch_mses and test_all_timestep_mses:
                print(f"Epoch {epoch + 1} Test Statistics:")
                print(
                    f"  Last timestep - Min: {np.min(test_epoch_mses):.6f}, Max: {np.max(test_epoch_mses):.6f}, Std: {np.std(test_epoch_mses):.6f}"
                )
                print(
                    f"  All timesteps - Min: {np.min(test_all_timestep_mses):.6f}, Max: {np.max(test_all_timestep_mses):.6f}, Std: {np.std(test_all_timestep_mses):.6f}"
                )
                print(f"  Total test samples: {len(test_epoch_mses)}")
        else:
            print(f"Epoch {epoch + 1}: No test data available - skipping evaluation")

        # ---------------------------------------------------------------------
        # 绘图验证：为每个高收益episode生成详细对比图像
        # ---------------------------------------------------------------------
        for idx, (batch_idx, start_idx, end_idx) in enumerate(high_reward_episodes):
            output_folder = os.path.join(
                args.test_output_folder, f"epoch_{epoch + 1}", f"episode_{idx}"
            )
            plot_smax_env.plot_episode_with_all_predictions(
                data,
                batch_idx,
                start_idx,
                end_idx,
                output_folder,
                model,
                params_for_test,
                args.K,
            )

    total_samples = total_steps * total_batch_size
    print(f"\nMulti-GPU training completed!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Total samples processed: {total_samples:,}")
    print(f"Training efficiency: {num_devices}x speedup with {num_devices} devices")


if __name__ == "__main__":
    main()
