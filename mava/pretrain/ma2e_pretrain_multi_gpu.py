"""Multi-GPU MA2E Pretraining Script"""

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

from mava.networks.ma2e_network import MA2ENetwork

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
    num_train_steps = num_train_epochs * train_ds_size
    num_warmup_steps = min(num_warmup_steps, num_train_steps - 1)
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
def generate_agent_masks_ma2e(
    rng: chex.PRNGKey, batch_size: int, n_agents: int, masking_strategy: str = "single"
) -> Tuple[chex.Array, chex.PRNGKey]:
    """生成MA2E的agent masks"""
    rng, subkey = jax.random.split(rng)

    if masking_strategy == "single":
        # 每次只mask一个agent (对应原论文的单agent推断)
        agent_indices = jax.random.randint(subkey, (batch_size,), minval=0, maxval=n_agents)
        masks = jnp.ones((batch_size, n_agents))
        masks = masks.at[jnp.arange(batch_size), agent_indices].set(0.0)

    elif masking_strategy == "random":
        # 随机mask 1到n_agents-1个agent
        rng, subkey = jax.random.split(rng)
        n_to_mask = jax.random.randint(subkey, (batch_size,), minval=1, maxval=n_agents)

        def make_single_mask(bs_idx, num_mask, key_inner):
            key_perm = jax.random.split(key_inner)[0]
            perm = jax.random.permutation(key_perm, n_agents)
            indices = jnp.arange(n_agents)
            select_mask = indices < num_mask
            masked_idx = jnp.where(select_mask, perm, n_agents)
            mask = jnp.ones((n_agents,))
            valid_indices = jnp.where(masked_idx < n_agents, masked_idx, 0)
            valid_mask = masked_idx < n_agents
            mask = mask.at[valid_indices].multiply(1.0 - valid_mask.astype(jnp.float32))
            return mask

        keys_inner = jax.random.split(subkey, batch_size)
        masks = jax.vmap(make_single_mask)(jnp.arange(batch_size), n_to_mask, keys_inner)

    elif masking_strategy == "all_combinations":
        # 生成所有可能的单agent mask组合
        eye_masks = jnp.eye(n_agents)
        mask_indices = jax.random.randint(subkey, (batch_size,), minval=0, maxval=n_agents)
        masks = 1.0 - eye_masks[mask_indices]  # 1=keep, 0=mask

    else:
        raise ValueError(f"Unknown masking strategy: {masking_strategy}")

    return masks, rng


# -----------------------------------------------------------------------------
# Multi-GPU Training Functions
# -----------------------------------------------------------------------------


def create_train_state(rng, model, learning_rate_fn, grad_clip_norm=1.0):
    """创建训练状态"""
    dummy_obs = jnp.zeros((1, model.n_agents, model.traj_length, model.obs_dim))
    dummy_action = jnp.zeros((1, model.n_agents, model.traj_length, model.action_dim))
    dummy_mask = jnp.ones((1, model.n_agents))
    params = model.init(rng, dummy_obs, dummy_action, dummy_mask)

    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate_fn, weight_decay=1e-5, b1=0.9, b2=0.95, eps=1e-8),
    )
    return TrainState.create(params=params, tx=tx, apply_fn=model.apply)


def ma2e_loss_fn(
    params, batch, model, rng, action_space_type="discrete", action_dim=10, loss_weights=None
):
    """MA2E损失函数"""
    obs_seq, action_seq, agent_masks = batch
    B, N, K, obs_dim = obs_seq.shape

    # 获取预测
    pred_obs, pred_action = model.apply(
        params, obs_seq, action_seq, agent_masks, deterministic=False
    )

    # 计算重建损失
    obs_error = jnp.square(pred_obs - obs_seq)  # (B, N, K, obs_dim)

    # 对于离散动作，需要特殊处理
    if action_space_type == "discrete":
        # 将动作索引转换为one-hot用于损失计算
        action_indices = action_seq[:, :, :, 0].astype(jnp.int32)  # (B, N, K)
        action_onehot = jax.nn.one_hot(action_indices, action_dim)  # (B, N, K, action_dim)
        action_error = jnp.square(pred_action - action_onehot)  # (B, N, K, action_dim)
    else:
        action_error = jnp.square(pred_action - action_seq)  # (B, N, K, action_dim)

    # 应用mask：只对被mask的agent计算损失
    # agent_masks: (B, N) 1=keep, 0=mask
    mask_for_loss = 1.0 - agent_masks  # 0=keep, 1=mask (计算这些位置的损失)

    # 扩展mask维度以匹配error形状
    obs_mask = mask_for_loss[:, :, None, None]  # (B, N, 1, 1)
    action_mask = mask_for_loss[:, :, None, None]  # (B, N, 1, 1)

    # 时序权重 (可选)
    if loss_weights is not None:
        temporal_weights = loss_weights["temporal"]  # (K,)
        temporal_weights = temporal_weights[None, None, :, None]  # (1, 1, K, 1)
        obs_mask = obs_mask * temporal_weights
        action_mask = action_mask * temporal_weights

    # 加权误差
    weighted_obs_error = obs_error * obs_mask
    weighted_action_error = action_error * action_mask

    # 计算平均损失
    obs_loss = jnp.sum(weighted_obs_error) / jnp.sum(obs_mask)
    action_loss = jnp.sum(weighted_action_error) / jnp.sum(action_mask)

    # 组合损失
    total_loss = obs_loss + action_loss

    return total_loss


def train_step(state, batch, model, rng, action_space_type="discrete", action_dim=10):
    """单步训练 - 将被pmap包装"""
    grad_fn = jax.value_and_grad(ma2e_loss_fn, argnums=0)
    loss, grads = grad_fn(state.params, batch, model, rng, action_space_type, action_dim)

    # 跨设备平均梯度
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")

    state = state.apply_gradients(grads=grads)
    return state, loss


# 使用pmap包装训练步骤
p_train_step = jax.pmap(train_step, axis_name="batch", static_broadcasted_argnums=(2, 4, 5))


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


@partial(jax.jit, static_argnums=(0,))
def eval_apply_fn(model, params, obs_seq, action_seq, agent_mask):
    """专门用于评估的apply函数，deterministic=True是静态的"""
    return model.apply(params, obs_seq, action_seq, agent_mask, deterministic=True)


def evaluate_ma2e(
    model, params, test_data_list, n_agents, action_space_type="discrete", action_dim=10
):
    """评估MA2E模型性能"""
    test_losses = []

    for obs_seq_test, action_seq_test in test_data_list:
        # 为每个agent创建单独的mask进行测试
        batch_size = obs_seq_test.shape[0]

        agent_losses = []
        for agent_idx in range(n_agents):
            # 创建只mask当前agent的mask
            agent_mask = jnp.ones((batch_size, n_agents))
            agent_mask = agent_mask.at[:, agent_idx].set(0.0)  # mask当前agent

            # 预测
            pred_obs, pred_action = eval_apply_fn(
                model, params, obs_seq_test, action_seq_test, agent_mask
            )

            # 计算被mask的agent的重建误差
            obs_error = jnp.mean(
                jnp.square(pred_obs[:, agent_idx, :, :] - obs_seq_test[:, agent_idx, :, :])
            )

            # 对于离散动作，需要特殊处理
            if action_space_type == "discrete":
                action_indices = action_seq_test[:, agent_idx, :, 0].astype(jnp.int32)  # (B, K)
                action_onehot = jax.nn.one_hot(action_indices, action_dim)  # (B, K, action_dim)
                action_error = jnp.mean(jnp.square(pred_action[:, agent_idx, :, :] - action_onehot))
            else:
                action_error = jnp.mean(
                    jnp.square(
                        pred_action[:, agent_idx, :, :] - action_seq_test[:, agent_idx, :, :]
                    )
                )

            total_error = obs_error + action_error
            agent_losses.append(float(total_error))

        test_losses.append(np.mean(agent_losses))

    return np.mean(test_losses)


# -----------------------------------------------------------------------------
# Main Training Script
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU MA2E Pretraining")
    parser.add_argument("--vault_dir", type=str, default="/home/mxfeng/aaai25_projects/Mava/vaults")
    parser.add_argument("--vault_name", type=str, default="rec_mappo")
    parser.add_argument("--vault_uid", type=str, default="20250624093826")

    # 模型参数
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--K", type=int, default=5, help="Trajectory length")
    parser.add_argument("--embed_dim", type=int, default=24, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument(
        "--num_encoder_layers", type=int, default=3, help="Number of encoder layers"
    )
    parser.add_argument(
        "--num_decoder_layers", type=int, default=2, help="Number of decoder layers"
    )
    parser.add_argument("--mlp_dim", type=int, default=96, help="MLP hidden dimension")
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout rate")
    parser.add_argument(
        "--positional_type",
        type=str,
        default="both",
        choices=["agent", "time", "both"],
        help="Positional encoding type",
    )

    # 训练参数
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--grad_clip_norm", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--warmup_epochs", type=float, default=0.0)
    parser.add_argument(
        "--lr_scaling", type=str, default="linear", choices=["linear", "sqrt", "constant"]
    )

    # MA2E特定参数
    parser.add_argument(
        "--masking_strategy",
        type=str,
        default="single",
        choices=["single", "random", "all_combinations"],
        help="Agent masking strategy",
    )

    # 日志和保存
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default="./ma2e_checkpoints")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # 测试参数
    parser.add_argument("--num_test_episodes", type=int, default=3)
    parser.add_argument("--min_reward", type=float, default=1.5)
    parser.add_argument("--test_output_folder", type=str, default="./ma2e_test_plots")

    args = parser.parse_args()

    # 多GPU设置
    num_devices = jax.device_count()
    print(f"Available devices: {num_devices}")
    print(f"Device list: {jax.devices()}")

    # 超参数缩放
    total_batch_size = args.batch_size * num_devices
    per_device_batch_size = args.batch_size

    if args.lr_scaling == "linear":
        scaled_lr = args.lr * num_devices
    elif args.lr_scaling == "sqrt":
        scaled_lr = args.lr * math.sqrt(num_devices)
    else:
        scaled_lr = args.lr

    effective_steps_per_epoch = args.steps_per_epoch
    total_steps = args.epochs * effective_steps_per_epoch
    warmup_steps = int(args.warmup_epochs * effective_steps_per_epoch)

    print(f"\n=== MA2E Training Configuration ===")
    print(f"Multi-GPU scaling:")
    print(f"  Devices: {num_devices}")
    print(f"  Batch size per device: {per_device_batch_size}")
    print(f"  Total batch size: {total_batch_size}")
    print(f"  Scaled LR: {scaled_lr:.2e}")
    print(f"  Masking strategy: {args.masking_strategy}")
    print(f"  Positional encoding: {args.positional_type}")

    # 加载数据
    print("\nLoading vault data...")
    vlt = Vault(rel_dir=args.vault_dir, vault_name=args.vault_name, vault_uid=args.vault_uid)
    data = vlt.read()

    # 推断数据shape
    obs_shape = data.experience["observation"].shape
    _, seq_len, n_agents, obs_dim = obs_shape

    action_field = data.experience["action"]
    if action_field.ndim == 4:
        action_dim = action_field.shape[-1]
        action_space_type = "continuous"
    else:
        action_dim = int(np.array(action_field).max()) + 1
        action_space_type = "discrete"

    print(f"Data info: obs_dim={obs_dim}, action_dim={action_dim}, n_agents={n_agents}")
    print(f"Action type: {action_space_type}")

    # 创建trajectory buffer
    sample_sequence_length = args.K + 1
    buffer = fbx.make_trajectory_buffer(
        sample_batch_size=total_batch_size,
        sample_sequence_length=sample_sequence_length,
        period=1,
        max_length_time_axis=1_000_000,
        min_length_time_axis=sample_sequence_length,
        add_batch_size=1,
    )
    buffer_sample = jax.jit(buffer.sample)

    # 创建MA2E模型
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    model = MA2ENetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        traj_length=args.K,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        mlp_dim=args.mlp_dim,
        dropout_rate=args.dropout_rate,
        positional_type=args.positional_type,
        action_space_type=action_space_type,
    )

    # 创建学习率调度
    learning_rate_fn = create_learning_rate_fn(
        train_ds_size=effective_steps_per_epoch,
        train_batch_size=total_batch_size,
        num_train_epochs=args.epochs,
        num_warmup_steps=warmup_steps,
        learning_rate=scaled_lr,
    )

    # 初始化训练状态
    state = create_train_state(init_rng, model, learning_rate_fn, args.grad_clip_norm)

    # 加载检查点
    if args.load_checkpoint is not None:
        print(f"Loading checkpoint from {args.load_checkpoint}")
        with open(args.load_checkpoint, "rb") as f:
            state = serialization.from_bytes(state, f.read())

    # 复制到多设备
    if num_devices > 1:
        state = jax.device_put_replicated(state, jax.devices())
        print(f"Replicated training state to {num_devices} devices")

    # 准备测试数据
    print(f"Collecting test episodes with reward > {args.min_reward}...")
    test_data_list = []
    rewards = np.array(data.experience["reward"])
    dones = np.array(data.experience["done"])
    batch_size_data, seq_len_data, _ = rewards.shape

    high_reward_episodes = []
    for batch_idx in range(batch_size_data):
        cumulative_reward = 0.0
        episode_start = 0
        for t in range(seq_len_data):
            cumulative_reward += np.sum(rewards[batch_idx, t, :])
            if np.any(dones[batch_idx, t, :]):
                if cumulative_reward > args.min_reward:
                    high_reward_episodes.append((batch_idx, episode_start, t))
                    if len(high_reward_episodes) >= args.num_test_episodes:
                        break
                episode_start = t + 1
                cumulative_reward = 0.0
        if len(high_reward_episodes) >= args.num_test_episodes:
            break

    # 构建测试数据
    for batch_idx, start_idx, end_idx in high_reward_episodes:
        for t in range(start_idx + args.K, end_idx + 1):
            seq_start = t - args.K
            seq_end = t
            obs_window = data.experience["observation"][batch_idx, seq_start:seq_end, :, :]
            action_window = data.experience["action"][batch_idx, seq_start:seq_end, ...]

            obs_seq_test = jnp.array(obs_window)[None, ...].transpose(0, 2, 1, 3)

            if action_space_type == "discrete":
                action_seq_test = jnp.array(action_window).transpose(1, 0)[None, ...]
                # 为离散动作添加维度
                action_seq_test = action_seq_test[..., None]
            else:
                action_seq_test = jnp.array(action_window).transpose(1, 0, 2)[None, ...]

            test_data_list.append((obs_seq_test, action_seq_test))

    # 创建检查点目录和测试输出目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.test_output_folder, exist_ok=True)

    # 训练循环
    print("\nStarting MA2E multi-GPU training...")

    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        epoch_pbar = tqdm(range(effective_steps_per_epoch), desc=f"Epoch {epoch + 1}")
        epoch_losses = []

        for step_in_epoch in epoch_pbar:
            global_step += 1

            # 生成随机数
            rng, sample_key, mask_key = jax.random.split(rng, 3)

            # 采样数据
            sample = buffer_sample(data, sample_key)
            processed = remove_cross_episode_contamination(sample.experience)

            # 准备输入数据
            obs_seq = processed["observation"][:, -args.K :, :, :].transpose(0, 2, 1, 3)
            action_raw = processed["action"][:, -args.K :, :]

            if action_space_type == "discrete":
                action_seq = action_raw.transpose(0, 2, 1)
                # 为离散动作添加维度以匹配网络期望
                action_seq = action_seq[..., None]
            else:
                action_seq = action_raw.transpose(0, 2, 1, 3)

            # 生成agent masks
            agent_masks, _ = generate_agent_masks_ma2e(
                mask_key, total_batch_size, n_agents, args.masking_strategy
            )

            if num_devices > 1:
                # 多设备训练
                device_rngs = jax.random.split(mask_key, num_devices)
                batch = prepare_batch_for_devices((obs_seq, action_seq, agent_masks), num_devices)
                state, loss = p_train_step(
                    state, batch, model, device_rngs, action_space_type, action_dim
                )
                loss_value = float(loss[0])
            else:
                # 单设备训练
                batch = (obs_seq, action_seq, agent_masks)
                state, loss = train_step(
                    state, batch, model, mask_key, action_space_type, action_dim
                )
                loss_value = float(loss)

            epoch_losses.append(loss_value)
            current_lr = learning_rate_fn(global_step - 1)

            epoch_pbar.set_postfix(
                {
                    "loss": f"{loss_value:.6f}",
                    "lr": f"{current_lr:.2e}",
                    "best": f"{best_loss:.6f}",
                }
            )

            if loss_value < best_loss:
                best_loss = loss_value

            # 详细日志
            if global_step % args.log_interval == 0:
                avg_loss = np.mean(epoch_losses[-args.log_interval :])
                print(f"\nStep {global_step}/{total_steps}, Avg Loss: {avg_loss:.6f}")

            # 保存检查点
            if global_step % args.save_interval == 0:
                if num_devices > 1:
                    state_to_save = jax.tree.map(lambda x: jax.device_get(x)[0], state)
                else:
                    state_to_save = jax.device_get(state)

                checkpoint_path = os.path.join(
                    args.checkpoint_dir, f"ma2e_checkpoint_{global_step}"
                )
                with open(checkpoint_path, "wb") as f:
                    f.write(serialization.to_bytes(state_to_save))
                print(f"Saved checkpoint to {checkpoint_path}")

        # ---------------------------------------------------------------------
        # 本轮测试评估
        # ---------------------------------------------------------------------
        if num_devices > 1:
            params_for_test = jax.tree.map(lambda x: x[0], state).params
        else:
            params_for_test = state.params

        # 评估所有固定测试样本
        if test_data_list:
            test_loss = evaluate_ma2e(
                model, params_for_test, test_data_list, n_agents, action_space_type, action_dim
            )
            print(f"Epoch {epoch + 1} Test Loss (MA2E reconstruction): {test_loss:.6f}")

            # 额外评估：计算最后时刻的重建误差（类似S2MP）
            test_epoch_mses = []
            for obs_seq_test, action_seq_test in test_data_list:
                # 对每个agent分别mask进行预测
                for agent_idx in range(n_agents):
                    # 创建只mask当前agent的mask
                    agent_mask = jnp.ones((1, n_agents))
                    agent_mask = agent_mask.at[:, agent_idx].set(0.0)

                    # 预测
                    pred_obs, pred_action = eval_apply_fn(
                        model, params_for_test, obs_seq_test, action_seq_test, agent_mask
                    )

                    # 计算最后时刻的重建误差
                    obs_error_last = jnp.mean(
                        jnp.square(
                            pred_obs[:, agent_idx, -1, :] - obs_seq_test[:, agent_idx, -1, :]
                        )
                    )
                    test_epoch_mses.append(float(obs_error_last))

            avg_epoch_mse_last = np.mean(test_epoch_mses)
            print(f"Epoch {epoch + 1} Test MSE (last timestep): {avg_epoch_mse_last:.6f}")

        # ---------------------------------------------------------------------
        # 绘图验证：为每个高收益episode生成详细对比图像
        # ---------------------------------------------------------------------
        print(f"Generating visualization plots for epoch {epoch + 1}...")
        for idx, (batch_idx, start_idx, end_idx) in enumerate(high_reward_episodes):
            output_folder = os.path.join(
                args.test_output_folder, f"epoch_{epoch + 1}", f"episode_{idx}"
            )
            try:
                plot_smax_env.plot_ma2e_episode_predictions(
                    data,
                    batch_idx,
                    start_idx,
                    end_idx,
                    output_folder,
                    model,
                    params_for_test,
                    args.K,
                    action_space_type,
                    action_dim,
                )
                print(f"Saved plots for episode {idx} to {output_folder}")
            except Exception as e:
                print(f"Warning: Failed to generate plots for episode {idx}: {e}")

        epoch_avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1} completed. Average loss: {epoch_avg_loss:.6f}")

    print(f"\nMA2E training completed! Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
