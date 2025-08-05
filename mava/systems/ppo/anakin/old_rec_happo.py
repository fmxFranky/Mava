# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import time
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import chex
import flashbax as fbx
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flashbax.vault import Vault
from flax.core.frozen_dict import FrozenDict
from gymnasium.spaces import Discrete, MultiDiscrete
from jax import tree
from jumanji.specs import DiscreteArray, MultiDiscreteArray  # Local import to avoid circular deps
from omegaconf import DictConfig, OmegaConf
from optax._src.base import OptState
from tqdm import tqdm

from mava.evaluator import (
    get_eval_fn_with_traj,
    get_num_eval_envs,
    make_rec_eval_act_fn_with_traj,
)
from mava.networks import RecurrentActor as Actor
from mava.networks import RecurrentValueNet as Critic
from mava.networks.base import ScannedRNN, ScannedRNNPerAgent
from mava.systems.ppo.types import (
    HiddenStates,
    OptStates,
    Params,
    RNNLearnerState,
    RNNPPOTransition,
    TrajectoryState,
)
from mava.types import (
    ExperimentOutput,
    JointTrajectory,
    MarlEnv,
    RecActorApply,
    RecCriticApply,
)
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.multistep import calculate_gae
from mava.utils.network_utils import get_action_head
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics

# Define the learner function type that returns experience data
StoreExpLearnerFn = Callable[
    [RNNLearnerState], Tuple[ExperimentOutput[RNNLearnerState], RNNPPOTransition]
]


def remove_cross_episode_contamination(experience):
    """Remove cross-episode contamination from experience data."""
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


# 性能监控装饰器
def profile_time(func_name: str):
    """性能监控装饰器"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            # 只在第一次调用时显示，避免训练过程中的过多输出
            if not hasattr(wrapper, "_first_call_done"):
                print(f"🚀 JIT加速 - {func_name}: {(end_time - start_time) * 1000:.2f}ms")
                wrapper._first_call_done = True
            return result

        return wrapper

    return decorator


# 标准化掩码策略
@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def generate_ratio_mask(
    rng: chex.PRNGKey, batch_size: int, n_agents: int, augment_size: int, mask_ratio: float
) -> chex.Array:
    """策略1：根据预定义比例随机mask，内部进行增广

    根据mask_ratio确定要mask的智能体数量（N*mask_ratio个），
    然后随机sample出augment_size种不同的方式来选择这么多个智能体进行mask

    Args:
        rng: 随机种子
        batch_size: 批次大小
        n_agents: 智能体数量
        augment_size: 增广大小（生成augment_size种不同的mask方式）
        mask_ratio: 掩码比例

    Returns:
        shape: (augment_size * batch_size, n_agents)
    """
    # 根据mask_ratio确定要mask的智能体数量
    num_to_mask = int(n_agents * mask_ratio)
    num_to_mask = max(1, min(num_to_mask, n_agents - 1))  # 至少保留一个agent

    def make_mask_for_aug_idx(aug_idx, key):
        # 为当前aug_idx生成batch_size个mask，都使用相同的mask方式
        def make_single_mask(single_key):
            perm = jax.random.permutation(single_key, n_agents)
            mask = jnp.ones(n_agents)
            # 使用JAX友好的方式进行mask，避免动态切片
            indices = jnp.arange(n_agents)
            should_mask = indices < num_to_mask
            mask_indices = jnp.where(should_mask, perm, n_agents)  # 无效索引设为n_agents
            valid_mask = mask_indices < n_agents
            # 只对有效索引进行mask操作
            mask = mask.at[mask_indices].multiply(1.0 - valid_mask.astype(jnp.float32))
            return mask

        single_keys = jax.random.split(key, batch_size)
        return jax.vmap(make_single_mask)(single_keys)

    # 为每个aug_idx生成对应的masks（augment_size种不同的mask方式）
    aug_keys = jax.random.split(rng, augment_size)
    aug_indices = jnp.arange(augment_size)

    all_masks = jax.vmap(make_mask_for_aug_idx)(aug_indices, aug_keys)
    # all_masks shape: (augment_size, batch_size, n_agents)

    # 重新排列为 (augment_size * batch_size, n_agents)
    masks = all_masks.reshape(-1, n_agents)
    return masks


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def generate_random_agents_mask(
    rng: chex.PRNGKey, batch_size: int, n_agents: int, augment_size: int, mask_ratio: float
) -> chex.Array:
    """策略2：随机选择不同数量的智能体进行mask，内部进行增广

    在[1,2,...,N-1]中sample出augment_size种不同的数量，
    每种数量对应一种mask方式

    Args:
        rng: 随机种子
        batch_size: 批次大小
        n_agents: 智能体数量
        augment_size: 增广大小（sample出augment_size种不同的mask数量）
        mask_ratio: 掩码比例（此策略不使用）

    Returns:
        shape: (augment_size * batch_size, n_agents)
    """
    # 可mask的智能体数量范围：[1, n_agents-1]
    max_maskable = n_agents - 1
    available_nums = jnp.arange(1, n_agents)  # [1, 2, ..., N-1]

    # 从[1, 2, ..., N-1]中随机sample出augment_size个不同的数量
    if augment_size >= max_maskable:
        # 如果augment_size >= 可用数量，直接使用所有可用数量，然后重复
        sampled_indices = jnp.arange(max_maskable) % max_maskable
        mask_nums = available_nums[sampled_indices[:augment_size]]
    else:
        # 随机sample出augment_size个不重复的数量
        perm = jax.random.permutation(rng, max_maskable)
        sampled_indices = perm[:augment_size]
        mask_nums = available_nums[sampled_indices]

    def make_mask_for_aug_idx(aug_idx, key):
        # 获取当前aug_idx对应的mask数量
        num_to_mask = mask_nums[aug_idx]

        # 为当前aug_idx生成batch_size个mask
        def make_single_mask(single_key):
            perm = jax.random.permutation(single_key, n_agents)
            mask = jnp.ones(n_agents)
            # 使用JAX友好的方式进行mask，避免动态切片
            indices = jnp.arange(n_agents)
            should_mask = indices < num_to_mask
            mask_indices = jnp.where(should_mask, perm, n_agents)  # 无效索引设为n_agents
            valid_mask = mask_indices < n_agents
            # 只对有效索引进行mask操作
            mask = mask.at[mask_indices].multiply(1.0 - valid_mask.astype(jnp.float32))
            return mask

        single_keys = jax.random.split(key, batch_size)
        return jax.vmap(make_single_mask)(single_keys)

    # 为每个aug_idx生成对应的masks
    aug_keys = jax.random.split(rng, augment_size)
    aug_indices = jnp.arange(augment_size)

    all_masks = jax.vmap(make_mask_for_aug_idx)(aug_indices, aug_keys)
    # all_masks shape: (augment_size, batch_size, n_agents)

    # 重新排列为 (augment_size * batch_size, n_agents)
    masks = all_masks.reshape(-1, n_agents)
    return masks


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def generate_single_agent_mask(
    rng: chex.PRNGKey, batch_size: int, n_agents: int, augment_size: int, mask_ratio: float
) -> chex.Array:
    """策略3：仅保留单智能体，mask掉其他所有智能体，内部进行增广

    基于不同的10个key生成10个不同的从0到N-1的permutation，
    然后取前augment_size个作为对应的要保留的single agent

    Args:
        rng: 随机种子
        batch_size: 批次大小
        n_agents: 智能体数量
        augment_size: 增广大小（取前augment_size个permutation对应的智能体）
        mask_ratio: 掩码比例（此策略不使用）

    Returns:
        shape: (augment_size * batch_size, n_agents)
    """
    # 生成10个不同的key，用于产生10个不同的permutation
    perm_keys = jax.random.split(rng, 2)

    # 生成10个不同的从0到N-1的permutation
    def make_permutation(key):
        return jax.random.permutation(key, n_agents)

    all_permutations = jax.vmap(make_permutation)(perm_keys)
    # all_permutations shape: (10, n_agents)

    # 取前augment_size个permutation的第0个元素作为要保留的智能体
    selected_agents = all_permutations[:augment_size, 0]  # shape: (augment_size,)

    def make_mask_for_aug_idx(aug_idx):
        # 获取当前aug_idx对应的要保留的智能体
        selected_agent = selected_agents[aug_idx]

        # 为当前aug_idx生成batch_size个相同的mask（所有样本保留同一个智能体）
        # 使用JAX友好的方式避免动态索引
        agent_indices = jnp.arange(n_agents)
        agent_mask = (agent_indices == selected_agent).astype(jnp.float32)  # (n_agents,)
        masks = jnp.broadcast_to(
            agent_mask[None, :], (batch_size, n_agents)
        )  # (batch_size, n_agents)
        return masks

    # 为每个aug_idx生成对应的masks
    aug_indices = jnp.arange(augment_size)
    all_masks = jax.vmap(make_mask_for_aug_idx)(aug_indices)
    # all_masks shape: (augment_size, batch_size, n_agents)

    # 重新排列为 (augment_size * batch_size, n_agents)
    masks = all_masks.reshape(-1, n_agents)
    return masks


def build_s2mp_training_dataset(data, buffer, num_samples, action_space_type, action_dim, traj_len):
    """构建S2MP训练数据集，使用改进的数据处理流程"""
    print(f"构建S2MP训练数据集，目标样本数: {num_samples}")
    training_data = []

    # 使用JIT加速采样
    buffer_sample = jax.jit(buffer.sample)

    for i in range(num_samples):
        rng_key = jax.random.PRNGKey(i)
        sample = buffer_sample(data, rng_key)
        processed = remove_cross_episode_contamination(sample.experience)

        # 提取观测和动作序列
        obs_seq = processed["observation"]  # (1, T, N, obs_dim)
        action_seq = processed["action"]  # (1, T, N) 或 (1, T, N, action_dim)

        # 处理动作序列：转换为4维并归一化
        if action_space_type == "discrete":
            # 离散动作：转换为4维 [B, T, N, 1] 并归一化
            if action_seq.ndim == 3:  # (B, T, N)
                action_seq = action_seq[..., None]  # (B, T, N, 1)
            action_seq = action_seq.astype(jnp.float32) / action_dim  # 归一化
        else:
            # 连续动作：确保是4维
            if action_seq.ndim == 3:  # (B, T, N) -> (B, T, N, 1)
                action_seq = action_seq[..., None]

        # 提取长度为traj_len的窗口
        T = obs_seq.shape[1]
        if T >= traj_len:
            # 提取随机窗口
            start_idx = jax.random.randint(rng_key, (), 0, T - traj_len + 1)
            obs_window = obs_seq[0, start_idx : start_idx + traj_len]  # (traj_len, N, obs_dim)
            action_window = action_seq[
                0, start_idx : start_idx + traj_len
            ]  # (traj_len, N, action_dim)

            training_data.append((obs_window, action_window))

    print(f"S2MP训练数据集构建完成，实际样本数: {len(training_data)}")
    return training_data


def pretrain_s2mp_single_actor(
    single_actor_params: Params,
    actor_network: Actor,
    obs_dim: int,
    action_dim: int,
    n_agents: int,
    traj_len: int,
    action_space_type: str = "discrete",
    pretrain_epochs: int = 5,
    learning_rate: float = 1e-3,
    vault_dir: Optional[str] = None,
    vault_name: Optional[str] = None,
    vault_uid: Optional[str] = None,
    num_training_samples: int = 1000,
    batch_size: int = 64,
    ratio_augment_size: int = 6,
    random_augment_size: int = 5,
    single_augment_size: int = 5,
    mask_ratio: float = 0.75,
    grad_clip_norm: float = 1.0,
) -> Params:
    """
    使用统一的掩码-重构预训练框架对S2MP actor进行预训练

    基于 unified_mask_reconstruction_pretrain.py 的改进实现，包含渐进式ratio策略：
    - 支持三种标准化掩码策略，统一接口设计
    - JIT编译优化
    - 改进的数据处理流程
    - 性能监控
    - 支持多epoch训练和进度条显示
    - 90%训练数据，10%验证数据
    - 真正的批处理 + 内部数据增广策略
    - 每种掩码策略独立指定增广倍数，函数内部完成增广，三种策略合并后一次训练
    - ratio策略: ratio_augment_size倍增广，动态mask数量从1/N线性增加到(N-1)/N，生成augment_size种不同选择方式
    - random策略: random_augment_size倍增广，从[1,2,...,N-1]中sample出augment_size种不同mask数量
    - single策略: single_augment_size倍增广，10个permutation取前augment_size个对应的保留智能体
    - 统一掩码接口: 所有策略使用相同的函数签名，内部返回(augment_size*batch_size, n_agents)大小的mask
    - 损失统计: 分别计算各策略真实损失，反映真实的策略表现
    """
    # 定义掩码策略和对应的增广倍数
    strategy_configs = [
        ("ratio", generate_ratio_mask, ratio_augment_size),
        ("random", generate_random_agents_mask, random_augment_size),
        ("single", generate_single_agent_mask, single_augment_size),
    ]
    mask_strategies = [(name, fn) for name, fn, _ in strategy_configs]

    # 计算总的增广倍数
    total_augment_size = ratio_augment_size + random_augment_size + single_augment_size

    print(f"开始S2MP预训练，epochs: {pretrain_epochs}, 数据样本: {num_training_samples}")
    print(f"🚀 已启用JIT编译加速优化")
    print(f"📊 批处理设置: batch_size={batch_size}")
    print(f"📈 策略增广配置 (内部增广):")
    min_ratio = 1.0 / n_agents
    max_ratio = (n_agents - 1.0) / n_agents
    print(
        f"  - ratio策略: {ratio_augment_size}倍增广 (动态mask从{min_ratio:.3f}到{max_ratio:.3f}，{ratio_augment_size}种选择方式)"
    )
    print(
        f"  - random策略: {random_augment_size}倍增广 (从[1,{n_agents - 1}]中sample出{random_augment_size}种不同mask数量)"
    )
    print(
        f"  - single策略: {single_augment_size}倍增广 (10个permutation取前{single_augment_size}个对应的保留智能体)"
    )
    print(f"📋 每次训练的有效batch大小: {batch_size * total_augment_size}")
    print(f"🔄 掩码策略: {len(mask_strategies)}种 (ratio, random, single)")
    print(
        f"🔧 统一接口: 所有掩码函数使用相同参数，内部完成增广，返回(augment_size*batch_size, n_agents)大小的mask"
    )

    # 创建优化器
    s2mp_optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(learning_rate, eps=1e-5),
    )

    # 初始化优化器状态
    s2mp_opt_state = s2mp_optimizer.init(single_actor_params)

    # 从vault加载训练数据
    training_data = []
    if vault_dir and vault_name and vault_uid:
        try:
            print(f"从vault加载训练数据: {vault_dir}/{vault_name}/{vault_uid}")

            # 初始化vault和buffer
            vault = Vault(
                vault_name=vault_name,
                vault_uid=vault_uid,
                rel_dir=vault_dir,
            )

            # 加载vault数据
            data = vault.read()

            # 创建trajectory buffer用于采样
            sample_sequence_length = traj_len + 1
            buffer = fbx.make_trajectory_buffer(
                sample_batch_size=1,
                sample_sequence_length=sample_sequence_length,
                period=1,
                max_length_time_axis=1_000_000,
                min_length_time_axis=sample_sequence_length,
                add_batch_size=1,
            )

            # 构建训练数据集
            training_data = build_s2mp_training_dataset(
                data=data,
                buffer=buffer,
                num_samples=num_training_samples,
                action_space_type=action_space_type,
                action_dim=action_dim,
                traj_len=traj_len,
            )

            print(f"成功加载 {len(training_data)} 个训练样本")

        except Exception as e:
            print(f"Vault数据加载失败: {e}")
            print("回退到随机数据生成...")
            training_data = []

    # 定义损失函数
    def s2mp_loss_fn(
        params: Params, obs_seq: jnp.ndarray, action_seq: jnp.ndarray, agent_mask: jnp.ndarray
    ) -> jnp.ndarray:
        """S2MP重构损失函数 - 改进版本"""
        # 使用get_predictions方法调用s2mp_torso
        pred_obs = actor_network.apply(
            params, obs_seq, action_seq, agent_mask, method=actor_network.get_predictions
        )

        # 计算重构损失 - 只对被mask的agents计算损失
        obs_error = jnp.square(pred_obs - obs_seq)
        mask_for_loss = 1.0 - agent_mask  # 0=keep, 1=mask (B, N)
        obs_mask = mask_for_loss[:, :, None, None]  # (B, N, 1, 1) -> 广播到 (B, N, K, obs_dim)

        weighted_obs_error = obs_error * obs_mask
        mse_loss = jnp.mean(weighted_obs_error)
        return mse_loss

    # JIT编译训练步骤
    @profile_time("S2MP训练步骤")
    @jax.jit
    def update_step(params, opt_state, obs_seq, action_seq, agent_mask):
        loss, grads = jax.value_and_grad(s2mp_loss_fn)(params, obs_seq, action_seq, agent_mask)
        updates, new_opt_state = s2mp_optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # 预训练循环
    current_params = single_actor_params
    current_opt_state = s2mp_opt_state
    rng = jax.random.PRNGKey(42)

    if not training_data:
        raise ValueError("必须提供vault数据进行预训练，不支持随机数据生成")

    # 数据集分割：90%训练，10%验证
    total_samples = len(training_data)
    train_size = int(total_samples * 0.9)
    val_size = total_samples - train_size

    # 打乱数据并分割
    rng, shuffle_key = jax.random.split(rng)
    indices = jax.random.permutation(shuffle_key, total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_data = [training_data[int(i)] for i in train_indices]
    val_data = [training_data[int(i)] for i in val_indices]

    print(f"数据集分割完成: 训练样本 {len(train_data)}, 验证样本 {len(val_data)}")

    # 验证函数
    @jax.jit
    def validate_step(params, obs_seq, action_seq, agent_mask):
        loss = s2mp_loss_fn(params, obs_seq, action_seq, agent_mask)
        return loss

    # 多epoch训练循环
    total_training_steps = 0
    steps_per_epoch = len(train_data) // batch_size
    val_steps_per_epoch = len(val_data) // batch_size

    for epoch in range(pretrain_epochs):
        # ratio策略的动态mask_ratio：从1/N线性增加到(N-1)/N
        if pretrain_epochs == 1:
            current_mask_ratio = mask_ratio  # 只有一个epoch时使用原始值
        else:
            # 线性插值：epoch=0时为1/N，epoch=pretrain_epochs-1时为(N-1)/N
            min_ratio = 1.0 / n_agents
            max_ratio = (n_agents - 1.0) / n_agents
            progress = epoch / (pretrain_epochs - 1)  # 0.0 到 1.0
            current_mask_ratio = min_ratio + progress * (max_ratio - min_ratio)

        print(
            f"\nEpoch {epoch + 1}/{pretrain_epochs} (ratio策略动态mask_ratio: {current_mask_ratio:.3f})"
        )
        epoch_train_losses = {"ratio": [], "random": [], "single": []}

        # 创建进度条
        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch + 1}",
            miniters=max(1, steps_per_epoch // 50),  # 最多50次更新
            mininterval=0.5,  # 最少0.5秒更新一次
        )

        for step in pbar:
            # 构建真正的batch（包含batch_size个不同样本）
            batch_obs_list = []
            batch_actions_list = []

            for batch_idx in range(batch_size):
                data_idx = (step * batch_size + batch_idx) % len(train_data)
                obs_window, action_window = train_data[data_idx]

                # 转换格式: (T, N, obs_dim) -> (N, T, obs_dim)
                obs_formatted = obs_window.transpose(1, 0, 2)

                # 处理动作序列格式
                if action_space_type == "discrete" and action_window.shape[-1] == 1:
                    # S2MP需要3维离散动作: (T, N, 1) -> (N, T)
                    action_formatted = (
                        (action_window[..., 0] * action_dim).astype(jnp.int32).transpose(1, 0)
                    )
                else:
                    # 连续动作: (T, N, action_dim) -> (N, T, action_dim)
                    action_formatted = action_window.transpose(1, 0, 2)

                batch_obs_list.append(obs_formatted)
                batch_actions_list.append(action_formatted)

            # 堆叠成真正的batch: (batch_size, N, T, obs_dim)
            batch_obs = jnp.stack(batch_obs_list, axis=0)
            batch_actions = jnp.stack(batch_actions_list, axis=0)

            # 对每种掩码策略进行独立数据增广，然后合并
            all_obs_batches = []
            all_action_batches = []
            all_masks = []

            for strategy_name, mask_fn, aug_size in strategy_configs:
                rng, mask_key = jax.random.split(rng)

                # 使用统一的接口生成mask (返回 aug_size * batch_size 个mask)
                # ratio策略使用动态mask_ratio，其他策略使用原始值
                mask_ratio_to_use = current_mask_ratio if strategy_name == "ratio" else mask_ratio
                agent_masks = mask_fn(mask_key, batch_size, n_agents, aug_size, mask_ratio_to_use)

                # 扩展obs和action到对应的增广大小
                expanded_obs = jnp.repeat(
                    batch_obs, aug_size, axis=0
                )  # (aug_size * batch_size, N, T, obs_dim)
                expanded_actions = jnp.repeat(
                    batch_actions, aug_size, axis=0
                )  # (aug_size * batch_size, N, T, action_dim)

                # 添加到增广列表
                all_obs_batches.append(expanded_obs)
                all_action_batches.append(expanded_actions)
                all_masks.append(agent_masks)

            # 合并所有增广batch: (batch_size * total_augment_size, N, T, obs_dim)
            final_obs_batch = jnp.concatenate(all_obs_batches, axis=0)
            final_action_batch = jnp.concatenate(all_action_batches, axis=0)
            final_mask_batch = jnp.concatenate(all_masks, axis=0)

            # 分别计算每种策略的损失，然后进行一次合并训练更新
            strategy_losses = {}
            strategy_batch_start = 0

            # 分别计算每种策略的损失（用于统计）
            for strategy_name, _, aug_size in strategy_configs:
                strategy_batch_end = strategy_batch_start + (batch_size * aug_size)
                strategy_obs = final_obs_batch[strategy_batch_start:strategy_batch_end]
                strategy_actions = final_action_batch[strategy_batch_start:strategy_batch_end]
                strategy_mask = final_mask_batch[strategy_batch_start:strategy_batch_end]

                # 计算该策略的损失（仅用于统计，不用于反向传播）
                strategy_loss = s2mp_loss_fn(
                    current_params, strategy_obs, strategy_actions, strategy_mask
                )
                strategy_losses[strategy_name] = float(strategy_loss)
                epoch_train_losses[strategy_name].append(float(strategy_loss))

                strategy_batch_start = strategy_batch_end

            # 对整个合并batch进行一次训练更新（用于实际的参数更新）
            current_params, current_opt_state, total_loss = update_step(
                current_params,
                current_opt_state,
                final_obs_batch,
                final_action_batch,
                final_mask_batch,
            )

            total_training_steps += 1

            # 更新进度条
            if step % max(1, steps_per_epoch // 50) == 0:
                final_batch_size = final_obs_batch.shape[0]
                effective_samples = batch_size * total_augment_size

                # 计算各策略的累计平均损失
                strategy_avg_losses = {
                    mask_name: jnp.mean(jnp.array(losses)) if losses else 0.0
                    for mask_name, losses in epoch_train_losses.items()
                }

                pbar.set_postfix(
                    {
                        "loss": f"{float(total_loss):.6f}",
                        "eff_batch": f"{effective_samples}",
                        "ratio": f"{strategy_losses.get('ratio', 0.0):.6f}",
                        "random": f"{strategy_losses.get('random', 0.0):.6f}",
                        "single": f"{strategy_losses.get('single', 0.0):.6f}",
                    }
                )

        # Epoch结束后显示训练统计
        epoch_train_avg_losses = {
            mask_name: jnp.mean(jnp.array(losses)) if losses else 0.0
            for mask_name, losses in epoch_train_losses.items()
        }
        epoch_actual_steps = steps_per_epoch
        epoch_effective_samples = steps_per_epoch * batch_size * total_augment_size
        print(
            f"Epoch {epoch + 1} 训练完成 - 批处理步数: {epoch_actual_steps}, 有效样本数: {epoch_effective_samples}"
        )
        print(
            f"  ratio策略本epoch使用mask_ratio: {current_mask_ratio:.3f} (mask {int(n_agents * current_mask_ratio)}个智能体)"
        )
        print(f"Epoch {epoch + 1} 训练平均损失 (各策略真实损失):")
        for strategy_name, _, aug_size in strategy_configs:
            avg_loss = epoch_train_avg_losses[strategy_name]
            print(f"  {strategy_name}: {avg_loss:.6f} (增广: {aug_size}x)")

        # 验证阶段
        print(f"Epoch {epoch + 1} 验证中...")
        val_losses = {"ratio": [], "random": [], "single": []}

        val_pbar = tqdm(
            range(val_steps_per_epoch),
            desc=f"Validation",
            miniters=max(1, val_steps_per_epoch // 10),  # 验证进度条更新频率更低
            mininterval=1.0,
        )

        for val_step in val_pbar:
            # 构建真正的验证batch
            val_batch_obs_list = []
            val_batch_actions_list = []

            for batch_idx in range(batch_size):
                val_data_idx = (val_step * batch_size + batch_idx) % len(val_data)
                obs_window, action_window = val_data[val_data_idx]

                # 转换格式: (T, N, obs_dim) -> (N, T, obs_dim)
                obs_formatted = obs_window.transpose(1, 0, 2)

                if action_space_type == "discrete" and action_window.shape[-1] == 1:
                    # S2MP需要3维离散动作: (T, N, 1) -> (N, T)
                    action_formatted = (
                        (action_window[..., 0] * action_dim).astype(jnp.int32).transpose(1, 0)
                    )
                else:
                    # 连续动作: (T, N, action_dim) -> (N, T, action_dim)
                    action_formatted = action_window.transpose(1, 0, 2)

                val_batch_obs_list.append(obs_formatted)
                val_batch_actions_list.append(action_formatted)

            # 堆叠成真正的验证batch
            val_batch_obs = jnp.stack(val_batch_obs_list, axis=0)
            val_batch_actions = jnp.stack(val_batch_actions_list, axis=0)

            # 对每种掩码策略进行独立数据增广，然后合并
            val_all_obs_batches = []
            val_all_action_batches = []
            val_all_masks = []

            for strategy_name, mask_fn, aug_size in strategy_configs:
                rng, mask_key = jax.random.split(rng)

                # 使用统一的接口生成mask (返回 aug_size * batch_size 个mask)
                # ratio策略使用动态mask_ratio，其他策略使用原始值
                mask_ratio_to_use = current_mask_ratio if strategy_name == "ratio" else mask_ratio
                agent_masks = mask_fn(mask_key, batch_size, n_agents, aug_size, mask_ratio_to_use)

                # 扩展obs和action到对应的增广大小
                val_expanded_obs = jnp.repeat(val_batch_obs, aug_size, axis=0)
                val_expanded_actions = jnp.repeat(val_batch_actions, aug_size, axis=0)

                val_all_obs_batches.append(val_expanded_obs)
                val_all_action_batches.append(val_expanded_actions)
                val_all_masks.append(agent_masks)

            # 合并所有验证增广batch
            val_final_obs_batch = jnp.concatenate(val_all_obs_batches, axis=0)
            val_final_action_batch = jnp.concatenate(val_all_action_batches, axis=0)
            val_final_mask_batch = jnp.concatenate(val_all_masks, axis=0)

            # 分别计算每种策略的验证损失
            val_strategy_losses = {}
            val_strategy_batch_start = 0

            # 分别计算每种策略的验证损失（用于统计）
            for strategy_name, _, aug_size in strategy_configs:
                val_strategy_batch_end = val_strategy_batch_start + (batch_size * aug_size)
                val_strategy_obs = val_final_obs_batch[
                    val_strategy_batch_start:val_strategy_batch_end
                ]
                val_strategy_actions = val_final_action_batch[
                    val_strategy_batch_start:val_strategy_batch_end
                ]
                val_strategy_mask = val_final_mask_batch[
                    val_strategy_batch_start:val_strategy_batch_end
                ]

                # 计算该策略的验证损失
                val_strategy_loss = validate_step(
                    current_params, val_strategy_obs, val_strategy_actions, val_strategy_mask
                )
                val_strategy_losses[strategy_name] = float(val_strategy_loss)
                val_losses[strategy_name].append(float(val_strategy_loss))

                val_strategy_batch_start = val_strategy_batch_end

            # 计算总验证损失（仅用于显示）
            total_val_loss = validate_step(
                current_params, val_final_obs_batch, val_final_action_batch, val_final_mask_batch
            )

            # 更新验证进度条
            if val_step % max(1, val_steps_per_epoch // 10) == 0:
                val_effective_samples = batch_size * total_augment_size

                # 计算各策略的累计平均验证损失
                val_strategy_avg_losses = {
                    mask_name: jnp.mean(jnp.array(losses)) if losses else 0.0
                    for mask_name, losses in val_losses.items()
                }
                overall_val_avg = jnp.mean(jnp.array(list(val_strategy_avg_losses.values())))

                val_pbar.set_postfix(
                    {
                        "loss": f"{float(total_val_loss):.6f}",
                        "eff_batch": f"{val_effective_samples}",
                        "ratio": f"{val_strategy_losses.get('ratio', 0.0):.6f}",
                        "random": f"{val_strategy_losses.get('random', 0.0):.6f}",
                        "single": f"{val_strategy_losses.get('single', 0.0):.6f}",
                    }
                )

        # 显示验证结果
        epoch_val_avg_losses = {
            mask_name: jnp.mean(jnp.array(losses)) if losses else 0.0
            for mask_name, losses in val_losses.items()
        }
        val_epoch_actual_steps = val_steps_per_epoch
        val_epoch_effective_samples = val_steps_per_epoch * batch_size * total_augment_size
        print(
            f"Epoch {epoch + 1} 验证完成 - 批处理步数: {val_epoch_actual_steps}, 有效样本数: {val_epoch_effective_samples}"
        )
        print(
            f"  ratio策略本epoch使用mask_ratio: {current_mask_ratio:.3f} (mask {int(n_agents * current_mask_ratio)}个智能体)"
        )
        print(f"Epoch {epoch + 1} 验证平均损失 (各策略真实损失):")
        for strategy_name, _, aug_size in strategy_configs:
            avg_loss = epoch_val_avg_losses[strategy_name]
            print(f"  {strategy_name}: {avg_loss:.6f} (增广: {aug_size}x)")

        overall_val_loss = jnp.mean(jnp.array(list(epoch_val_avg_losses.values())))
        print(f"Epoch {epoch + 1} 总体验证损失: {overall_val_loss:.6f}")

    # 显示最终统计
    final_total_steps = total_training_steps
    total_train_effective_samples = (
        pretrain_epochs * steps_per_epoch * batch_size * total_augment_size
    )
    total_val_effective_samples = (
        pretrain_epochs * val_steps_per_epoch * batch_size * total_augment_size
    )
    print(f"\n🎉 S2MP预训练完成！")
    print(f"📈 总统计:")
    print(f"  - 训练epochs: {pretrain_epochs}")
    print(f"  - 原始训练样本总数: {len(train_data)}")
    print(f"  - 原始验证样本总数: {len(val_data)}")
    print(f"  - 批处理大小: {batch_size}")
    print(
        f"  - 策略增广配置: ratio({ratio_augment_size}x), random({random_augment_size}x), single({single_augment_size}x)"
    )
    print(f"  - 总增广倍数: {total_augment_size}")
    print(f"  - 掩码策略数: {len(mask_strategies)}")
    print(f"  - 总训练批处理步数: {final_total_steps}")
    print(f"  - 总有效训练样本数: {total_train_effective_samples}")
    print(f"  - 总有效验证样本数: {total_val_effective_samples}")
    print(f"  - 有效batch放大倍数: {total_augment_size}x")
    print(f"  📊 损失计算说明: 分别计算各策略真实损失，反映真实策略表现")
    print(f"  🔧 增广机制详情:")
    final_min_ratio = 1.0 / n_agents
    final_max_ratio = (n_agents - 1.0) / n_agents
    print(
        f"    - ratio: 动态mask从{final_min_ratio:.3f}到{final_max_ratio:.3f}，{ratio_augment_size}种选择方式"
    )
    print(f"    - random: 从[1-{n_agents - 1}]中sample {random_augment_size}种mask数量")
    print(f"    - single: 10个permutation取前{single_augment_size}个保留智能体")
    return current_params


def construct_joint_trajectory_for_timestep(
    traj_batch: RNNPPOTransition, timestep_idx: int, config: DictConfig
) -> JointTrajectory:
    """Construct joint trajectory for a specific timestep with proper history.

    This function constructs the trajectory history for timestep_idx,
    where the history includes [timestep_idx-traj_len+1, ..., timestep_idx].
    """
    traj_len = config.system.get("traj_len", 5)
    T, B, N = traj_batch.obs.agents_view.shape[:3]
    obs_shape = traj_batch.obs.agents_view.shape[3:]

    # For timestep_idx, we want history [timestep_idx-traj_len+1, ..., timestep_idx]
    start_idx = max(0, timestep_idx - traj_len + 1)
    end_idx = timestep_idx + 1

    # Extract the available history
    obs_slice = traj_batch.obs.agents_view[start_idx:end_idx]  # [actual_len, B, N, *obs_dim]
    action_slice = traj_batch.action[start_idx:end_idx]  # [actual_len, B, N] or [actual_len, B]

    actual_len = obs_slice.shape[0]

    # Handle action dimensions - expand if needed
    if action_slice.ndim == 2:  # [actual_len, B] -> expand to [actual_len, B, N]
        action_slice = jnp.broadcast_to(action_slice[:, :, jnp.newaxis], (actual_len, B, N))

    # Pad if necessary (when timestep_idx < traj_len-1)
    if actual_len < traj_len:
        pad_len = traj_len - actual_len
        obs_pad = jnp.zeros((pad_len, B, N, *obs_shape))
        action_pad = jnp.zeros((pad_len, B, N), dtype=action_slice.dtype)

        obs_hist = jnp.concatenate([obs_pad, obs_slice], axis=0)  # [traj_len, B, N, *obs_dim]
        action_hist = jnp.concatenate([action_pad, action_slice], axis=0)  # [traj_len, B, N]
    else:
        obs_hist = obs_slice[-traj_len:]  # Take last traj_len steps
        action_hist = action_slice[-traj_len:]

    # Transpose to [B, N, traj_len, *dims] format expected by networks
    obs_hist = obs_hist.transpose(1, 2, 0, *range(3, obs_hist.ndim))  # [B, N, traj_len, *obs_dim]
    action_hist = action_hist.transpose(1, 2, 0)  # [B, N, traj_len]

    return JointTrajectory(
        observations=obs_hist,  # [B, N, traj_len, *obs_dim]
        actions=action_hist,  # [B, N, traj_len]
        last_actions=None,  # Not available during env interaction
        last_action_masks=None,  # Would need additional parameters to provide
    )


def construct_joint_trajectory_for_agents(
    trajectory_state: TrajectoryState, config: DictConfig
) -> JointTrajectory:
    """Construct joint trajectory for all agents from trajectory state.

    Args:
        trajectory_state: Current trajectory state with obs_history [B, N, traj_len, *obs_dim]
                         and action_history [B, N, traj_len, *action_dim]
        config: Configuration containing traj_len

    Returns:
        JointTrajectory with shape [N, 1, B, N, traj_len, *dims] for vmapped agent processing
        Each agent sees the same complete multi-agent trajectory data
    """
    traj_len = config.system.get("traj_len", 10)
    B, N = trajectory_state.obs_history.shape[:2]

    # trajectory_state.obs_history: [B, N, traj_len, *obs_dim]
    # trajectory_state.action_history: [B, N, traj_len, *action_dim]

    # For HAPPO, each agent should see the same complete joint trajectory
    # Add batch dimension and repeat for each agent
    # [B, N, traj_len, *obs_dim] -> [1, B, N, traj_len, *obs_dim] -> [N, 1, B, N, traj_len, *obs_dim]
    joint_obs = trajectory_state.obs_history[jnp.newaxis, ...]  # [1, B, N, traj_len, *obs_dim]
    joint_obs = jnp.repeat(joint_obs, N, axis=0)  # [N, 1, B, N, traj_len, *obs_dim]

    joint_actions = trajectory_state.action_history[
        jnp.newaxis, ...
    ]  # [1, B, N, traj_len, *action_dim]
    joint_actions = jnp.repeat(joint_actions, N, axis=0)  # [N, 1, B, N, traj_len, *action_dim]

    return JointTrajectory(
        observations=joint_obs,  # [N, 1, B, N, traj_len, *obs_dim]
        actions=joint_actions,  # [N, 1, B, N, traj_len, *action_dim]
        last_actions=None,  # Not available during env interaction
        last_action_masks=None,  # Not available during env interaction - only current agent's mask available
    )


def construct_joint_trajectory_window_slide(
    traj_batch: RNNPPOTransition, config: DictConfig
) -> JointTrajectory:
    """Construct joint trajectory using window slide for minibatch training.

    For each timestep t in the rollout, constructs trajectory history of length traj_len
    ending at timestep t: [t-traj_len+1, ..., t]

    Args:
        traj_batch: Trajectory batch with shape [T, B, N, *dims]
        config: Configuration containing traj_len

    Returns:
        JointTrajectory with shape [T, B, N, traj_len, *dims]
    """
    traj_len = config.system.get("traj_len", 10)
    T, B, N = traj_batch.obs.agents_view.shape[:3]
    obs_shape = traj_batch.obs.agents_view.shape[3:]

    # Handle action dimensions - expand if needed
    action_slice = traj_batch.action
    if action_slice.ndim == 2:  # [T, B] -> expand to [T, B, N]
        action_slice = jnp.broadcast_to(action_slice[:, :, jnp.newaxis], (T, B, N))

    # Efficiently create window slides using JAX operations
    # Create indices for each window: for timestep t, indices are [max(0, t-traj_len+1), ..., t]
    timesteps = jnp.arange(T)  # [0, 1, 2, ..., T-1]

    # For each timestep t, create window indices [t-traj_len+1, ..., t]
    # Shape: [T, traj_len]
    window_offsets = jnp.arange(traj_len)[jnp.newaxis, :]  # [1, traj_len]: [0, 1, ..., traj_len-1]
    window_indices = timesteps[:, jnp.newaxis] - traj_len + 1 + window_offsets  # [T, traj_len]

    # Clip indices to valid range [0, T-1] and handle padding
    window_indices = jnp.clip(window_indices, 0, T - 1)

    # Create mask for valid indices (those that are >= current_timestep - traj_len + 1)
    valid_mask = window_indices >= (timesteps[:, jnp.newaxis] - traj_len + 1)

    # ------------------------------------------------------------------
    # Zero-out cross-episode history: we only want transitions from the
    # current episode.  Compute for every timestep how many steps have
    # elapsed since the last done (per env, per agent). If window offset
    # exceeds this number, we mark it invalid.
    # ------------------------------------------------------------------
    done_flags = traj_batch.done  # shape [T, B, N]

    def _scan_fn(carry, d):
        # carry is steps since last done
        new_carry = jnp.where(d, 1, carry + 1)
        return new_carry, new_carry

    init_steps = jnp.zeros((B, N), dtype=jnp.int32)
    episode_steps, _ = jax.lax.scan(_scan_fn, init_steps, done_flags)
    # episode_steps[t] = 1 at first step after reset, then 2,3,...

    # Expand to [T, traj_len, B, N] for comparison with window offsets
    ep_steps_exp = episode_steps[:, jnp.newaxis, :, :]  # [T,1,B,N]
    k_offsets = window_offsets[:, :, jnp.newaxis, jnp.newaxis]  # [T, traj_len,1,1]
    cross_ep_mask = k_offsets < ep_steps_exp  # True where within episode

    # Combine masks
    valid_mask = valid_mask & cross_ep_mask

    # Extract windows using advanced indexing
    obs_windows = traj_batch.obs.agents_view[window_indices]  # [T, traj_len, B, N, *obs_dim]
    action_windows = action_slice[window_indices]  # [T, traj_len, B, N]

    # Apply padding mask - set invalid positions to zero
    obs_pad_value = jnp.zeros((B, N) + obs_shape)
    action_pad_value = jnp.zeros((B, N), dtype=action_slice.dtype)

    # Create mask shapes that are compatible with observations and actions
    obs_mask_shape = (valid_mask.shape[0], valid_mask.shape[1]) + (1, 1) + (1,) * len(obs_shape)
    action_mask_shape = (valid_mask.shape[0], valid_mask.shape[1]) + (1, 1)

    obs_mask = valid_mask.reshape(obs_mask_shape)
    action_mask = valid_mask.reshape(action_mask_shape)

    obs_windows = jnp.where(obs_mask, obs_windows, obs_pad_value)
    action_windows = jnp.where(action_mask, action_windows, action_pad_value)

    # Transpose to [T, B, N, traj_len, *dims] format
    # obs_windows: [T, traj_len, B, N, *obs_dim] -> [T, B, N, traj_len, *obs_dim]
    perm_obs = (0, 2, 3, 1) + tuple(range(4, obs_windows.ndim))
    obs_windows = obs_windows.transpose(perm_obs)
    # action_windows: [T, traj_len, B, N] -> [T, B, N, traj_len]
    action_windows = action_windows.transpose(0, 2, 3, 1)

    return JointTrajectory(
        observations=obs_windows,  # [T, B, N, traj_len, *obs_dim]
        actions=action_windows,  # [T, B, N, traj_len]
        last_actions=None,  # Not available during env interaction
        last_action_masks=None,  # Would need additional parameters to provide
    )


def construct_dummy_joint_trajectory(
    obs_shape: Tuple,
    action_shape: Tuple,
    batch_size: int,
    num_agents: int,
    traj_len: int,
    action_dtype: jnp.dtype = jnp.int32,
) -> JointTrajectory:
    """Construct dummy joint trajectory for initialization and evaluation.

    Args:
        obs_shape: Shape of observation excluding batch and agent dimensions
        action_shape: Shape of action excluding batch and agent dimensions
        batch_size: Batch size
        num_agents: Number of agents
        traj_len: Trajectory length
        action_dtype: Data type for actions

    Returns:
        Dummy JointTrajectory with appropriate shapes
    """
    dummy_obs = jnp.zeros((batch_size, num_agents, traj_len, *obs_shape))
    dummy_actions = jnp.zeros((batch_size, num_agents, traj_len, *action_shape), dtype=action_dtype)

    return JointTrajectory(
        observations=dummy_obs,
        actions=dummy_actions,
        last_actions=None,  # Dummy trajectory
        last_action_masks=None,  # Dummy trajectory
    )


def construct_training_joint_trajectory(
    traj_batch: RNNPPOTransition, env_num_agents: int, config: DictConfig
) -> JointTrajectory:
    """Construct joint trajectory for training using each timestep's own observation.

    This creates a more realistic joint trajectory where each timestep uses its actual
    observation repeated across the trajectory length, rather than using dummy data.
    """
    obs_shape_full = traj_batch.obs.agents_view.shape
    T = obs_shape_full[0]  # recurrent_chunk_size
    B = obs_shape_full[1]  # minibatch_size

    # Check if we have agent dimension
    if len(obs_shape_full) >= 3 and obs_shape_full[2] == env_num_agents:
        # Case: [T, B, N, *obs_dim] - we have per-agent observations
        N = obs_shape_full[2]
        obs_shape = obs_shape_full[3:]
        obs_data = traj_batch.obs.agents_view  # [T, B, N, *obs_dim]

    else:
        # Case: [T, B, *obs_dim] - single agent case, expand to multi-agent
        N = env_num_agents
        obs_shape = obs_shape_full[2:]
        obs_data = jnp.broadcast_to(
            traj_batch.obs.agents_view[:, :, jnp.newaxis, :], (T, B, N, *obs_shape)
        )

    traj_len = config.system.get("traj_len", 10)

    # For each timestep, repeat its observation across traj_len
    # IMPORTANT: Keep agent-specific observations distinct
    joint_obs = jnp.broadcast_to(
        obs_data[:, :, :, jnp.newaxis, ...],  # [T, B, N, 1, *obs_dim]
        (T, B, N, traj_len, *obs_shape),
    )

    # Handle actions similarly
    action_shape_full = traj_batch.action.shape
    action_data = traj_batch.action

    if len(action_shape_full) == 2:  # [T, B] discrete actions
        action_data = jnp.broadcast_to(
            action_data[:, :, jnp.newaxis],  # [T, B, 1] -> [T, B, N]
            (T, B, N),
        )
        joint_actions = jnp.broadcast_to(
            action_data[:, :, :, jnp.newaxis],  # [T, B, N, 1]
            (T, B, N, traj_len),
        )
    elif len(action_shape_full) == 3 and action_shape_full[2] == env_num_agents:
        # [T, B, N] multi-agent discrete actions
        joint_actions = jnp.broadcast_to(
            action_data[:, :, :, jnp.newaxis],  # [T, B, N, 1]
            (T, B, N, traj_len),
        )
    else:  # [T, B, N, *action_dim] multi-agent continuous actions
        joint_actions = jnp.broadcast_to(
            action_data[:, :, :, jnp.newaxis, ...],  # [T, B, N, 1, *action_dim]
            (T, B, N, traj_len, *action_data.shape[3:]),
        )

    # Extract last actions from the last timestep [T-1, B, N, *action_dim] -> [B, N, *action_dim]
    last_actions = action_data[-1]  # [B, N, *action_dim]
    
    # Extract last action masks from the last timestep observation
    # traj_batch.obs should have action_mask field with shape [T, B, N, *action_mask_dim]
    last_action_masks = None
    if hasattr(traj_batch.obs, 'action_mask') and traj_batch.obs.action_mask is not None:
        last_action_masks = traj_batch.obs.action_mask[-1]  # [B, N, *action_mask_dim]

    return JointTrajectory(
        observations=joint_obs,  # [T, B, N, traj_len, *obs_dim]
        actions=joint_actions,  # [T, B, N, traj_len] or [T, B, N, traj_len, *action_dim]
        last_actions=last_actions,  # [B, N, *action_dim]
        last_action_masks=last_action_masks,  # [B, N, *action_mask_dim] or None
    )


def get_learner_fn(
    env: MarlEnv,
    apply_fns: Tuple[Tuple[RecActorApply, RecActorApply], RecCriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> StoreExpLearnerFn:
    """Get the learner function."""

    (actor_train_apply_fn, actor_exec_apply_fn), critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(learner_state: RNNLearnerState, _: Any) -> Tuple[RNNLearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The current model parameters.
                - opt_states (OptStates): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
                - last_done (bool): Whether the last timestep was a terminal state.
                - hstates (HiddenStates): The current hidden states of the RNN.
            _ (Any): The current metrics info.

        """

        def _env_step(
            learner_state: RNNLearnerState, _: Any
        ) -> Tuple[RNNLearnerState, RNNPPOTransition]:
            """Step the environment."""
            (
                params,
                opt_states,
                key,
                env_state,
                last_timestep,
                last_done,
                last_hstates,
                trajectory_state,
            ) = learner_state

            # Allocate action tensor based on discrete vs continuous action spaces
            is_discrete = isinstance(
                env.action_spec,
                (DiscreteArray, MultiDiscreteArray, Discrete, MultiDiscrete),
            )
            if is_discrete:
                # Discrete action: single integer per agent
                actions = jnp.zeros((config.arch.num_envs, env.num_agents), dtype=jnp.int32)
            else:
                # Continuous action: vector per agent
                actions = jnp.zeros(
                    (config.arch.num_envs, env.num_agents, env.action_dim), dtype=jnp.float32
                )

            log_probs = jnp.zeros((config.arch.num_envs, env.num_agents))
            policy_hidden_states = jnp.zeros_like(
                last_hstates.policy_hidden_state, dtype=jnp.float32
            )

            # Add a batch dimension to the observation.
            batched_observation = tree.map(lambda x: x[jnp.newaxis, :], last_timestep.observation)
            ac_in = (
                batched_observation,
                last_done[jnp.newaxis, :],
            )

            # Update trajectory state first with current observation to ensure consistency
            # Shift history and add current observation
            updated_obs_history = jnp.roll(trajectory_state.obs_history, -1, axis=2)

            updated_obs_history = updated_obs_history.at[:, :, -1].set(
                last_timestep.observation.agents_view
            )

            # For actions, we'll use the previous action history as we don't have current action yet
            # This will be updated after network computation
            updated_action_history = trajectory_state.action_history

            # Create updated trajectory state with current observation
            updated_trajectory_state = TrajectoryState(
                obs_history=updated_obs_history,
                action_history=updated_action_history,
                buffer_idx=trajectory_state.buffer_idx,
                buffer_full=trajectory_state.buffer_full,
            )

            # Construct joint trajectory from updated trajectory state for all agents
            joint_traj_all_agents = construct_joint_trajectory_for_agents(
                updated_trajectory_state, config
            )

            # ---------------- Vectorised per-agent actor forward pass ----------------
            # Split RNG for each agent once
            key, policy_key = jax.random.split(key)
            agent_keys = jax.random.split(policy_key, env.num_agents)

            # Prepare per-agent observations and done flags:  (N, 1, B, ...)
            obs_agents = tree.map(
                lambda x: jnp.transpose(x, (2, 0, 1) + tuple(range(3, x.ndim))),  # (N, 1, B, ...)
                batched_observation,
            )
            done_expand = last_done[jnp.newaxis, ...]  # (1, B, N)
            done_agents = jnp.transpose(done_expand, (2, 0, 1))  # (N, 1, B)

            # Vectorised apply + sampling with per-agent joint trajectory
            def _per_agent_apply(p, h, o, d, k, jt):
                new_h, pi = actor_exec_apply_fn(p, [h], (o, d), jt, key=k)
                act = pi.sample(seed=k)
                lp = pi.log_prob(act)
                # squeeze out the leading time dimension (0) we added (size=1)
                return new_h, act.squeeze(0), lp.squeeze(0)

            vmapped_apply = jax.vmap(_per_agent_apply, in_axes=(0, 1, 0, 0, 0, 0))

            # Run vmapped apply
            new_h_states, agent_actions, agent_log_probs = vmapped_apply(
                params.actor_params,
                last_hstates.policy_hidden_state,  # [B, N, H]  -> vmap agent axis=1
                obs_agents,
                done_agents,
                agent_keys,
                joint_traj_all_agents,  # [N, 1, B, N, traj_len, *dims]
            )

            # Reshape back to (B, N, ...)
            policy_hidden_states = new_h_states.transpose(1, 0, 2)  # (B, N, H)
            actions = agent_actions.transpose(1, 0, *range(2, agent_actions.ndim))  # (B, N, ...)
            log_probs = agent_log_probs.transpose(1, 0)  # (B, N)

            # We can keep the critic as is.
            # Construct joint trajectory for critic (single trajectory, not per-agent)
            critic_joint_traj = JointTrajectory(
                observations=trajectory_state.obs_history[
                    jnp.newaxis, ...
                ],  # [1, B, N, traj_len, *obs_dim]
                actions=trajectory_state.action_history[
                    jnp.newaxis, ...
                ],  # [1, B, N, traj_len, *action_dim]
                last_actions=None,  # Not available during env interaction
                last_action_masks=None,  # Not available during env interaction - only current agent's mask available
            )
            critic_hidden_state, value = critic_apply_fn(
                params.critic_params, last_hstates.critic_hidden_state, ac_in, critic_joint_traj
            )

            value = value.squeeze(0)

            # Step the environment.
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, actions)

            # Update trajectory state with new observation and action
            # Shift history and add new data (this time include the actions we just computed)
            new_action_history = jnp.roll(updated_trajectory_state.action_history, -1, axis=2)
            new_action_history = new_action_history.at[:, :, -1].set(actions)

            # The observation history is already updated with current observation
            new_obs_history = updated_trajectory_state.obs_history

            # Update buffer index (capped at traj_len)
            traj_len = getattr(config.system, "traj_len", 10)
            new_buffer_idx = jnp.minimum(trajectory_state.buffer_idx + 1, traj_len)
            new_buffer_full = trajectory_state.buffer_full | (new_buffer_idx == traj_len)

            new_trajectory_state = TrajectoryState(
                obs_history=new_obs_history,
                action_history=new_action_history,
                buffer_idx=new_buffer_idx,
                buffer_full=new_buffer_full,
            )

            # log episode return and length
            # Duplicate info over agents since we need to be able to slice per agent in the
            # traj_batch later on in the trainer.
            done = timestep.last().repeat(env.num_agents).reshape(config.arch.num_envs, -1)

            hstates = HiddenStates(policy_hidden_states, critic_hidden_state)
            transition = RNNPPOTransition(
                last_done,
                actions,
                value,
                timestep.reward,
                log_probs,
                last_timestep.observation,
                last_hstates,
            )
            learner_state = RNNLearnerState(
                params, opt_states, key, env_state, timestep, done, hstates, new_trajectory_state
            )
            return learner_state, transition

        # INITIALISE RNN STATE
        initial_hstates = learner_state.hstates

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # CALCULATE ADVANTAGE
        (
            params,
            opt_states,
            key,
            env_state,
            last_timestep,
            last_done,
            hstates,
            trajectory_state,
        ) = learner_state

        # Add a batch dimension to the observation.
        batched_last_observation = tree.map(lambda x: x[jnp.newaxis, :], last_timestep.observation)
        ac_in = (
            batched_last_observation,
            last_done[jnp.newaxis, :],
        )

        # Construct joint trajectory for last value computation with proper per-timestep history
        # Use the current trajectory_state which contains the latest history
        joint_traj_last = JointTrajectory(
            observations=trajectory_state.obs_history[
                jnp.newaxis, ...
            ],  # [1, B, N, traj_len, *obs_dim]
            actions=trajectory_state.action_history[
                jnp.newaxis, ...
            ],  # [1, B, N, traj_len, *action_dim]
            last_actions=None,  # Not available during env interaction
            last_action_masks=None,  # Not available during env interaction - only current agent's mask available
        )

        # Run the network.
        _, last_val = critic_apply_fn(
            params.critic_params, hstates.critic_hidden_state, ac_in, joint_traj_last
        )
        # Squeeze out the batch dimension and mask out the value of terminal states.
        last_val = last_val.squeeze(0)
        last_val = jnp.where(last_done, jnp.zeros_like(last_val), last_val)

        advantages, targets = calculate_gae(
            traj_batch, last_val, last_done, config.system.gamma, config.system.gae_lambda
        )

        

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""

                # UNPACK TRAIN STATE AND BATCH INFO
                params, opt_states, key = train_state
                traj_batch, advantages, targets = batch_info

                # Before defining actor_grad_fn, precompute full joint trajectory with all agents.
                full_joint_traj = construct_training_joint_trajectory(
                    traj_batch, env.num_agents, config
                )

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    actor_opt_state: OptState,
                    traj_batch: RNNPPOTransition,
                    gae: chex.Array,
                    key: chex.PRNGKey,
                ) -> Tuple:
                    """Calculate the actor loss using full multi-agent joint trajectory."""
                    # RERUN NETWORK with single-agent obs/done as before
                    obs_and_done = (traj_batch.obs, traj_batch.done)

                    # Use precomputed full_joint_traj (contains all agents' data)
                    _, actor_policy = actor_train_apply_fn(
                        actor_params,
                        [traj_batch.hstates.policy_hidden_state[0]],
                        obs_and_done,
                        full_joint_traj,
                    )
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config.system.clip_eps,
                            1.0 + config.system.clip_eps,
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    # The seed will be used in the TanhTransformedDistribution:
                    entropy = actor_policy.entropy(seed=key).mean()

                    total_loss = loss_actor - config.system.ent_coef * entropy
                    return total_loss, (loss_actor, entropy)

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    critic_opt_state: OptState,
                    traj_batch: RNNPPOTransition,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # RERUN NETWORK
                    obs_and_done = (traj_batch.obs, traj_batch.done)

                    # Construct joint trajectory for minibatch training using actual timestep data
                    joint_traj = construct_training_joint_trajectory(
                        traj_batch, env.num_agents, config
                    )
                    _, value = critic_apply_fn(
                        critic_params,
                        traj_batch.hstates.critic_hidden_state,
                        obs_and_done,
                        joint_traj,  # Joint trajectory with proper shape for minibatch
                    )

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config.system.clip_eps, config.system.clip_eps
                    )
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    total_loss = config.system.vf_coef * value_loss
                    return total_loss, (value_loss)

                # CALCULATE ACTOR LOSS
                # we assume advantages are the same over all agents.
                advantages_single_agent = advantages[:, :, 0]
                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                agents_params = params.actor_params
                agent_opt_states = opt_states.actor_opt_state

                key, shuffle_key = jax.random.split(key, 2)
                shuffled_agents = jax.random.permutation(shuffle_key, env.num_agents)

                def _agent_update(carry, agent_idx):
                    """Update actor params/opt_state for one agent."""
                    agents_params, agent_opt_states, adv_sa, inner_key = carry

                    inner_key, entropy_key = jax.random.split(inner_key)

                    agent_params = tree.map(lambda x: x[agent_idx], agents_params)
                    agent_traj = tree.map(lambda x: x[:, :, agent_idx], traj_batch)
                    agent_opt_state = tree.map(lambda x: x[agent_idx], agent_opt_states)

                    # Actor loss & gradients
                    (actor_loss_info_per_agent, actor_grads_per_agent) = actor_grad_fn(
                        agent_params,
                        agent_opt_state,
                        agent_traj,
                        adv_sa,
                        entropy_key,
                    )

                    actor_grads_per_agent, _ = jax.lax.pmean(
                        (actor_grads_per_agent, actor_loss_info_per_agent), axis_name="batch"
                    )
                    actor_grads_per_agent, _ = jax.lax.pmean(
                        (actor_grads_per_agent, actor_loss_info_per_agent), axis_name="device"
                    )

                    # Apply gradients
                    actor_updates, actor_new_opt_state = actor_update_fn(
                        actor_grads_per_agent, agent_opt_state
                    )
                    actor_new_params = optax.apply_updates(agent_params, actor_updates)

                    # Write back into full structures
                    agents_params = tree.map(
                        lambda x, y: x.at[agent_idx].set(y), agents_params, actor_new_params
                    )
                    agent_opt_states = tree.map(
                        lambda x, y: x.at[agent_idx].set(y), agent_opt_states, actor_new_opt_state
                    )

                    # Update advantage via importance sampling
                    agent_obs_and_done = (agent_traj.obs, agent_traj.done)

                    # For agent update, we have two choices:
                    # 1. Use the FULL joint trajectory with ALL agent observations (HAPPO style)
                    # 2. Use per-agent trajectory that matches the agent_obs_and_done

                    # Let's use approach 1: Full joint trajectory for proper HAPPO
                    # Use the shared full_joint_traj; remove redundant construction.
                    agent_joint_traj = full_joint_traj

                    _, actor_policy = actor_train_apply_fn(
                        actor_new_params,
                        [agent_traj.hstates.policy_hidden_state[0]],
                        agent_obs_and_done,  # Single agent obs/done for this specific agent
                        agent_joint_traj,  # Full multi-agent joint trajectory
                    )
                    log_prob = actor_policy.log_prob(agent_traj.action)
                    ratio = jnp.exp(log_prob - agent_traj.log_prob)
                    adv_sa *= ratio

                    new_carry = (agents_params, agent_opt_states, adv_sa, inner_key)
                    return new_carry, None

                # Scan over agents in shuffled order
                init_carry = (agents_params, agent_opt_states, advantages_single_agent, key)
                (agents_params, agent_opt_states, advantages_single_agent, key), _ = jax.lax.scan(
                    _agent_update, init_carry, shuffled_agents
                )

                # CALCULATE CRITIC LOSS
                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                critic_loss_info, critic_grads = critic_grad_fn(
                    params.critic_params, opt_states.critic_opt_state, traj_batch, targets
                )

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                # This pmean could be a regular mean as the batch axis is on the same device.

                critic_grads, critic_loss_info = jax.lax.pmean(
                    (critic_grads, critic_loss_info), axis_name="batch"
                )
                # pmean over devices.
                critic_grads, critic_loss_info = jax.lax.pmean(
                    (critic_grads, critic_loss_info), axis_name="device"
                )

                # UPDATE CRITIC PARAMS AND OPTIMISER STATE
                critic_updates, critic_new_opt_state = critic_update_fn(
                    critic_grads, opt_states.critic_opt_state
                )
                critic_new_params = optax.apply_updates(params.critic_params, critic_updates)

                new_params = Params(agents_params, critic_new_params)
                new_opt_state = OptStates(agent_opt_states, critic_new_opt_state)

                # PACK LOSS INFO
                return (new_params, new_opt_state, key), {"total_loss": critic_loss_info[1]}

            params, opt_states, init_hstates, traj_batch, advantages, targets, key = update_state
            key, shuffle_key, entropy_key = jax.random.split(key, 3)

            # SHUFFLE MINIBATCHES
            batch = (traj_batch, advantages, targets)
            num_recurrent_chunks = (
                config.system.rollout_length // config.system.recurrent_chunk_size
            )
            batch = tree.map(
                lambda x: x.reshape(
                    config.system.recurrent_chunk_size,
                    config.arch.num_envs * num_recurrent_chunks,
                    *x.shape[2:],
                ),
                batch,
            )
            permutation = jax.random.permutation(
                shuffle_key, config.arch.num_envs * num_recurrent_chunks
            )
            shuffled_batch = tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
            reshaped_batch = tree.map(
                lambda x: jnp.reshape(
                    x, (x.shape[0], config.system.num_minibatches, -1, *x.shape[2:])
                ),
                shuffled_batch,
            )

            minibatches = tree.map(lambda x: jnp.swapaxes(x, 1, 0), reshaped_batch)


            # UPDATE MINIBATCHES
            (params, opt_states, entropy_key), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states, entropy_key), minibatches
            )

            update_state = (
                params,
                opt_states,
                init_hstates,
                traj_batch,
                advantages,
                targets,
                key,
            )
            return update_state, loss_info

        init_hstates = tree.map(lambda x: x[None, :], initial_hstates)
        update_state = (
            params,
            opt_states,
            init_hstates,
            traj_batch,
            advantages,
            targets,
            key,
        )

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, _, traj_batch, advantages, targets, key = update_state
        learner_state = RNNLearnerState(
            params,
            opt_states,
            key,
            env_state,
            last_timestep,
            last_done,
            hstates,
            trajectory_state,
        )
        metric = last_timestep.extras["episode_metrics"] | last_timestep.extras["env_metrics"]
        return learner_state, (metric, loss_info, traj_batch)

    def learner_fn(
        learner_state: RNNLearnerState,
    ) -> Tuple[ExperimentOutput[RNNLearnerState], RNNPPOTransition]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The initial model parameters.
                - opt_states (OptStates): The initial optimizer states.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
                - dones (bool): Whether the initial timestep was a terminal state.
                - hstates (HiddenStates): The initial hidden states of the RNN.

        """

        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info, traj_batch) = jax.lax.scan(
            batched_update_step, learner_state, None, config.system.num_updates_per_eval
        )
        return (
            ExperimentOutput(
                learner_state=learner_state,
                episode_metrics=episode_info,
                train_metrics=loss_info,
            ),
            traj_batch,
        )

    return learner_fn


def learner_setup(
    env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple[StoreExpLearnerFn, Actor, RNNLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Determine number of devices: default to 1 unless override via config
    n_devices = min(config.arch.get("n_devices", 1), len(jax.devices()))

    # Get number of agents.
    num_agents = env.num_agents
    config.system.num_agents = num_agents

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys
    actor_net_keys = jax.random.split(actor_net_key, num_agents)

    # Define network and optimisers.
    if config.system.use_ma2e_fusion:
        config.network.actor_network.pre_torso.layer_sizes = [
            ls // 2 for ls in config.network.actor_network.pre_torso.layer_sizes
        ]
        actor_pre_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    else:
        actor_pre_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_post_torso = hydra.utils.instantiate(config.network.actor_network.post_torso)
    action_head, _ = get_action_head(env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=env.action_dim)
    critic_pre_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_post_torso = hydra.utils.instantiate(config.network.critic_network.post_torso)

    # Create S2MP network for actor
    # Get observation dimensions
    sample_obs = env.observation_spec.generate_value()
    if hasattr(sample_obs, "agents_view"):
        obs_dims = sample_obs.agents_view.shape[1:]
    else:
        obs_dims = sample_obs.shape[1:]
    obs_dim = obs_dims[0] if len(obs_dims) > 0 else 32  # fallback

    # Determine action space type and action_dim
    is_discrete = isinstance(
        env.action_spec, (DiscreteArray, MultiDiscreteArray, Discrete, MultiDiscrete)
    )
    action_space_type = "discrete" if is_discrete else "continuous"
    from mava.networks.s2mp_network import S2MPNetwork

    s2mp_torso = S2MPNetwork(
        obs_dim=obs_dim,
        action_dim=env.action_dim,
        n_agents=num_agents,
        K=getattr(config.system, "traj_len", 10),
        embed_dim=getattr(config.network, "s2mp_embed_dim", 64),
        num_heads=getattr(config.network, "s2mp_num_heads", 4),
        num_encoder_layers=getattr(config.network, "s2mp_num_encoder_layers", 3),
        num_decoder_layers=getattr(config.network, "s2mp_num_decoder_layers", 3),
        latent_tokens=getattr(config.network, "s2mp_latent_tokens", 4),
        mlp_dim=getattr(config.network, "s2mp_mlp_dim", 128),
        dropout_rate=getattr(config.network, "s2mp_dropout_rate", 0.0),
        action_space_type=action_space_type,
    )
    actor_network = Actor(
        pre_torso=actor_pre_torso,
        post_torso=actor_post_torso,
        action_head=actor_action_head,
        pred_torso=s2mp_torso,
        hidden_state_dim=config.network.hidden_state_dim,
        traj_len=getattr(config.system, "traj_len", 10),
        use_ma2e_fusion=config.system.use_ma2e_fusion,
        scan_fn=ScannedRNNPerAgent,
    )
    critic_network = Critic(
        pre_torso=critic_pre_torso,
        post_torso=critic_post_torso,
        hidden_state_dim=config.network.hidden_state_dim,
        centralised_critic=True,
        traj_len=getattr(config.system, "traj_len", 10),
        n_agent=num_agents,
        use_transformer_torso=config.system.use_transformer_torso,
    )

    actor_lr = make_learning_rate(config.system.actor_lr, config)
    critic_lr = make_learning_rate(config.system.critic_lr, config)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    # Initialise observation with obs of all agents.
    init_obs = env.observation_spec.generate_value()
    init_obs = tree.map(
        lambda x: jnp.repeat(x[jnp.newaxis, ...], config.arch.num_envs, axis=0),
        init_obs,
    )
    init_obs = tree.map(lambda x: x[jnp.newaxis, ...], init_obs)
    init_done = jnp.zeros((1, config.arch.num_envs, num_agents), dtype=bool)
    init_x = (init_obs, init_done)

    # Initialise hidden states.
    init_policy_hstate = ScannedRNNPerAgent.initialize_carry(
        config.arch.num_envs, config.network.hidden_state_dim
    )
    # # Duplicate the hidden states across the number of agents.
    init_policy_hstate = jnp.repeat(init_policy_hstate[:, jnp.newaxis, :], num_agents, axis=1)
    init_critic_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )

    # initialise params and optimiser state.

    # Create dummy joint trajectory for initialization
    sample_obs = env.observation_spec.generate_value()
    obs_shape = (
        sample_obs.agents_view.shape[1:]
        if hasattr(sample_obs, "agents_view")
        else sample_obs.shape[1:]
    )
    sample_action = env.action_spec.generate_value()
    action_shape = sample_action.shape[1:]
    action_dtype = jnp.array(sample_action).dtype
    traj_len = config.system.get("traj_len", 10)

    # Create dummy joint trajectory for each agent
    # Each agent needs the joint trajectory with proper batch dimension
    dummy_joint_traj_single = construct_dummy_joint_trajectory(
        obs_shape=obs_shape,
        action_shape=action_shape,
        batch_size=1,  # batch size for single call
        num_agents=num_agents,
        traj_len=traj_len,
        action_dtype=action_dtype,
    )

    # Expand for num_envs to match expected format: (1, num_envs, num_agents, traj_len, *dims)
    dummy_joint_traj_expanded = JointTrajectory(
        observations=jnp.broadcast_to(
            dummy_joint_traj_single.observations[jnp.newaxis, ...],  # Add leading dimension
            (1, config.arch.num_envs, num_agents, traj_len, *obs_shape),
        ),
        actions=jnp.broadcast_to(
            dummy_joint_traj_single.actions[jnp.newaxis, ...],  # Add leading dimension
            (1, config.arch.num_envs, num_agents, traj_len, *action_shape),
        ),
        last_actions=None,  # Dummy trajectory for initialization
        last_action_masks=None,  # Dummy trajectory for initialization
    )

    # actor net keys has agent dim at 0,
    # init_policy_hstate has agent dim at 1,
    # init_x has agent dims at (2, 2)
    # For joint_trajectory, we don't vmap it since each agent should see the same full trajectory
    actor_params = jax.vmap(actor_network.init, in_axes=(0, 1, (2, 2), None))(
        actor_net_keys, init_policy_hstate, init_x, dummy_joint_traj_expanded
    )
    actor_opt_state = jax.vmap(actor_optim.init)(actor_params)

    # We leave the critic as is since it is centralised.
    critic_params = critic_network.init(
        critic_net_key, init_critic_hstate, init_x, dummy_joint_traj_expanded
    )
    critic_opt_state = critic_optim.init(critic_params)

    # 检查是否需要进行S2MP预训练
    pretrain_s2mp_torso = getattr(config.system, "pretrain_s2mp_torso", False)

    if pretrain_s2mp_torso:
        single_actor_params = jax.tree.map(lambda x: jnp.copy(x[0]), actor_params)
        # Pretrain the S2MP component of the single actor
        print("正在进行S2MP预训练...")
        pretrained_single_actor_params = pretrain_s2mp_single_actor(
            single_actor_params=single_actor_params,
            actor_network=actor_network,
            obs_dim=obs_dim,
            action_dim=env.action_dim,
            n_agents=num_agents,
            traj_len=traj_len,
            action_space_type=action_space_type,
            pretrain_epochs=getattr(config.system, "s2mp_pretrain_epochs", 5),
            batch_size=getattr(config.system, "s2mp_pretrain_batch_size", 512),
            ratio_augment_size=getattr(config.system, "s2mp_ratio_augment_size", num_agents),
            random_augment_size=getattr(config.system, "s2mp_random_augment_size", num_agents),
            single_augment_size=getattr(config.system, "s2mp_single_augment_size", num_agents),
            learning_rate=getattr(config.system, "s2mp_pretrain_lr", 1e-3),
            vault_dir=getattr(config.system, "s2mp_vault_dir", None),
            vault_name=getattr(config.system, "s2mp_vault_name", None),
            vault_uid=getattr(config.system, "s2mp_vault_uid", None),
            num_training_samples=getattr(config.system, "s2mp_num_training_samples", 1000),
        )

        # 将single_actor_param中的关于pred_torso的参数数值copy到每个智能体自身的actor_network param中
        def copy_pred_torso_to_all_agents(actor_params, single_actor_params):
            """
            将single_actor_params中'pred_torso'部分的参数，复制到actor_params每个智能体自身的'pred_torso'部分。
            假设actor_params和single_actor_params都是PyTree结构，且'pred_torso'为一层key。
            """

            def replace_pred_torso(agent_param, single_param):
                # 遍历agent_param的所有key
                if isinstance(agent_param, dict):
                    new_param = {}
                    for k, v in agent_param.items():
                        if k == "pred_torso" and "pred_torso" in single_param:
                            new_param[k] = single_param["pred_torso"]
                        else:
                            new_param[k] = replace_pred_torso(v, single_param.get(k, {}))
                    return new_param
                elif isinstance(agent_param, (list, tuple)):
                    return type(agent_param)(
                        replace_pred_torso(
                            v,
                            single_param[i]
                            if isinstance(single_param, (list, tuple)) and i < len(single_param)
                            else {},
                        )
                        for i, v in enumerate(agent_param)
                    )
                else:
                    return agent_param

            # actor_params: (num_agents, ...)
            # single_actor_params: ...
            # 对每个agent的param进行替换
            if isinstance(actor_params, (list, tuple)):
                return type(actor_params)(
                    replace_pred_torso(agent_param, single_actor_params)
                    for agent_param in actor_params
                )
            elif hasattr(actor_params, "tree_map"):
                # 兼容flax/jax的PyTree结构
                return jax.tree.map(
                    lambda agent_param: replace_pred_torso(agent_param, single_actor_params),
                    actor_params,
                )
            else:
                # 假设是jnp.ndarray等
                return actor_params

        # 执行参数复制
        actor_params = copy_pred_torso_to_all_agents(actor_params, pretrained_single_actor_params)
        print("S2MP预训练完成，参数已复制到所有智能体")
    else:
        print("跳过S2MP预训练 (pretrain_s2mp_torso=False)")

    # Get network apply functions and optimiser updates.
    # Use __call__ for training and get_actions for execution/evaluation
    actor_train_apply_fn = actor_network.apply
    actor_exec_apply_fn = lambda params, *args, **kwargs: actor_network.apply(
        params, *args, method=actor_network.get_actions, **kwargs
    )
    apply_fns = ((actor_train_apply_fn, actor_exec_apply_fn), critic_network.apply)
    update_fns = (actor_optim.update, critic_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    # Map learner over specified devices
    devices = jax.local_devices()[:n_devices]
    learn = jax.pmap(learn, axis_name="device", devices=devices)

    # Pack params and initial states.
    params = Params(actor_params, critic_params)
    hstates = HiddenStates(init_policy_hstate, init_critic_hstate)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, restored_hstates = loaded_checkpoint.restore_params(
            input_params=params, restore_hstates=True, THiddenState=HiddenStates
        )
        # Update the params and hstates
        params = restored_params
        hstates = restored_hstates if restored_hstates else hstates

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )
    reshape_states = lambda x: x.reshape(
        (n_devices, config.system.update_batch_size, config.arch.num_envs, *x.shape[1:])
    )
    # (devices, update batch size, num_envs, ...)
    env_states = tree.map(reshape_states, env_states)
    timesteps = tree.map(reshape_states, timesteps)

    # Define params to be replicated across devices and batches.
    dones = jnp.zeros(
        (config.arch.num_envs, num_agents),
        dtype=bool,
    )

    # Initialize trajectory state
    traj_len = getattr(config.system, "traj_len", 10)
    # Get observation dims from first timestep
    sample_obs = env.observation_spec.generate_value()
    if hasattr(sample_obs, "agents_view"):
        obs_dims = sample_obs.agents_view.shape[1:]
    else:
        obs_dims = sample_obs.shape[1:]

    # Infer action shape and dtype from action_spec
    sample_action = env.action_spec.generate_value()  # shape: (num_agents, *act_dim)
    action_shape = sample_action.shape[1:]
    action_dtype = jnp.array(sample_action).dtype

    # Initialize trajectory buffers with zeros
    # obs_history: [num_envs, num_agents, traj_len, *obs_dims]
    init_obs_history = jnp.zeros((config.arch.num_envs, num_agents, traj_len, *obs_dims))
    # action_history: [num_envs, num_agents, traj_len, *action_shape]
    init_action_history = jnp.zeros(
        (config.arch.num_envs, num_agents, traj_len, *action_shape),
        dtype=action_dtype,
    )

    trajectory_state = TrajectoryState(
        obs_history=init_obs_history,
        action_history=init_action_history,
        buffer_idx=0,
        buffer_full=False,
    )

    key, step_keys = jax.random.split(key)
    opt_states = OptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states, hstates, step_keys, dones, trajectory_state)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *jnp.shape(x)))
    replicate_learner = tree.map(broadcast, replicate_learner)

    # Duplicate learner across specified devices
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=devices)

    # Initialise learner state.
    params, opt_states, hstates, step_keys, dones, trajectory_state = replicate_learner
    init_learner_state = RNNLearnerState(
        params=params,
        opt_states=opt_states,
        key=step_keys,
        env_state=env_states,
        timestep=timesteps,
        dones=dones,
        hstates=hstates,
        trajectory_state=trajectory_state,
    )
    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    _config.logger.system_name = "rec_happo"
    config = copy.deepcopy(_config)

    # Vault configuration
    save_vault = getattr(config.system, "save_vault", False)
    vault_name = getattr(config.system, "vault_name", "rec_happo")
    vault_uid = getattr(config.system, "vault_uid", None)
    vault_save_interval = getattr(config.system, "vault_save_interval", 5)

    # Determine number of devices: default to 1 unless override via config
    n_devices = min(config.arch.get("n_devices", 1), len(jax.devices()))

    # Set recurrent chunk size.
    if config.system.recurrent_chunk_size is None:
        config.system.recurrent_chunk_size = config.system.rollout_length
    else:
        assert config.system.rollout_length % config.system.recurrent_chunk_size == 0, (
            "Rollout length must be divisible by recurrent chunk size."
        )

        assert config.arch.num_envs % config.system.num_minibatches == 0, (
            "Number of envs must be divisibile by number of minibatches."
        )

    # Create the enviroments for train and eval.
    env, eval_env = environments.make(config=config, add_global_state=True)

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    # Setup evaluator.
    # One key per device for evaluation.
    eval_keys = jax.random.split(key_e, n_devices)

    # Determine action type and dimension for HAPPO evaluator
    from gymnasium.spaces import Discrete, MultiDiscrete
    from jumanji.specs import DiscreteArray, MultiDiscreteArray

    is_discrete = isinstance(
        env.action_spec, (DiscreteArray, MultiDiscreteArray, Discrete, MultiDiscrete)
    )
    action_type = "discrete" if is_discrete else "continuous"
    action_dim = None if is_discrete else env.action_dim

    # Create actor apply function for execution/evaluation using get_actions method
    actor_exec_apply_fn = lambda params, *args, **kwargs: actor_network.apply(
        params, *args, method=actor_network.get_actions, **kwargs
    )
    eval_act_fn = make_rec_eval_act_fn_with_traj(
        actor_exec_apply_fn, config, is_happo=True, action_type=action_type, action_dim=action_dim
    )
    evaluator = get_eval_fn_with_traj(eval_env, eval_act_fn, config, absolute_metric=False)

    # Calculate total timesteps.
    config = check_total_timesteps(config)
    assert config.system.num_updates > config.arch.num_evaluation, (
        "Number of updates per evaluation must be less than total number of updates."
    )

    # Calculate number of updates per evaluation.
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.system.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup
    logger = MavaLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Set up vault for experience storage if enabled
    buffer_state = None
    vault = None
    buffer_add = None
    if save_vault:
        # Get observation dims from environment spec
        sample_obs = env.observation_spec.generate_value()
        obs_shape = (
            sample_obs.agents_view.shape[1:]
            if hasattr(sample_obs, "agents_view")
            else sample_obs.shape[1:]
        )

        # Get global state shape
        global_state_shape = (
            sample_obs.global_state.shape
            if hasattr(sample_obs, "global_state")
            else (1,)  # fallback if no global state
        )

        # Get action spec to determine correct action shape and dtype
        sample_action = env.action_spec.generate_value()
        action_shape = sample_action.shape
        action_dtype = jnp.array(sample_action).dtype

        # Set up dummy transition based on actual environment specs
        dummy_flashbax_transition = {
            "done": jnp.zeros((config.system.num_agents,), dtype=bool),
            "action": jnp.zeros(action_shape, dtype=action_dtype),
            "reward": jnp.zeros((config.system.num_agents,), dtype=jnp.float32),
            "observation": jnp.zeros(
                (config.system.num_agents, *obs_shape),
                dtype=jnp.float32,
            ),
            "global_state": jnp.zeros(global_state_shape, dtype=jnp.float32),
            "legal_action_mask": jnp.zeros(
                (config.system.num_agents, env.action_dim),
                dtype=bool,
            ),
        }

        buffer = fbx.make_flat_buffer(
            max_length=int(5e5),  # Max number of transitions to store
            min_length=int(1),
            sample_batch_size=1,
            add_sequences=True,
            add_batch_size=(
                n_devices
                * config.system.num_updates_per_eval
                * config.system.update_batch_size
                * config.arch.num_envs
            ),
        )
        buffer_state = buffer.init(dummy_flashbax_transition)
        buffer_add = jax.jit(buffer.add, donate_argnums=(0))

        # Create vault
        vault = Vault(
            vault_name=vault_name,
            experience_structure=buffer_state.experience,
            vault_uid=vault_uid,
            metadata=OmegaConf.to_container(config, resolve=True),
        )

        # Shape legend:
        # D: Number of devices
        # NU: Number of updates per evaluation
        # UB: Update batch size
        # T: Time steps per rollout
        # NE: Number of environments

        @jax.jit
        def _reshape_experience(experience: Dict[str, chex.Array]) -> Dict[str, chex.Array]:
            """Reshape experience to match buffer."""
            # Swap the T and NE axes (D, NU, UB, T, NE, ...) -> (D, NU, UB, NE, T, ...)
            experience = tree.map(lambda x: x.swapaxes(3, 4), experience)
            # Merge 4 leading dimensions into 1. (D, NU, UB, NE, T ...) -> (D * NU * UB * NE, T, ...)
            experience = tree.map(lambda x: x.reshape(-1, *x.shape[4:]), experience)
            return experience

    # Create an initial hidden state and trajectory history used for resetting memory for evaluation
    eval_batch_size = get_num_eval_envs(config, absolute_metric=False)
    # Single-device hidden state: [batch, agents, hidden_dim]
    eval_hs_single = ScannedRNNPerAgent.initialize_carry(
        eval_batch_size,
        config.network.hidden_state_dim,
    )
    eval_hs_single = jnp.repeat(eval_hs_single[:, jnp.newaxis, :], env.num_agents, axis=1)
    # Broadcast across devices: [n_devices, batch, agents, hidden_dim]
    eval_hs = jnp.broadcast_to(eval_hs_single[jnp.newaxis, ...], (n_devices, *eval_hs_single.shape))

    # Initialize trajectory history for evaluation - make sure structure matches what eval_act_fn expects
    traj_len = getattr(config.system, "traj_len", 10)
    sample_obs = env.observation_spec.generate_value()
    obs_shape = (
        sample_obs.agents_view.shape[1:]
        if hasattr(sample_obs, "agents_view")
        else sample_obs.shape[1:]
    )
    sample_action = env.action_spec.generate_value()
    action_shape = sample_action.shape[1:]
    action_dtype = jnp.array(sample_action).dtype

    # Initialize empty trajectory history for evaluation
    eval_traj_history = {
        "obs_history": jnp.zeros(
            (n_devices, eval_batch_size, env.num_agents, traj_len, *obs_shape)
        ),
        "action_history": jnp.zeros(
            (n_devices, eval_batch_size, env.num_agents, traj_len, *action_shape),
            dtype=action_dtype,
        ),
    }

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = None
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()
        learner_output, experience_to_store = learn(learner_state)

        # Record data into the vault if enabled
        if save_vault:
            # Pack transition
            flashbax_transition = _reshape_experience(
                {
                    # (D, NU, UB, T, NE, ...)
                    "done": experience_to_store.done,
                    "action": experience_to_store.action,
                    "reward": experience_to_store.reward,
                    "observation": experience_to_store.obs.agents_view,
                    "global_state": experience_to_store.obs.global_state,
                    "legal_action_mask": experience_to_store.obs.action_mask,
                }
            )
            # Add to fbx buffer
            buffer_state = buffer_add(buffer_state, flashbax_transition)

            # Save buffer into vault
            if eval_step % vault_save_interval == 0:
                write_length = vault.write(buffer_state)
                print(f"(Wrote {write_length}) Vault index = {vault.vault_index}")

        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Calculate remaining training time
        total_steps = steps_per_rollout * config.arch.num_evaluation
        remaining_steps = total_steps - t
        steps_per_second = steps_per_rollout / elapsed_time
        remaining_minutes = (remaining_steps / steps_per_second) / 60 if steps_per_second > 0 else 0

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log(
            {"timestep": t, "remaining_minutes": remaining_minutes}, t, eval_step, LogEvent.MISC
        )
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        start_time = time.time()

        trained_params = unreplicate_batch_dim(learner_state.params.actor_params)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        # Evaluate.
        eval_metrics = evaluator(
            trained_params,
            eval_keys,
            {"hidden_state": eval_hs, "trajectory_history": eval_traj_history},
        )
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
        episode_return = jnp.mean(eval_metrics["episode_return"])

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Final write to vault for any remaining data
    if save_vault and vault is not None:
        vault.write(buffer_state)

    # Record the performance for the final evaluation run.
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    # Measure absolute metric.
    if config.arch.absolute_metric:
        start_time = time.time()

        eval_batch_size = get_num_eval_envs(config, absolute_metric=True)
        # Single-device hidden state for absolute eval
        eval_hs_single = ScannedRNNPerAgent.initialize_carry(
            eval_batch_size,
            config.network.hidden_state_dim,
        )
        eval_hs_single = jnp.repeat(eval_hs_single[:, jnp.newaxis, :], env.num_agents, axis=1)
        # Broadcast across devices
        eval_hs = jnp.broadcast_to(
            eval_hs_single[jnp.newaxis, ...], (n_devices, *eval_hs_single.shape)
        )

        # Initialize trajectory history for absolute metric evaluation
        eval_traj_history_abs = {
            "obs_history": jnp.zeros(
                (n_devices, eval_batch_size, env.num_agents, traj_len, *obs_shape)
            ),
            "action_history": jnp.zeros(
                (n_devices, eval_batch_size, env.num_agents, traj_len, *action_shape),
                dtype=action_dtype,
            ),
        }

        abs_metric_evaluator = get_eval_fn_with_traj(
            eval_env, eval_act_fn, config, absolute_metric=True
        )
        eval_keys = jax.random.split(key, n_devices)

        eval_metrics = abs_metric_evaluator(
            best_params,
            eval_keys,
            {"hidden_state": eval_hs, "trajectory_history": eval_traj_history_abs},
        )

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()

    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="rec_happo.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}Recurrent HAPPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
