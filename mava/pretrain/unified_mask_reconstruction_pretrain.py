"""
统一的多智能体掩码-重构预训练脚本

这个脚本提供了一个标准化的框架，适用于多智能体场景的掩码-重构任务预训练。
支持不同的网络架构（S2MP, MA2E等），实现三种标准化的掩码策略。

主要特性：
1. 网络标准化定义：支持多种网络架构的统一接口
2. 数据读取标准化：从vault读取并构建训练/验证集
3. 掩码策略标准化：三种掩码策略的实现
4. 预训练流程标准化：完整的训练/验证/保存流程
"""

import argparse
import os
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import chex
import flashbax as fbx
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

# 新增imports用于可视化
import pandas as pd
from flashbax.vault import Vault
from flax import serialization
from flax.training.train_state import TrainState
from tqdm import tqdm

try:
    from tabulate import tabulate

    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print(
        "警告: 未安装tabulate库，将使用简化的表格输出。运行 'pip install tabulate' 以获得更好的表格格式。"
    )

try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print(
        "警告: 未安装matplotlib库，将跳过误差图表绘制。运行 'pip install matplotlib' 以启用图表功能。"
    )

from mava.networks.ma2e_network import MA2ENetwork
from mava.networks.s2mp_network import S2MPNetwork


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


# SMAX观测解析常量
# 基于smax_env.py中的定义
SMAX_OWN_FEATURES = ["health", "position_x", "position_y", "weapon_cooldown"]
SMAX_UNIT_FEATURES = [
    "health",
    "position_x",
    "position_y",
    "last_movement_x",
    "last_movement_y",
    "last_targeted",
    "weapon_cooldown",
]
SMAX_UNIT_TYPE_NAMES = ["marine", "marauder", "stalker", "zealot", "zergling", "hydralisk"]
SMAX_MAP_WIDTH = 32
SMAX_MAP_HEIGHT = 32


# -----------------------------------------------------------------------------
# SMAX观测解析和可视化函数
# -----------------------------------------------------------------------------


def parse_smax_observation(
    obs: chex.Array, n_agents: int, unit_type_bits: int = 6
) -> Dict[str, chex.Array]:
    """
    解析SMAX观测，提取关键信息

    Args:
        obs: 原始观测 (obs_dim,) - 已去掉前n_agents维的one-hot id
        n_agents: 智能体数量
        unit_type_bits: 单位类型位数 (默认6种单位类型)

    Returns:
        包含解析信息的字典
    """
    # 确保使用numpy数组以避免JAX索引问题
    obs = np.array(obs)

    # 计算各部分维度
    own_features_dim = len(SMAX_OWN_FEATURES) + unit_type_bits  # 4 + 6 = 10
    unit_features_dim = len(SMAX_UNIT_FEATURES) + unit_type_bits  # 7 + 6 = 13

    # unit_list观测结构：[其他单位特征, 自身特征]
    # 其他单位特征：(n_agents-1) * unit_features_dim
    other_units_dim = (n_agents - 1) * unit_features_dim

    # 检查观测维度是否足够
    if len(obs) < other_units_dim + own_features_dim:
        # 观测维度不够，返回默认值
        return {
            "own_health": 0.0,
            "own_pos_x": 0.0,
            "own_pos_y": 0.0,
            "own_weapon_cooldown": 0.0,
            "own_unit_type": 0,
            "own_unit_type_name": "unknown",
            "other_units": [],
        }

    # 提取各部分
    other_units_obs = obs[:other_units_dim]
    own_obs = obs[other_units_dim : other_units_dim + own_features_dim]

    # 检查自身观测维度
    if len(own_obs) < 4:
        # 自身观测维度不够
        return {
            "own_health": 0.0,
            "own_pos_x": 0.0,
            "own_pos_y": 0.0,
            "own_weapon_cooldown": 0.0,
            "own_unit_type": 0,
            "own_unit_type_name": "unknown",
            "other_units": [],
        }

    # 解析自身特征
    own_health = float(own_obs[0])
    own_pos_x = float(own_obs[1]) * SMAX_MAP_WIDTH  # 反归一化位置
    own_pos_y = float(own_obs[2]) * SMAX_MAP_HEIGHT
    own_weapon_cooldown = float(own_obs[3])

    # 解析单位类型
    if len(own_obs) >= 4 + unit_type_bits:
        own_unit_type_bits = own_obs[4 : 4 + unit_type_bits]
        max_val = float(np.max(own_unit_type_bits))
        own_unit_type = int(np.argmax(own_unit_type_bits)) if max_val > 0 else 0
    else:
        own_unit_type = 0

    # 解析其他单位特征
    other_units_info = []
    for i in range(n_agents - 1):
        start_idx = i * unit_features_dim
        end_idx = start_idx + unit_features_dim
        unit_obs = other_units_obs[start_idx:end_idx]

        if len(unit_obs) >= unit_features_dim:
            unit_health = float(unit_obs[0])
            # 相对位置需要加上自身位置才是绝对位置
            unit_rel_x = float(unit_obs[1])
            unit_rel_y = float(unit_obs[2])
            unit_weapon_cooldown = float(unit_obs[6]) if len(unit_obs) > 6 else 0.0

            # 单位类型
            if len(unit_obs) >= 7 + unit_type_bits:
                unit_type_bits_arr = unit_obs[7 : 7 + unit_type_bits]
                max_val = float(np.max(unit_type_bits_arr))
                unit_type = int(np.argmax(unit_type_bits_arr)) if max_val > 0 else 0
            else:
                unit_type = 0

            other_units_info.append(
                {
                    "health": float(unit_health),
                    "rel_pos_x": float(unit_rel_x),
                    "rel_pos_y": float(unit_rel_y),
                    "weapon_cooldown": float(unit_weapon_cooldown),
                    "unit_type": int(unit_type),
                    "unit_type_name": SMAX_UNIT_TYPE_NAMES[unit_type]
                    if 0 <= unit_type < len(SMAX_UNIT_TYPE_NAMES)
                    else "unknown",
                }
            )

    return {
        "own_health": float(own_health),
        "own_pos_x": float(own_pos_x),
        "own_pos_y": float(own_pos_y),
        "own_weapon_cooldown": float(own_weapon_cooldown),
        "own_unit_type": int(own_unit_type),
        "own_unit_type_name": SMAX_UNIT_TYPE_NAMES[own_unit_type]
        if 0 <= own_unit_type < len(SMAX_UNIT_TYPE_NAMES)
        else "unknown",
        "other_units": other_units_info,
    }


def compute_observation_errors(
    pred_obs: chex.Array, true_obs: chex.Array, n_agents: int, unit_type_bits: int = 6
) -> Dict[str, Dict]:
    """
    计算预测观测和真实观测之间的误差

    Args:
        pred_obs: 预测观测 (n_agents, obs_dim) - 已去掉one-hot id
        true_obs: 真实观测 (n_agents, obs_dim) - 已去掉one-hot id
        n_agents: 智能体数量
        unit_type_bits: 单位类型位数

    Returns:
        每个智能体的误差信息
    """
    # 确保使用numpy数组
    pred_obs = np.array(pred_obs)
    true_obs = np.array(true_obs)

    agent_errors = {}

    for agent_idx in range(n_agents):
        # 解析预测和真实观测
        pred_parsed = parse_smax_observation(pred_obs[agent_idx], n_agents, unit_type_bits)
        true_parsed = parse_smax_observation(true_obs[agent_idx], n_agents, unit_type_bits)

        # 计算自身特征误差
        own_errors = {
            "health_error": float(abs(pred_parsed["own_health"] - true_parsed["own_health"])),
            "pos_x_error": float(abs(pred_parsed["own_pos_x"] - true_parsed["own_pos_x"])),
            "pos_y_error": float(abs(pred_parsed["own_pos_y"] - true_parsed["own_pos_y"])),
            "weapon_cd_error": float(
                abs(pred_parsed["own_weapon_cooldown"] - true_parsed["own_weapon_cooldown"])
            ),
            "type_match": bool(pred_parsed["own_unit_type"] == true_parsed["own_unit_type"]),
        }

        # 计算其他单位误差
        other_units_errors = []
        min_len = min(len(pred_parsed["other_units"]), len(true_parsed["other_units"]))

        for i in range(min_len):
            pred_unit = pred_parsed["other_units"][i]
            true_unit = true_parsed["other_units"][i]

            unit_error = {
                "health_error": float(abs(pred_unit["health"] - true_unit["health"])),
                "rel_pos_x_error": float(abs(pred_unit["rel_pos_x"] - true_unit["rel_pos_x"])),
                "rel_pos_y_error": float(abs(pred_unit["rel_pos_y"] - true_unit["rel_pos_y"])),
                "weapon_cd_error": float(
                    abs(pred_unit["weapon_cooldown"] - true_unit["weapon_cooldown"])
                ),
                "type_match": bool(pred_unit["unit_type"] == true_unit["unit_type"]),
            }
            other_units_errors.append(unit_error)

        agent_errors[f"agent_{agent_idx}"] = {
            "own_errors": own_errors,
            "other_units_errors": other_units_errors,
            "true_parsed": true_parsed,
            "pred_parsed": pred_parsed,
        }

    return agent_errors


def plot_position_errors(agent_errors: Dict[str, Dict], example_idx: int, save_path: str = None):
    """绘制位置误差可视化图 - 在一个平面图上显示所有agent的位置对比"""
    if not HAS_MATPLOTLIB:
        return

    # 设置中文字体支持（如果可用）
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 创建单个大图显示所有agent
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plt.style.use("seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default")

    # 为每个agent设置不同的颜色和标记
    colors = ["red", "blue", "orange", "purple", "brown", "pink", "gray", "olive"]
    markers = ["o", "s", "^", "D", "v", "p", "*", "h"]

    all_positions = []  # 存储所有位置用于计算误差统计

    for idx, (agent_name, errors) in enumerate(agent_errors.items()):
        agent_idx = int(agent_name.split("_")[1]) if "_" in agent_name else idx
        color = colors[agent_idx % len(colors)]
        marker_true = markers[agent_idx % len(markers)]
        marker_pred = "s"  # 统一用方形表示预测位置

        true_parsed = errors["true_parsed"]
        pred_parsed = errors["pred_parsed"]

        # 获取自身位置
        true_x, true_y = true_parsed["own_pos_x"], true_parsed["own_pos_y"]
        pred_x, pred_y = pred_parsed["own_pos_x"], pred_parsed["own_pos_y"]

        # 绘制真实位置（实心圆）
        ax.scatter(
            true_x,
            true_y,
            c=color,
            s=150,
            marker=marker_true,
            label=f"Agent {agent_idx} True",
            alpha=0.8,
            edgecolors="black",
            linewidth=2,
        )

        # 绘制预测位置（空心方形）
        ax.scatter(
            pred_x,
            pred_y,
            facecolors="none",
            edgecolors=color,
            s=150,
            marker=marker_pred,
            label=f"Agent {agent_idx} Pred",
            alpha=0.8,
            linewidth=2,
        )

        # 绘制误差连线和箭头
        distance = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
        ax.annotate(
            "",
            xy=(pred_x, pred_y),
            xytext=(true_x, true_y),
            arrowprops=dict(arrowstyle="->", color=color, lw=2, alpha=0.7),
        )

        # 在连线中点添加误差距离标注
        mid_x, mid_y = (true_x + pred_x) / 2, (true_y + pred_y) / 2
        ax.text(
            mid_x,
            mid_y,
            f"{distance:.1f}",
            fontsize=8,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

        all_positions.append(
            {
                "agent": agent_idx,
                "true_x": true_x,
                "true_y": true_y,
                "pred_x": pred_x,
                "pred_y": pred_y,
                "distance": distance,
            }
        )

    # 设置图表属性
    ax.set_xlim(0, SMAX_MAP_WIDTH)
    ax.set_ylim(0, SMAX_MAP_HEIGHT)
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)

    # 计算总体误差统计
    distances = [pos["distance"] for pos in all_positions]
    avg_distance = np.mean(distances)
    max_distance = np.max(distances)

    ax.set_title(
        f"Example {example_idx + 1} - Position Prediction Errors\n"
        f"Avg Distance Error: {avg_distance:.2f}, Max: {max_distance:.2f}",
        fontsize=14,
        fontweight="bold",
    )

    # 添加图例
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # 添加误差统计文本框
    stats_text = f"Error Statistics:\n"
    for pos in all_positions:
        stats_text += f"Agent {pos['agent']}: {pos['distance']:.2f}\n"

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(
            f"{save_path}/position_errors_example_{example_idx + 1}.png",
            dpi=150,
            bbox_inches="tight",
        )
        print(
            f"Position error plot saved to: {save_path}/position_errors_example_{example_idx + 1}.png"
        )
    else:
        plt.show()
    plt.close()


def plot_health_errors(agent_errors: Dict[str, Dict], example_idx: int, save_path: str = None):
    """绘制血量误差对比图"""
    if not HAS_MATPLOTLIB:
        return

    # 设置字体
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    agents = list(agent_errors.keys())
    n_agents = len(agents)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.style.use("seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default")

    # 自身血量对比
    true_healths = []
    pred_healths = []
    health_errors = []

    for agent_name, errors in agent_errors.items():
        true_healths.append(errors["true_parsed"]["own_health"])
        pred_healths.append(errors["pred_parsed"]["own_health"])
        health_errors.append(errors["own_errors"]["health_error"])

    x = np.arange(len(agents))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2, true_healths, width, label="True Health", color="green", alpha=0.7
    )
    bars2 = ax1.bar(
        x + width / 2, pred_healths, width, label="Predicted Health", color="red", alpha=0.7
    )

    ax1.set_xlabel("Agent")
    ax1.set_ylabel("Health Value")
    ax1.set_title("Health Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Agent {i}" for i in range(n_agents)])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 血量误差柱状图
    bars3 = ax2.bar(x, health_errors, color="orange", alpha=0.7)
    ax2.set_xlabel("Agent")
    ax2.set_ylabel("Health Error (Absolute)")
    ax2.set_title("Health Prediction Error")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Agent {i}" for i in range(n_agents)])
    ax2.grid(True, alpha=0.3)

    # 添加误差数值标签
    for bar in bars3:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.001,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.suptitle(
        f"Example {example_idx + 1} - Health Error Analysis", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(
            f"{save_path}/health_errors_example_{example_idx + 1}.png", dpi=150, bbox_inches="tight"
        )
        print(
            f"Health error plot saved to: {save_path}/health_errors_example_{example_idx + 1}.png"
        )
    else:
        plt.show()
    plt.close()


def plot_agent_errors_summary(
    agent_errors: Dict[str, Dict], example_idx: int, save_path: str = None
):
    """绘制智能体误差综合摘要图"""
    if not HAS_MATPLOTLIB:
        return

    # 设置字体
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    agents = list(agent_errors.keys())
    n_agents = len(agents)

    # 准备数据
    metrics = ["Health Error", "X Position Error", "Y Position Error", "Weapon CD Error"]
    agent_data = []

    for agent_name, errors in agent_errors.items():
        own_errors = errors["own_errors"]
        agent_data.append(
            [
                own_errors["health_error"],
                own_errors["pos_x_error"],
                own_errors["pos_y_error"],
                own_errors["weapon_cd_error"],
            ]
        )

    agent_data = np.array(agent_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.style.use("seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default")

    # 热力图
    im = ax1.imshow(agent_data.T, cmap="YlOrRd", aspect="auto")
    ax1.set_xticks(range(n_agents))
    ax1.set_yticks(range(len(metrics)))
    ax1.set_xticklabels([f"Agent {i}" for i in range(n_agents)])
    ax1.set_yticklabels(metrics)
    ax1.set_title("Error Heatmap")

    # 添加数值标签
    for i in range(len(metrics)):
        for j in range(n_agents):
            text = ax1.text(
                j,
                i,
                f"{agent_data[j, i]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label("Error Value")

    # 综合误差柱状图
    x = np.arange(len(metrics))
    width = 0.8 / n_agents

    colors = plt.cm.Set3(np.linspace(0, 1, n_agents))

    for i, agent_name in enumerate(agents):
        offset = (i - n_agents / 2 + 0.5) * width
        bars = ax2.bar(
            x + offset, agent_data[i], width, label=f"Agent {i}", color=colors[i], alpha=0.8
        )

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax2.set_xlabel("Error Type")
    ax2.set_ylabel("Error Value")
    ax2.set_title("Error Type Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Example {example_idx + 1} - Agent Error Summary Analysis", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(
            f"{save_path}/summary_errors_example_{example_idx + 1}.png",
            dpi=150,
            bbox_inches="tight",
        )
        print(
            f"Summary error plot saved to: {save_path}/summary_errors_example_{example_idx + 1}.png"
        )
    else:
        plt.show()
    plt.close()


def display_prediction_errors(
    agent_errors: Dict[str, Dict], example_idx: int, save_plots: bool = False, save_path: str = None
):
    """
    以表格形式展示预测误差，并可选地绘制可视化图表
    """
    print(f"\n{'=' * 80}")
    print(f"Example {example_idx + 1} - 预测误差分析")
    print(f"{'=' * 80}")

    for agent_name, errors in agent_errors.items():
        print(f"\n{agent_name.upper()} 误差分析:")
        print("-" * 60)

        # 自身特征误差表格
        own_errors = errors["own_errors"]
        true_parsed = errors["true_parsed"]
        pred_parsed = errors["pred_parsed"]

        own_data = [
            [
                "血量",
                f"{true_parsed['own_health']:.3f}",
                f"{pred_parsed['own_health']:.3f}",
                f"{own_errors['health_error']:.3f}",
            ],
            [
                "位置X",
                f"{true_parsed['own_pos_x']:.2f}",
                f"{pred_parsed['own_pos_x']:.2f}",
                f"{own_errors['pos_x_error']:.2f}",
            ],
            [
                "位置Y",
                f"{true_parsed['own_pos_y']:.2f}",
                f"{pred_parsed['own_pos_y']:.2f}",
                f"{own_errors['pos_y_error']:.2f}",
            ],
            [
                "武器冷却",
                f"{true_parsed['own_weapon_cooldown']:.3f}",
                f"{pred_parsed['own_weapon_cooldown']:.3f}",
                f"{own_errors['weapon_cd_error']:.3f}",
            ],
            [
                "单位类型",
                true_parsed["own_unit_type_name"],
                pred_parsed["own_unit_type_name"],
                "✓" if own_errors["type_match"] else "✗",
            ],
        ]

        print("自身特征:")
        if HAS_TABULATE:
            print(tabulate(own_data, headers=["特征", "真实值", "预测值", "误差"], tablefmt="grid"))
        else:
            # 简化的表格输出
            print(f"{'特征':<10} {'真实值':<12} {'预测值':<12} {'误差':<10}")
            print("-" * 50)
            for row in own_data:
                print(f"{row[0]:<10} {row[1]:<12} {row[2]:<12} {row[3]:<10}")

        # 其他单位误差表格
        if errors["other_units_errors"]:
            print(f"\n其他单位特征 (共{len(errors['other_units_errors'])}个单位):")

            other_data = []
            for i, unit_error in enumerate(errors["other_units_errors"]):
                true_unit = true_parsed["other_units"][i]
                pred_unit = pred_parsed["other_units"][i]

                other_data.append(
                    [
                        f"单位{i + 1}",
                        f"{true_unit['health']:.3f}",
                        f"{pred_unit['health']:.3f}",
                        f"{unit_error['health_error']:.3f}",
                        f"{true_unit['rel_pos_x']:.3f}",
                        f"{pred_unit['rel_pos_x']:.3f}",
                        f"{unit_error['rel_pos_x_error']:.3f}",
                        f"{true_unit['rel_pos_y']:.3f}",
                        f"{pred_unit['rel_pos_y']:.3f}",
                        f"{unit_error['rel_pos_y_error']:.3f}",
                        true_unit["unit_type_name"],
                        pred_unit["unit_type_name"],
                        "✓" if unit_error["type_match"] else "✗",
                    ]
                )

            if HAS_TABULATE:
                headers = [
                    "单位",
                    "真实血量",
                    "预测血量",
                    "血量误差",
                    "真实相对X",
                    "预测相对X",
                    "X误差",
                    "真实相对Y",
                    "预测相对Y",
                    "Y误差",
                    "真实类型",
                    "预测类型",
                    "类型匹配",
                ]
                print(tabulate(other_data, headers=headers, tablefmt="grid"))
            else:
                # 简化的表格输出
                print(f"{'单位':<6} {'血量误差':<8} {'X误差':<8} {'Y误差':<8} {'类型匹配':<8}")
                print("-" * 45)
                for row in other_data:
                    print(f"{row[0]:<6} {row[3]:<8} {row[6]:<8} {row[9]:<8} {row[12]:<8}")

    # 绘制可视化图表
    if HAS_MATPLOTLIB:
        print(f"\n📊 Drawing error visualization charts...")

        # 绘制位置误差图
        plot_position_errors(agent_errors, example_idx, save_path if save_plots else None)

        # 绘制血量误差图
        plot_health_errors(agent_errors, example_idx, save_path if save_plots else None)

        # 绘制综合误差图
        plot_agent_errors_summary(agent_errors, example_idx, save_path if save_plots else None)

        print(f"✓ Error visualization charts completed")
    else:
        print(f"\n⚠️  matplotlib not installed, skipping chart generation")


# -----------------------------------------------------------------------------
# 网络接口标准化
# -----------------------------------------------------------------------------


class UnifiedNetworkInterface:
    """统一网络接口，适配不同的网络架构"""

    def __init__(self, network_type: str, network_params: Dict[str, Any]):
        self.network_type = network_type.lower()
        self.network_params = network_params

        if self.network_type == "s2mp":
            self.network = S2MPNetwork(**network_params)
        elif self.network_type == "ma2e":
            self.network = MA2ENetwork(**network_params)
        else:
            raise ValueError(f"Unsupported network type: {network_type}")

        # 创建JIT编译的函数
        self._create_jit_functions()

    def _create_jit_functions(self):
        """创建JIT编译的网络函数"""
        # JIT编译重构函数
        if self.network_type == "s2mp":

            @profile_time(f"S2MP重构函数")
            @jax.jit
            def _jit_reconstruction(params, obs_seq, action_seq, agent_mask):
                return self.network.apply(params, obs_seq, action_seq, agent_mask, rngs={})

            self._jit_reconstruction = _jit_reconstruction

        elif self.network_type == "ma2e":

            @profile_time(f"MA2E重构函数")
            @partial(jax.jit, static_argnums=(4,))  # deterministic是静态参数
            def _jit_reconstruction(params, obs_seq, action_seq, agent_mask, deterministic):
                return self.network.apply(
                    params, obs_seq, action_seq, agent_mask, deterministic=deterministic
                )

            self._jit_reconstruction = _jit_reconstruction

        # JIT编译推理函数
        @profile_time(f"{self.network_type.upper()}推理函数")
        @partial(jax.jit, static_argnums=(4, 5))  # agent_idx和deterministic是静态参数
        def _jit_inference(
            params, single_agent_traj, current_gt_obs, agent_mask_template, agent_idx, deterministic
        ):
            return self._inference_core(
                params,
                single_agent_traj,
                current_gt_obs,
                agent_mask_template,
                agent_idx,
                deterministic,
            )

        self._jit_inference = _jit_inference

    def init(
        self, rng: chex.PRNGKey, obs_seq: chex.Array, action_seq: chex.Array, agent_mask: chex.Array
    ) -> Any:
        """初始化网络参数"""
        if self.network_type == "s2mp":
            return self.network.init(rng, obs_seq, action_seq, agent_mask)
        elif self.network_type == "ma2e":
            return self.network.init(rng, obs_seq, action_seq, agent_mask)

    def apply_reconstruction(
        self,
        params: Any,
        obs_seq: chex.Array,
        action_seq: chex.Array,
        agent_mask: chex.Array,
        deterministic: bool = False,
    ) -> Union[chex.Array, Tuple[chex.Array, chex.Array]]:
        """应用网络进行重构（JIT加速版本）"""
        if self.network_type == "s2mp":
            return self._jit_reconstruction(params, obs_seq, action_seq, agent_mask)
        elif self.network_type == "ma2e":
            return self._jit_reconstruction(params, obs_seq, action_seq, agent_mask, deterministic)

    def _inference_core(
        self,
        params: Any,
        single_agent_traj: chex.Array,
        current_gt_obs: chex.Array,
        agent_mask_template: chex.Array,
        agent_idx: int,
        deterministic: bool = True,
    ) -> chex.Array:
        """推理函数核心逻辑（JIT友好）"""
        batch_size, traj_len, obs_dim = single_agent_traj.shape
        num_agents = current_gt_obs.shape[1]

        # 调试：如果智能体数量为0，直接返回空结果
        if num_agents == 0:
            return jnp.zeros((batch_size, 0, obs_dim))

        # 构造输入序列：只保留指定智能体的历史，其他智能体用零填充
        obs_seq = jnp.zeros((batch_size, num_agents, traj_len, obs_dim))
        obs_seq = obs_seq.at[:, agent_idx, :, :].set(single_agent_traj)

        # 构造虚拟动作序列（预训练时可能需要）
        action_dim_param = self.network_params.get("action_dim", 1)
        if self.network_type == "s2mp":
            action_seq = jnp.zeros((batch_size, num_agents, traj_len))  # S2MP期望3维
        else:  # ma2e
            if self.network_params.get("action_space_type", "discrete") == "discrete":
                # MA2E期望离散动作也是4维的，已归一化
                action_seq = jnp.zeros((batch_size, num_agents, traj_len, 1))
            else:
                action_seq = jnp.zeros((batch_size, num_agents, traj_len, action_dim_param))

        # 使用预构建的mask模板并设置对应的agent
        agent_mask = agent_mask_template.at[:, agent_idx].set(1.0)

        # 获取重构结果
        if self.network_type == "s2mp":
            pred_obs = self._jit_reconstruction(params, obs_seq, action_seq, agent_mask)
            # S2MP只返回观测重构
            predicted_obs = pred_obs[:, :, -1, :]  # 取最后一个时间步
        else:  # ma2e
            pred_obs, pred_action = self._jit_reconstruction(
                params, obs_seq, action_seq, agent_mask, deterministic
            )
            predicted_obs = pred_obs[:, :, -1, :]  # 取最后一个时间步

        # 用真实观测替换指定智能体的预测观测
        predicted_obs = predicted_obs.at[:, agent_idx, :].set(current_gt_obs[:, agent_idx, :])

        return predicted_obs

    def apply_inference(
        self,
        params: Any,
        single_agent_traj: chex.Array,
        current_gt_obs: chex.Array,
        agent_idx: int,
        deterministic: bool = True,
    ) -> chex.Array:
        """推理函数：基于单智能体历史轨迹预测当前timestep所有智能体观测（JIT加速版本）"""
        # single_agent_traj: (B, traj_len, obs_dim)
        # current_gt_obs: (B, num_agents, obs_dim) - 当前时刻所有智能体的真实观测
        # 返回: (B, num_agents, obs_dim) - 预测的所有智能体观测

        batch_size = single_agent_traj.shape[0]
        num_agents = current_gt_obs.shape[1]

        # 检查输入有效性
        if num_agents == 0:
            print(f"警告: num_agents为0，返回空结果")
            return jnp.zeros(
                (batch_size, 0, current_gt_obs.shape[2] if len(current_gt_obs.shape) > 2 else 1)
            )

        if agent_idx >= num_agents:
            print(f"警告: agent_idx({agent_idx}) >= num_agents({num_agents})，调整为0")
            agent_idx = 0

        # 创建零初始化的mask模板
        agent_mask_template = jnp.zeros((batch_size, num_agents))

        return self._jit_inference(
            params, single_agent_traj, current_gt_obs, agent_mask_template, agent_idx, deterministic
        )


# -----------------------------------------------------------------------------
# 数据处理函数
# -----------------------------------------------------------------------------


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


def build_training_dataset(
    data: Any, buffer: Any, num_samples: int, action_space_type: str, action_dim: int
) -> List[Tuple[chex.Array, chex.Array]]:
    """构建训练数据集"""
    print(f"构建训练数据集，目标样本数: {num_samples}")
    training_data = []

    # 使用jit加速采样
    buffer_sample = jax.jit(buffer.sample)

    for i in tqdm(range(num_samples), desc="构建训练集"):
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
            # 连续动作：确保是4维并归一化（假设连续动作已经在合理范围内）
            if action_seq.ndim == 3:  # (B, T, N) -> (B, T, N, 1)
                action_seq = action_seq[..., None]

        training_data.append((obs_seq[0], action_seq[0]))  # 去掉batch维度

    print(f"训练数据集构建完成，实际样本数: {len(training_data)}")
    return training_data


def build_validation_dataset(
    data: Any,
    min_reward_threshold: float,
    num_episodes: int,
    traj_length: int,
    action_space_type: str,
    action_dim: int,
) -> List[Tuple[chex.Array, chex.Array]]:
    """构建验证数据集：基于累计回报阈值筛选高质量episodes"""
    print(f"构建验证数据集，回报阈值: {min_reward_threshold}, 目标episodes: {num_episodes}")

    rewards = np.array(data.experience["reward"])  # (B, T, N)
    dones = np.array(data.experience["done"])  # (B, T, N)
    obs = np.array(data.experience["observation"])  # (B, T, N, obs_dim)
    actions = np.array(data.experience["action"])  # (B, T, N) 或 (B, T, N, action_dim)

    batch_size_data, seq_len, n_agents = rewards.shape
    high_reward_episodes = []

    # 寻找高回报episodes
    for batch_idx in range(batch_size_data):
        cumulative_reward = 0.0
        episode_start = 0

        for t in range(seq_len):
            cumulative_reward += np.sum(rewards[batch_idx, t, :])  # 累计多智能体共享奖励

            if np.any(dones[batch_idx, t, :]):  # episode结束
                if cumulative_reward > min_reward_threshold:
                    high_reward_episodes.append((batch_idx, episode_start, t, cumulative_reward))
                    print(
                        f"找到高回报episode: batch {batch_idx}, steps {episode_start}-{t}, reward {cumulative_reward:.3f}"
                    )

                    if len(high_reward_episodes) >= num_episodes:
                        break

                episode_start = t + 1
                cumulative_reward = 0.0

        if len(high_reward_episodes) >= num_episodes:
            break

    if not high_reward_episodes:
        print(f"警告：未找到回报大于 {min_reward_threshold} 的episodes")
        return []

    # 从高回报episodes中构建验证序列
    validation_data = []
    for batch_idx, start_idx, end_idx, reward in high_reward_episodes:
        # 从每个episode中提取多个长度为traj_length的序列
        for t in range(start_idx + traj_length, end_idx + 1):
            seq_start = t - traj_length
            seq_end = t

            obs_window = obs[batch_idx, seq_start:seq_end, :, :]  # (traj_length, N, obs_dim)
            action_window = actions[
                batch_idx, seq_start:seq_end, ...
            ]  # (traj_length, N) 或 (traj_length, N, action_dim)

            # 处理动作序列：转换为4维并归一化
            if action_space_type == "discrete":
                # 离散动作：转换为4维 [T, N, 1] 并归一化
                if action_window.ndim == 2:  # (T, N)
                    action_window = action_window[..., None]  # (T, N, 1)
                action_window = action_window.astype(np.float32) / action_dim  # 归一化
            else:
                # 连续动作：确保是4维
                if action_window.ndim == 2:  # (T, N) -> (T, N, 1)
                    action_window = action_window[..., None]

            validation_data.append((obs_window, action_window))

    print(f"验证数据集构建完成，共 {len(validation_data)} 个序列")
    return validation_data


# -----------------------------------------------------------------------------
# 掩码策略实现
# -----------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(1, 2, 3))
def generate_ratio_mask(
    rng: chex.PRNGKey, batch_size: int, n_agents: int, mask_ratio: float
) -> chex.Array:
    """策略1：根据预定义比例随机mask"""
    total_elements = n_agents
    num_to_mask = int(total_elements * mask_ratio)
    num_to_mask = max(1, min(num_to_mask, n_agents - 1))  # 至少保留一个agent

    def make_single_mask(key):
        perm = jax.random.permutation(key, n_agents)
        mask = jnp.ones(n_agents)
        mask = mask.at[perm[:num_to_mask]].set(0.0)  # 0=mask, 1=keep
        return mask

    keys = jax.random.split(rng, batch_size)
    masks = jax.vmap(make_single_mask)(keys)
    return masks


@partial(jax.jit, static_argnums=(1, 2))
def generate_random_agents_mask(rng: chex.PRNGKey, batch_size: int, n_agents: int) -> chex.Array:
    """策略2：随机mask [1, n_agents-1]个智能体的所有timesteps"""
    rng, subkey = jax.random.split(rng)

    # 随机选择要mask的智能体数量
    num_to_mask = jax.random.randint(subkey, (batch_size,), minval=1, maxval=n_agents)

    def make_single_mask(bs_idx, num_mask, key):
        perm = jax.random.permutation(key, n_agents)
        mask = jnp.ones(n_agents)
        indices = jnp.arange(n_agents)
        select_mask = indices < num_mask
        masked_indices = jnp.where(select_mask, perm, n_agents)
        valid_indices = jnp.where(masked_indices < n_agents, masked_indices, 0)
        valid_mask = masked_indices < n_agents
        mask = mask.at[valid_indices].multiply(1.0 - valid_mask.astype(jnp.float32))
        return mask

    keys = jax.random.split(rng, batch_size)
    masks = jax.vmap(make_single_mask)(jnp.arange(batch_size), num_to_mask, keys)
    return masks


@partial(jax.jit, static_argnums=(1, 2))
def generate_single_agent_mask(rng: chex.PRNGKey, batch_size: int, n_agents: int) -> chex.Array:
    """策略3：仅保留单智能体，mask掉其他所有智能体"""
    agent_indices = jax.random.randint(rng, (batch_size,), minval=0, maxval=n_agents)
    masks = jnp.zeros((batch_size, n_agents))  # 默认全部mask
    masks = masks.at[jnp.arange(batch_size), agent_indices].set(1.0)  # 只保留选中的智能体
    return masks


# -----------------------------------------------------------------------------
# 训练相关函数
# -----------------------------------------------------------------------------


def create_learning_rate_fn(
    total_steps: int,
    warmup_steps: int,
    learning_rate: float,
) -> Callable[[int], jnp.ndarray]:
    """创建学习率调度函数"""
    warmup_steps = min(warmup_steps, total_steps - 1)
    decay_steps = max(total_steps - warmup_steps, 1)

    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=learning_rate, transition_steps=warmup_steps
    )
    decay_fn = optax.cosine_decay_schedule(
        init_value=learning_rate, decay_steps=decay_steps, alpha=0.1
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps])
    return schedule_fn


def create_train_state(
    rng: chex.PRNGKey,
    network_interface: UnifiedNetworkInterface,
    learning_rate_fn: Callable,
    grad_clip_norm: float = 1.0,
) -> TrainState:
    """创建训练状态"""
    # 创建dummy输入
    obs_dim = network_interface.network_params["obs_dim"]
    action_dim = network_interface.network_params["action_dim"]
    n_agents = network_interface.network_params["n_agents"]
    traj_length = network_interface.network_params.get(
        "traj_length", network_interface.network_params.get("K", 10)
    )

    dummy_obs = jnp.zeros((1, n_agents, traj_length, obs_dim))
    if network_interface.network_type == "s2mp":
        dummy_action = jnp.zeros((1, n_agents, traj_length))  # S2MP期望3维离散动作
    else:  # ma2e
        if network_interface.network_params.get("action_space_type", "discrete") == "discrete":
            # MA2E期望离散动作也是4维的
            dummy_action = jnp.zeros((1, n_agents, traj_length, 1))
        else:
            dummy_action = jnp.zeros((1, n_agents, traj_length, action_dim))
    dummy_mask = jnp.ones((1, n_agents))

    params = network_interface.init(rng, dummy_obs, dummy_action, dummy_mask)

    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate_fn, weight_decay=1e-5, b1=0.9, b2=0.95, eps=1e-8),
    )
    return TrainState.create(params=params, tx=tx, apply_fn=network_interface.apply_reconstruction)


def reconstruction_loss_fn(
    params: Any,
    batch: Tuple,
    network_interface: UnifiedNetworkInterface,
    mask_type: str,
    rng: chex.PRNGKey,
) -> float:
    """重构损失函数"""
    obs_seq, action_seq, agent_mask = batch

    # 应用网络获取重构结果
    if network_interface.network_type == "s2mp":
        pred_obs = network_interface.apply_reconstruction(params, obs_seq, action_seq, agent_mask)
        # 计算观测重构损失
        obs_error = jnp.square(pred_obs - obs_seq)

        # 应用mask：只对被mask的agent计算损失
        mask_for_loss = 1.0 - agent_mask  # 0=keep, 1=mask (B, N)
        obs_mask = mask_for_loss[:, :, None, None]  # (B, N, 1, 1) -> 广播到 (B, N, K, obs_dim)

        # 与MA2E保持完全一致的损失计算方式
        weighted_obs_error = obs_error * obs_mask
        obs_loss = jnp.mean(weighted_obs_error)  # 简单的均值，数值范围与MA2E一致

        return obs_loss

    else:  # ma2e
        pred_obs, pred_action = network_interface.apply_reconstruction(
            params, obs_seq, action_seq, agent_mask, deterministic=False
        )

        # 计算重构损失 - 对所有智能体所有时间步计算MSE损失，不区分是否被mask
        obs_error = jnp.square(pred_obs - obs_seq)
        # action_error = jnp.square(pred_action - action_seq)

        # 计算所有智能体所有时间步的平均MSE损失
        obs_loss = jnp.mean(obs_error)
        # action_loss = jnp.mean(action_error)

        return obs_loss


def train_step(
    state: TrainState,
    batch: Tuple,
    network_interface: UnifiedNetworkInterface,
    mask_type: str,
    rng: chex.PRNGKey,
) -> Tuple[TrainState, float]:
    """单步训练"""
    grad_fn = jax.value_and_grad(reconstruction_loss_fn)
    loss, grads = grad_fn(state.params, batch, network_interface, mask_type, rng)
    state = state.apply_gradients(grads=grads)
    return state, loss


# JIT编译训练步骤
@profile_time("训练步骤")
@partial(jax.jit, static_argnums=(2, 3))
def jit_train_step(state, batch, network_interface, mask_type, rng):
    return train_step(state, batch, network_interface, mask_type, rng)


# JIT编译的评估辅助函数
def create_jit_evaluate_reconstruction(network_interface: UnifiedNetworkInterface):
    """创建网络特定的JIT评估函数"""
    if network_interface.network_type == "s2mp":

        @profile_time(f"S2MP评估函数")
        @jax.jit
        def _jit_eval(params, obs_seq_batch, action_seq_batch, agent_mask):
            pred_obs = network_interface._jit_reconstruction(
                params, obs_seq_batch, action_seq_batch, agent_mask
            )
            return pred_obs, None
    else:  # ma2e

        @profile_time(f"MA2E评估函数")
        @jax.jit
        def _jit_eval(params, obs_seq_batch, action_seq_batch, agent_mask):
            pred_obs, pred_action = network_interface._jit_reconstruction(
                params, obs_seq_batch, action_seq_batch, agent_mask, True
            )
            return pred_obs, pred_action

    return _jit_eval


@partial(jax.jit, static_argnums=(0,))  # network_type是静态参数
def _jit_compute_reconstruction_errors(
    network_type: str,
    pred_obs: chex.Array,
    pred_action: Optional[chex.Array],
    obs_seq_batch: chex.Array,
    action_seq_batch: chex.Array,
    mask_for_loss: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """JIT编译的重构误差计算函数"""
    obs_error = jnp.mean(jnp.square(pred_obs - obs_seq_batch) * mask_for_loss[:, :, None, None])

    if network_type == "s2mp" or pred_action is None:
        action_error = jnp.array(0.0)  # S2MP没有动作重构
    else:
        action_error = jnp.mean(
            jnp.square(pred_action - action_seq_batch) * mask_for_loss[:, :, None, None]
        )

    return obs_error, action_error


@jax.jit
def _jit_compute_inference_error(
    pred_current_obs: chex.Array, current_gt_obs: chex.Array, other_agents_mask: chex.Array
) -> chex.Array:
    """JIT编译的推理误差计算函数"""
    return jnp.mean(jnp.square(pred_current_obs - current_gt_obs) * other_agents_mask[:, :, None])


def evaluate_model(
    network_interface: UnifiedNetworkInterface,
    params: Any,
    validation_data: List[Tuple[chex.Array, chex.Array]],
    n_agents: int,
    show_examples: bool = True,
    max_examples: int = 5,
    save_plots: bool = False,
    plots_dir: str = None,
) -> Dict[str, float]:
    """在验证集上评估模型"""
    print("开始验证集评估...")

    # 创建JIT编译的评估函数
    jit_evaluate_reconstruction = create_jit_evaluate_reconstruction(network_interface)

    all_obs_errors = []
    all_action_errors = []
    inference_obs_errors = []

    # 用于存储example展示的数据
    example_data = []
    example_count = 0

    for seq_idx, (obs_seq, action_seq) in enumerate(tqdm(validation_data, desc="验证中")):
        # obs_seq: (T, N, obs_dim), action_seq: (T, N) 或 (T, N, action_dim)
        T, N, obs_dim = obs_seq.shape

        # 跳过无效数据
        if N == 0 or T == 0:
            print(f"警告: 跳过无效序列 {seq_idx} - 形状: obs_seq={obs_seq.shape}")
            continue

        # 转换格式为 (B=1, N, T, obs_dim)
        obs_seq_batch = obs_seq.transpose(1, 0, 2)[None, ...]  # (1, N, T, obs_dim)

        # action_seq现在已经是4维的 (T, N, action_dim) 或 (T, N, 1)
        if network_interface.network_type == "s2mp" and action_seq.shape[-1] == 1:
            # S2MP需要3维离散动作，需要去掉最后一维并反归一化
            action_dim = network_interface.network_params.get("action_dim", 10)
            action_seq_batch = (action_seq[..., 0] * action_dim).astype(jnp.int32)  # (T, N)
            action_seq_batch = action_seq_batch.transpose(1, 0)[None, ...]  # (1, N, T)
        else:
            # MA2E或连续动作，保持4维
            action_seq_batch = action_seq.transpose(1, 0, 2)[None, ...]  # (1, N, T, action_dim)

        # 测试每个智能体的重构性能（策略3：仅保留单智能体）
        for agent_idx in range(n_agents):
            # 创建只保留当前agent的mask
            agent_mask = jnp.zeros((1, n_agents))
            agent_mask = agent_mask.at[:, agent_idx].set(1.0)

            # 获取重构结果（使用JIT加速）
            pred_obs, pred_action = jit_evaluate_reconstruction(
                params, obs_seq_batch, action_seq_batch, agent_mask
            )

            # 计算被mask的agents的重构误差（使用JIT加速）
            mask_for_loss = 1.0 - agent_mask  # 对其他agents计算误差
            obs_error, action_error = _jit_compute_reconstruction_errors(
                network_interface.network_type,
                pred_obs,
                pred_action,
                obs_seq_batch,
                action_seq_batch,
                mask_for_loss,
            )

            all_obs_errors.append(float(obs_error))
            if network_interface.network_type == "ma2e":
                all_action_errors.append(float(action_error))

            # 测试推理函数性能（使用JIT加速）
            single_agent_traj = obs_seq_batch[:, agent_idx, :, :]  # (1, T, obs_dim)
            current_gt_obs = obs_seq_batch[:, :, -1, :]  # (1, N, obs_dim) - 最后时刻的真实观测

            pred_current_obs = network_interface.apply_inference(
                params, single_agent_traj, current_gt_obs, agent_idx, deterministic=True
            )

            # 计算当前时刻其他智能体观测的预测误差（使用JIT加速）
            other_agents_mask = 1.0 - agent_mask  # 其他智能体的mask
            inference_error = _jit_compute_inference_error(
                pred_current_obs, current_gt_obs, other_agents_mask
            )
            inference_obs_errors.append(float(inference_error))

            # 收集example数据用于展示（每5个序列选一个，只展示agent_idx=0的情况）
            if (
                show_examples
                and example_count < max_examples
                and seq_idx % 5 == 0
                and agent_idx == 0
            ):
                try:
                    # 提取最后时刻的观测
                    true_obs_last = np.array(current_gt_obs[0])  # (N, obs_dim)

                    # 处理pred_current_obs - 确保正确的形状
                    pred_obs_raw = np.array(pred_current_obs)

                    # 如果pred_current_obs有batch维度，去掉它
                    if pred_obs_raw.ndim == 3 and pred_obs_raw.shape[0] == 1:
                        pred_obs_last = pred_obs_raw[0]  # (N, obs_dim)
                    elif pred_obs_raw.ndim == 2:
                        pred_obs_last = pred_obs_raw  # (N, obs_dim)
                    else:
                        print(f"警告: pred_current_obs形状异常: {pred_obs_raw.shape}")
                        continue

                    # 检查维度匹配
                    if (
                        pred_obs_last.shape[0] != true_obs_last.shape[0]
                        or pred_obs_last.shape[0] == 0
                    ):
                        print(
                            f"警告: 智能体维度不匹配 - 真实: {true_obs_last.shape}, 预测: {pred_obs_last.shape}"
                        )
                        continue

                    # 处理one-hot编码的去除
                    # vault数据包含前n_agents维的one-hot id，网络输出可能也包含

                    # 统一处理：如果观测维度大于预期，去掉前n_agents维
                    expected_clean_obs_dim = obs_dim - n_agents  # 去掉one-hot后的维度

                    # 处理真实观测
                    if true_obs_last.shape[1] > expected_clean_obs_dim:
                        true_obs_clean = true_obs_last[:, n_agents:]  # 去掉前n_agents维
                    else:
                        true_obs_clean = true_obs_last

                    # 处理预测观测
                    if pred_obs_last.shape[1] > expected_clean_obs_dim:
                        pred_obs_clean = pred_obs_last[:, n_agents:]  # 去掉前n_agents维
                    else:
                        pred_obs_clean = pred_obs_last

                    # 如果维度仍然不匹配，尝试调整
                    if pred_obs_clean.shape[1] != true_obs_clean.shape[1]:
                        min_dim = min(pred_obs_clean.shape[1], true_obs_clean.shape[1])
                        if min_dim > 0:
                            pred_obs_clean = pred_obs_clean[:, :min_dim]
                            true_obs_clean = true_obs_clean[:, :min_dim]
                        else:
                            continue

                    # 最终检查
                    if true_obs_clean.shape != pred_obs_clean.shape or true_obs_clean.shape[0] == 0:
                        print(
                            f"警告: 清理后形状仍不匹配 - 真实: {true_obs_clean.shape}, 预测: {pred_obs_clean.shape}"
                        )
                        continue

                    example_data.append(
                        {
                            "seq_idx": seq_idx,
                            "agent_idx": agent_idx,
                            "true_obs": true_obs_clean,
                            "pred_obs": pred_obs_clean,
                        }
                    )
                    example_count += 1

                except Exception as e:
                    print(f"收集Example数据时出错: {e}")
                    print(f"current_gt_obs形状: {current_gt_obs.shape}")
                    print(f"pred_current_obs形状: {pred_current_obs.shape}")
                    continue

    # 计算平均误差
    results = {
        "reconstruction_obs_error": np.mean(all_obs_errors),
        "inference_obs_error": np.mean(inference_obs_errors),
    }

    if all_action_errors:
        results["reconstruction_action_error"] = np.mean(all_action_errors)

    print(f"验证结果: {results}")

    # 展示example误差分析
    if show_examples and example_data:
        print(f"\n📊 展示 {len(example_data)} 个预测误差样例:")
        unit_type_bits = len(SMAX_UNIT_TYPE_NAMES)  # 6

        for i, example in enumerate(example_data):
            try:
                agent_errors = compute_observation_errors(
                    example["pred_obs"], example["true_obs"], n_agents, unit_type_bits
                )
                display_prediction_errors(
                    agent_errors, i, save_plots=save_plots, save_path=plots_dir
                )
            except Exception as e:
                print(f"处理Example {i + 1}时出错: {e}")
                print(f"真实观测形状: {example['true_obs'].shape}")
                print(f"预测观测形状: {example['pred_obs'].shape}")

    return results


# -----------------------------------------------------------------------------
# 主训练函数
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="统一多智能体掩码-重构预训练")

    # 数据参数
    parser.add_argument("--vault_dir", type=str, default="/home/mxfeng/aaai25_projects/Mava/vaults")
    parser.add_argument("--vault_name", type=str, default="mat")
    parser.add_argument("--vault_uid", type=str, default="20250628105914")

    # 网络参数
    parser.add_argument(
        "--network_type", type=str, default="ma2e", choices=["s2mp", "ma2e"], help="网络类型"
    )
    parser.add_argument("--embed_dim", type=int, default=64, help="嵌入维度")
    parser.add_argument("--num_heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--mlp_dim", type=int, default=256)
    parser.add_argument("--dropout_rate", type=float, default=0.0)

    # 训练参数
    parser.add_argument("--traj_length", type=int, default=5, help="轨迹长度")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--grad_clip_norm", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples_per_epoch", type=int, default=1000)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # 掩码策略参数
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="策略1的掩码比例")

    # 数据集参数
    parser.add_argument("--train_samples", type=int, default=10000, help="训练集样本数")
    parser.add_argument("--val_episodes", type=int, default=10, help="验证集episode数")
    parser.add_argument("--min_reward_threshold", type=float, default=2.0, help="验证集回报阈值")

    # 保存和加载
    parser.add_argument("--checkpoint_dir", type=str, default="./unified_checkpoints")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=1000)

    # 可视化参数
    parser.add_argument(
        "--show_examples", action="store_true", default=True, help="是否展示预测误差样例"
    )
    parser.add_argument("--max_examples", type=int, default=5, help="展示的最大样例数量")
    parser.add_argument("--no_examples", action="store_true", help="不展示预测误差样例")
    parser.add_argument("--save_plots", action="store_true", help="保存误差可视化图表到文件")
    parser.add_argument("--plots_dir", type=str, default="./error_plots", help="误差图表保存目录")

    args = parser.parse_args()

    # 处理可视化参数
    if args.no_examples:
        args.show_examples = False

    # 处理图表保存参数
    if args.save_plots:
        os.makedirs(args.plots_dir, exist_ok=True)
        print(f"误差图表将保存到: {args.plots_dir}")

    print("=" * 60)
    print("统一多智能体掩码-重构预训练")
    print("=" * 60)
    print(f"网络类型: {args.network_type}")
    print(f"轨迹长度: {args.traj_length}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"训练epochs: {args.epochs}")
    print("🚀 已启用JIT编译加速优化")
    print("   - 网络前向传播加速")
    print("   - 训练步骤加速")
    print("   - 推理函数加速")
    print("   - 评估函数加速")

    # 加载数据
    print("\n加载vault数据...")
    vlt = Vault(rel_dir=args.vault_dir, vault_name=args.vault_name, vault_uid=args.vault_uid)
    data = vlt.read()

    # 推断数据维度
    obs_shape = data.experience["observation"].shape
    _, seq_len, n_agents, obs_dim = obs_shape

    action_field = data.experience["action"]
    if action_field.ndim == 4:
        action_dim = action_field.shape[-1]
        action_space_type = "continuous"
    else:
        action_dim = int(np.array(action_field).max()) + 1
        action_space_type = "discrete"

    print(f"数据信息: obs_dim={obs_dim}, action_dim={action_dim}, n_agents={n_agents}")
    print(f"动作空间类型: {action_space_type}")

    # 创建trajectory buffer用于训练数据采样
    sample_sequence_length = args.traj_length + 1
    buffer = fbx.make_trajectory_buffer(
        sample_batch_size=1,  # 每次采样一个序列
        sample_sequence_length=sample_sequence_length,
        period=1,
        max_length_time_axis=1_000_000,
        min_length_time_axis=sample_sequence_length,
        add_batch_size=1,
    )

    # 构建训练数据集
    print("\n构建数据集...")
    training_data = build_training_dataset(
        data, buffer, args.train_samples, action_space_type, action_dim
    )
    validation_data = build_validation_dataset(
        data,
        args.min_reward_threshold,
        args.val_episodes,
        args.traj_length,
        action_space_type,
        action_dim,
    )

    if not validation_data:
        print("警告：验证数据集为空，将跳过验证步骤")

    # 创建网络接口
    network_params = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "n_agents": n_agents,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "mlp_dim": args.mlp_dim,
        "dropout_rate": args.dropout_rate,
    }

    if args.network_type == "s2mp":
        network_params["K"] = args.traj_length
        network_params["action_space_type"] = "discrete"  # S2MP主要支持离散动作
    else:  # ma2e
        network_params["traj_length"] = args.traj_length
        network_params["action_space_type"] = action_space_type
        network_params["positional_type"] = "both"

    network_interface = UnifiedNetworkInterface(args.network_type, network_params)

    # 创建学习率调度
    total_steps = args.epochs * args.samples_per_epoch * 3  # 3种掩码策略
    warmup_steps = int(total_steps * args.warmup_ratio)
    learning_rate_fn = create_learning_rate_fn(total_steps, warmup_steps, args.learning_rate)

    # 初始化训练状态
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, network_interface, learning_rate_fn, args.grad_clip_norm)

    # 加载检查点
    if args.load_checkpoint is not None:
        print(f"加载检查点: {args.load_checkpoint}")
        with open(args.load_checkpoint, "rb") as f:
            state = serialization.from_bytes(state, f.read())

    # 创建保存目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 训练循环
    print("\n开始训练...")
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        epoch_losses = {"ratio": [], "random": [], "single": []}

        # 随机打乱训练数据
        rng, shuffle_key = jax.random.split(rng)
        indices = jax.random.permutation(shuffle_key, len(training_data))[: args.samples_per_epoch]

        epoch_pbar = tqdm(indices, desc=f"Epoch {epoch + 1}")

        for sample_idx in epoch_pbar:
            obs_seq, action_seq = training_data[int(sample_idx)]
            # obs_seq: (T, N, obs_dim), action_seq: (T, N) 或 (T, N, action_dim)

            # 转换为网络期望的格式: (B, N, T, ...)
            obs_batch = obs_seq.transpose(1, 0, 2)[None, ...]  # (1, N, T, obs_dim)

            # action_seq现在已经是4维的 (T, N, action_dim) 或 (T, N, 1)
            if args.network_type == "s2mp" and action_seq.shape[-1] == 1:
                # S2MP需要3维离散动作，需要去掉最后一维并反归一化
                action_batch = (action_seq[..., 0] * action_dim).astype(jnp.int32)  # (T, N)
                action_batch = action_batch.transpose(1, 0)[None, ...]  # (1, N, T)
            else:
                # MA2E或连续动作，保持4维
                action_batch = action_seq.transpose(1, 0, 2)[None, ...]  # (1, N, T, action_dim)

            # 扩展到batch_size
            obs_batch = jnp.repeat(obs_batch, args.batch_size, axis=0)
            action_batch = jnp.repeat(action_batch, args.batch_size, axis=0)

            # 三种掩码策略的训练
            mask_strategies = [
                ("ratio", generate_ratio_mask),
                ("random", generate_random_agents_mask),
                ("single", generate_single_agent_mask),
            ]

            for mask_name, mask_fn in mask_strategies:
                global_step += 1
                rng, mask_key, train_key = jax.random.split(rng, 3)

                # 生成对应的mask
                if mask_name == "ratio":
                    agent_mask = mask_fn(mask_key, args.batch_size, n_agents, args.mask_ratio)
                else:
                    agent_mask = mask_fn(mask_key, args.batch_size, n_agents)

                # 训练步骤
                batch = (obs_batch, action_batch, agent_mask)
                state, loss = jit_train_step(state, batch, network_interface, mask_name, train_key)

                epoch_losses[mask_name].append(float(loss))

                # 更新进度条
                current_lr = learning_rate_fn(global_step - 1)
                avg_loss = np.mean(
                    [
                        np.mean(epoch_losses["ratio"]) if epoch_losses["ratio"] else 0,
                        np.mean(epoch_losses["random"]) if epoch_losses["random"] else 0,
                        np.mean(epoch_losses["single"]) if epoch_losses["single"] else 0,
                    ]
                )

                epoch_pbar.set_postfix(
                    {
                        "loss": f"{avg_loss:.6f}",
                        "lr": f"{current_lr:.2e}",
                        "best": f"{best_val_loss:.6f}",
                    }
                )

                # 详细日志
                if global_step % args.log_interval == 0:
                    print(f"\nStep {global_step}, 策略{mask_name}, Loss: {loss:.6f}")

                # 保存检查点
                if global_step % args.save_interval == 0:
                    checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{global_step}")
                    with open(checkpoint_path, "wb") as f:
                        f.write(serialization.to_bytes(state))
                    print(f"保存检查点: {checkpoint_path}")

        # Epoch结束后的评估
        epoch_avg_losses = {
            mask_name: np.mean(losses) for mask_name, losses in epoch_losses.items()
        }
        print(f"\nEpoch {epoch + 1} 平均损失:")
        for mask_name, avg_loss in epoch_avg_losses.items():
            print(f"  {mask_name}: {avg_loss:.6f}")

        # 验证集评估
        if validation_data:
            val_results = evaluate_model(
                network_interface,
                state.params,
                validation_data,
                n_agents,
                show_examples=args.show_examples,
                max_examples=args.max_examples,
                save_plots=args.save_plots,
                plots_dir=args.plots_dir,
            )
            val_loss = val_results["reconstruction_obs_error"]

            print(f"\nEpoch {epoch + 1} 验证结果:")
            for metric, value in val_results.items():
                print(f"  {metric}: {value:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 保存最佳模型
                best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_model")
                with open(best_checkpoint_path, "wb") as f:
                    f.write(serialization.to_bytes(state))
                print(f"保存最佳模型: {best_checkpoint_path}")

    print(f"\n训练完成！最佳验证损失: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
