#!/usr/bin/env python3
"""
Memory-optimized S2MP pretraining launcher script
优化内存使用的S2MP预训练启动脚本
"""

import argparse
import os
import subprocess
import sys


def get_gpu_memory_info():
    """获取GPU内存信息"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpu_info = []
            for line in lines:
                total, free = map(int, line.split(", "))
                gpu_info.append((total, free))
            return gpu_info
    except:
        pass
    return None


def recommend_config(gpu_memory_mb, num_gpus):
    """根据GPU内存推荐配置"""
    # 基础配置（适用于8GB显存）
    if gpu_memory_mb < 6000:  # < 6GB
        return {"batch_size": 8, "K": 5, "num_random_masks": 1, "gradient_accumulation_steps": 8}
    elif gpu_memory_mb < 12000:  # 6-12GB
        return {"batch_size": 16, "K": 8, "num_random_masks": 2, "gradient_accumulation_steps": 4}
    elif gpu_memory_mb < 24000:  # 12-24GB
        return {"batch_size": 32, "K": 10, "num_random_masks": 3, "gradient_accumulation_steps": 2}
    else:  # > 24GB
        return {"batch_size": 64, "K": 15, "num_random_masks": 5, "gradient_accumulation_steps": 1}


def main():
    parser = argparse.ArgumentParser(description="Memory-optimized S2MP training launcher")
    parser.add_argument(
        "--auto_config",
        action="store_true",
        default=True,
        help="Automatically configure based on available GPU memory",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show the command that would be executed without running it",
    )

    # 允许手动覆盖配置
    parser.add_argument("--batch_size", type=int, help="Override batch size per GPU")
    parser.add_argument("--K", type=int, help="Override sequence length")
    parser.add_argument("--num_random_masks", type=int, help="Override number of random masks")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, help="Override gradient accumulation steps"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=500, help="Steps per epoch")

    args = parser.parse_args()

    # 检查GPU信息
    gpu_info = get_gpu_memory_info()
    num_gpus = len(gpu_info) if gpu_info else 1

    print(f"检测到 {num_gpus} 个GPU")
    if gpu_info:
        for i, (total, free) in enumerate(gpu_info):
            print(f"  GPU {i}: {total}MB total, {free}MB free")
        min_memory = min(total for total, free in gpu_info)
    else:
        print("无法检测GPU内存，使用保守配置")
        min_memory = 8000  # 假设8GB

    # 获取推荐配置
    if args.auto_config:
        config = recommend_config(min_memory, num_gpus)
        print(f"\n基于 {min_memory}MB 显存的推荐配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        config = {}

    # 手动覆盖优先
    for key in ["batch_size", "K", "num_random_masks", "gradient_accumulation_steps"]:
        if hasattr(args, key) and getattr(args, key) is not None:
            config[key] = getattr(args, key)

    # 构建命令
    cmd = [
        sys.executable,
        "mava/pretrain/s2mp_pretrain_multi_gpu.py",
        "--memory_efficient",
        "--epochs",
        str(args.epochs),
        "--steps_per_epoch",
        str(args.steps_per_epoch),
        "--log_interval",
        "50",
        "--save_interval",
        "500",
    ]

    # 添加配置参数
    for key, value in config.items():
        cmd.extend([f"--{key}", str(value)])

    print(f"\n将要执行的命令:")
    print(" ".join(cmd))

    if args.dry_run:
        print("\n[DRY RUN] 命令未执行")
        return

    # 设置环境变量以优化JAX内存使用
    env = os.environ.copy()
    env.update(
        {
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85",  # 限制JAX内存使用
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "JAX_ENABLE_X64": "false",  # 使用float32节省内存
        }
    )

    print("\n开始训练...")
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"训练失败，退出码: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(130)


if __name__ == "__main__":
    main()
