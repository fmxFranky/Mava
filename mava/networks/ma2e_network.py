"""Multi-Agent Masked Auto-Encoder (MA2E) Network Implementation."""

from typing import Optional, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp


class MA2EPositionalEncoding(nn.Module):
    """位置编码模块，支持agent和time维度。"""

    max_seq_len: int
    d_embed: int
    positional_type: str = "both"  # "agent", "time", "both"

    def setup(self) -> None:
        # 初始化位置编码参数
        self.pe_agent = self.param(
            "pe_agent", lambda rng, shape: jnp.zeros(shape), (self.max_seq_len, self.d_embed)
        )
        self.pe_time = self.param(
            "pe_time", lambda rng, shape: jnp.zeros(shape), (self.max_seq_len, self.d_embed)
        )

    def generate_encoding(self, positions: chex.Array, pe_type: str) -> chex.Array:
        """生成位置编码"""
        d_embed = self.d_embed
        position = positions.astype(jnp.float32)
        seq_len = len(positions)

        div_term = jnp.exp(jnp.arange(0, d_embed, 2) * (-jnp.log(10000.0) / d_embed))

        pe = jnp.zeros((seq_len, d_embed))
        pe = pe.at[:, 0::2].set(jnp.sin(position[:, None] * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position[:, None] * div_term))

        return pe

    def __call__(
        self, x: chex.Array, agent_positions: chex.Array, time_positions: chex.Array
    ) -> chex.Array:
        """
        Args:
            x: (B, S, E) 输入序列
            agent_positions: (S,) agent位置索引
            time_positions: (S,) time位置索引
        """
        batch_size, seq_len, embed_dim = x.shape

        # 截取位置数组到实际序列长度
        agent_positions = agent_positions[:seq_len]
        time_positions = time_positions[:seq_len]

        if self.positional_type == "agent":
            pe = self.generate_encoding(agent_positions, "agent")
        elif self.positional_type == "time":
            pe = self.generate_encoding(time_positions, "time")
        elif self.positional_type == "both":
            pe_agent = self.generate_encoding(agent_positions, "agent")
            pe_time = self.generate_encoding(time_positions, "time")
            # 组合agent和time编码
            pe = pe_agent + pe_time
        else:
            raise ValueError(f"Unknown positional_type: {self.positional_type}")

        # 广播到batch维度 - pe应该已经是正确的形状 (seq_len, embed_dim)
        pe = jnp.broadcast_to(pe[None, :, :], (batch_size, seq_len, embed_dim))

        return x + pe


class MA2ETransformerBlock(nn.Module):
    """Transformer encoder/decoder block"""

    embed_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.0
    is_decoder: bool = False

    def setup(self) -> None:
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.embed_dim, dropout_rate=self.dropout_rate
        )

        if self.is_decoder:
            self.cross_attention = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.embed_dim,
                dropout_rate=self.dropout_rate,
            )

        # MLP layers (manual construction to handle deterministic parameter)
        self.mlp_dense1 = nn.Dense(self.mlp_dim)
        self.mlp_dense2 = nn.Dense(self.embed_dim)
        if self.dropout_rate > 0.0:
            self.mlp_dropout1 = nn.Dropout(self.dropout_rate)
            self.mlp_dropout2 = nn.Dropout(self.dropout_rate)

        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        if self.is_decoder:
            self.norm3 = nn.LayerNorm()

    def __call__(
        self,
        x: chex.Array,
        encoder_output: Optional[chex.Array] = None,
        deterministic: bool = False,
    ) -> chex.Array:
        # Self-attention
        norm_x = self.norm1(x)
        attn_output = self.attention(norm_x, norm_x, deterministic=deterministic)
        x = x + attn_output

        # Cross-attention (decoder only)
        if self.is_decoder and encoder_output is not None:
            norm_x = self.norm2(x)
            cross_attn_output = self.cross_attention(
                norm_x, encoder_output, deterministic=deterministic
            )
            x = x + cross_attn_output
            norm_layer = self.norm3
        else:
            norm_layer = self.norm2

        # MLP
        norm_x = norm_layer(x)
        mlp_output = self.mlp_dense1(norm_x)
        mlp_output = nn.gelu(mlp_output)
        if self.dropout_rate > 0.0:
            mlp_output = self.mlp_dropout1(mlp_output, deterministic=deterministic)
        mlp_output = self.mlp_dense2(mlp_output)
        if self.dropout_rate > 0.0:
            mlp_output = self.mlp_dropout2(mlp_output, deterministic=deterministic)
        x = x + mlp_output

        return x


class MA2ENetwork(nn.Module):
    """Multi-Agent Masked Auto-Encoder Network"""

    obs_dim: int
    action_dim: int
    n_agents: int
    traj_length: int  # K
    embed_dim: int = 128
    num_heads: int = 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    mlp_dim: int = 512
    dropout_rate: float = 0.1
    positional_type: str = "both"  # "agent", "time", "both"
    action_space_type: str = "discrete"  # "discrete" or "continuous"

    def setup(self) -> None:
        # Embedding layers
        self.obs_embed = nn.Dense(self.embed_dim)
        self.action_embed = nn.Dense(self.embed_dim)

        # 位置编码 - 使用更大的缓冲区以处理可能的序列长度变化
        max_seq_len = self.n_agents * self.traj_length * 4  # 增加缓冲区，确保能处理更长序列
        self.pos_encoding = MA2EPositionalEncoding(
            max_seq_len=max_seq_len, d_embed=self.embed_dim, positional_type=self.positional_type
        )

        # Transformer encoder
        self.encoder_layers = [
            MA2ETransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                is_decoder=False,
            )
            for _ in range(self.num_encoder_layers)
        ]

        # Transformer decoder
        self.decoder_layers = [
            MA2ETransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                is_decoder=True,
            )
            for _ in range(self.num_decoder_layers)
        ]

        # Output heads
        self.obs_output_head = nn.Dense(self.obs_dim)
        self.action_output_head = nn.Dense(
            self.action_dim if self.action_space_type == "continuous" else self.action_dim
        )

        # Layer norm
        self.final_norm = nn.LayerNorm()

    def create_agent_time_positions(
        self, n_agents: int, traj_len: int
    ) -> Tuple[chex.Array, chex.Array]:
        """创建agent和time位置索引 - JAX向量化版本"""
        # 创建时间和智能体的网格
        t_indices = jnp.arange(traj_len)
        agent_indices = jnp.arange(n_agents)

        # 使用meshgrid创建所有组合
        t_grid, agent_grid = jnp.meshgrid(t_indices, agent_indices, indexing="ij")  # (T, N)

        # 重复每个位置两次（obs和action）
        t_expanded = jnp.repeat(t_grid.flatten(), 2)  # (T*N*2,)
        agent_expanded = jnp.repeat(agent_grid.flatten(), 2)  # (T*N*2,)

        return agent_expanded, t_expanded

    def apply_agent_mask(
        self, embedded_seq: chex.Array, agent_mask: chex.Array, n_agents: int, traj_len: int
    ) -> chex.Array:
        """应用agent级别的mask - JAX向量化版本"""
        # agent_mask: (B, N) 1=keep, 0=mask
        # embedded_seq: (B, N*K*2, E)

        batch_size, seq_len, embed_dim = embedded_seq.shape

        # 为每个agent创建在序列中的所有位置索引
        # 每个时间步有N*2个位置（N个obs + N个action）
        # agent i 在时间步 t 的位置：obs: t*N*2 + i*2, action: t*N*2 + i*2 + 1

        # 创建agent索引矩阵 (traj_len, n_agents, 2) -> obs和action的位置
        t_indices = jnp.arange(traj_len)[:, None, None]  # (T, 1, 1)
        agent_indices = jnp.arange(n_agents)[None, :, None]  # (1, N, 1)
        token_indices = jnp.array([0, 1])[None, None, :]  # (1, 1, 2) for obs and action

        # 计算每个agent在每个时间步的obs和action位置
        positions = t_indices * n_agents * 2 + agent_indices * 2 + token_indices  # (T, N, 2)
        positions = positions.reshape(-1)  # (T*N*2,)

        # 创建agent重复模式：每个agent的mask值应用到它的所有位置
        agent_mask_expanded = jnp.repeat(
            jnp.repeat(agent_mask, traj_len, axis=1),  # (B, N*T)
            2,
            axis=1,  # (B, N*T*2) - 每个agent的每个时间步有obs和action两个token
        )

        # 应用mask：1=keep原值，0=置零（masked）
        masked_seq = embedded_seq * agent_mask_expanded[:, :, None]

        return masked_seq

    def encode(self, embedded_seq: chex.Array, deterministic: bool = False) -> chex.Array:
        """Encoder forward pass"""
        x = embedded_seq

        for layer in self.encoder_layers:
            x = layer(x, deterministic=deterministic)

        return x

    def decode(
        self, decoder_input: chex.Array, encoder_output: chex.Array, deterministic: bool = False
    ) -> chex.Array:
        """Decoder forward pass"""
        x = decoder_input

        for layer in self.decoder_layers:
            x = layer(x, encoder_output=encoder_output, deterministic=deterministic)

        return x

    def __call__(
        self,
        obs_seq: chex.Array,
        action_seq: chex.Array,
        agent_mask: chex.Array,
        deterministic: bool = False,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Args:
            obs_seq: (B, N, K, obs_dim) 观测序列
            action_seq: (B, N, K, action_dim) 动作序列
            agent_mask: (B, N) agent mask, 1=keep, 0=mask
            deterministic: 是否为确定性模式（与training相反）

        Returns:
            reconstructed_obs: (B, N, K, obs_dim)
            reconstructed_action: (B, N, K, action_dim)
        """
        batch_size, n_agents, traj_len, _ = obs_seq.shape

        # 重组数据：交替排列obs和action - JAX向量化版本
        # 目标格式: (B, N*K*2, embed_dim) 其中每个时间步是 [obs1, act1, obs2, act2, ...]

        # 处理观测嵌入
        obs_seq_reshaped = obs_seq.reshape(batch_size, n_agents * traj_len, -1)  # (B, N*K, obs_dim)
        obs_embedded = self.obs_embed(obs_seq_reshaped)  # (B, N*K, embed_dim)

        # 处理动作嵌入
        if self.action_space_type == "discrete":
            # 根据标准化流程，离散动作已经归一化：action_value / action_dim
            # 需要反归一化为整数索引用于one-hot编码
            action_input = action_seq[:, :, :, 0]  # (B, N, K) 取出归一化的动作值
            action_indices = (action_input * self.action_dim).astype(jnp.int32)
            action_indices = jnp.clip(action_indices, 0, self.action_dim - 1)
            action_indices_reshaped = action_indices.reshape(
                batch_size, n_agents * traj_len
            )  # (B, N*K)
            action_onehot = jax.nn.one_hot(
                action_indices_reshaped, self.action_dim
            )  # (B, N*K, action_dim)
            action_embedded = self.action_embed(action_onehot)  # (B, N*K, embed_dim)
        else:
            # 连续动作直接嵌入
            action_seq_reshaped = action_seq.reshape(
                batch_size, n_agents * traj_len, -1
            )  # (B, N*K, action_dim)
            action_embedded = self.action_embed(action_seq_reshaped)  # (B, N*K, embed_dim)

        # 交替排列obs和action：使用JAX向量化操作
        # 创建交替索引模式
        obs_embedded_expanded = obs_embedded.reshape(
            batch_size, traj_len, n_agents, self.embed_dim
        )  # (B, K, N, E)
        action_embedded_expanded = action_embedded.reshape(
            batch_size, traj_len, n_agents, self.embed_dim
        )  # (B, K, N, E)

        # 堆叠obs和action，然后重新排列
        stacked = jnp.stack(
            [obs_embedded_expanded, action_embedded_expanded], axis=3
        )  # (B, K, N, 2, E)
        embedded_seq = stacked.reshape(
            batch_size, traj_len * n_agents * 2, self.embed_dim
        )  # (B, K*N*2, E)

        # 添加位置编码
        agent_positions, time_positions = self.create_agent_time_positions(n_agents, traj_len)
        embedded_seq = self.pos_encoding(embedded_seq, agent_positions, time_positions)

        # 应用agent mask
        masked_embedded_seq = self.apply_agent_mask(embedded_seq, agent_mask, n_agents, traj_len)

        # Encoder
        encoder_output = self.encode(masked_embedded_seq, deterministic=deterministic)

        # Decoder (使用原始embedded_seq作为target)
        decoder_output = self.decode(embedded_seq, encoder_output, deterministic=deterministic)
        decoder_output = self.final_norm(decoder_output)

        # 分离obs和action的输出并重组为 (B, N, K, dim) 格式 - JAX向量化版本
        # decoder_output: (B, K*N*2, embed_dim)

        # 重新排列为 (B, K, N, 2, embed_dim)
        decoder_reshaped = decoder_output.reshape(batch_size, traj_len, n_agents, 2, self.embed_dim)

        # 分离obs和action
        obs_output = decoder_reshaped[:, :, :, 0, :]  # (B, K, N, embed_dim)
        action_output = decoder_reshaped[:, :, :, 1, :]  # (B, K, N, embed_dim)

        # 应用输出头
        obs_output_flat = obs_output.reshape(batch_size * traj_len * n_agents, self.embed_dim)
        action_output_flat = action_output.reshape(batch_size * traj_len * n_agents, self.embed_dim)

        reconstructed_obs_flat = self.obs_output_head(obs_output_flat)  # (B*K*N, obs_dim)
        reconstructed_action_flat = self.action_output_head(
            action_output_flat
        )  # (B*K*N, action_dim)

        # 重新排列为最终格式
        reconstructed_obs = reconstructed_obs_flat.reshape(
            batch_size, traj_len, n_agents, self.obs_dim
        )
        reconstructed_obs = reconstructed_obs.transpose(0, 2, 1, 3)  # (B, N, K, obs_dim)

        reconstructed_action = reconstructed_action_flat.reshape(
            batch_size, traj_len, n_agents, self.action_dim
        )
        reconstructed_action = reconstructed_action.transpose(0, 2, 1, 3)  # (B, N, K, action_dim)

        return reconstructed_obs, reconstructed_action
