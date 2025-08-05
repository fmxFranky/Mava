"""Masked AutoEncoder-like network for multi-agent sequential observation reconstruction.

This module implements the S2MPNetwork (Sequence-to-Masked-Pretraining) which takes as input
for each agent a sequence of K observations together with the previous K actions. A binary
mask at the agent level selects which agents' trajectories are kept and which are replaced
with a learnable mask token. The model then tries to reconstruct the observations of all
agents via a two-stage Transformer pipeline similar to the original MAE paper.

Processing pipeline (symbol conventions follow the user description):

1. Encode observations and actions independently into the same embedding space.
2. Arrange the 2*n*K resulting tokens in the order:
   â_t-K-1^1, ô_t-K^1, …, â_t-1^n, ô_t^n (action, obs interleaved, agent-major ordering).
3. Replace all tokens of agents with M_i = 0 by a shared learnable mask token.
4. Add positional embedding obtained by concatenating agent id embedding (d/2) and
   temporal id embedding (d/2).
5. Append L shared learnable latent tokens and pass everything through a Transformer
   encoder, keeping only the last L output tokens.
6. Concatenate those L tokens with n*K newly introduced learnable decoder tokens (after
   adding again the same positional embeddings) and process them with a second
   Transformer encoder.
7. Use a small MLP head to reconstruct the original observations from the last n*K
   decoder tokens.

The network is implemented with Flax/JAX and follows the coding style of the other
networks in mava/networks.
"""

from typing import List, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import orthogonal

from mava.utils.network_utils import _CONTINUOUS, _DISCRETE


class TransformerEncoderBlock(nn.Module):
    """A standard Transformer encoder block (Pre-LN)."""

    embed_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.0

    def setup(self) -> None:
        self.ln1 = nn.LayerNorm()
        self.attn = nn.SelfAttention(num_heads=self.num_heads, dropout_rate=self.dropout_rate)
        self.ln2 = nn.LayerNorm()
        self.mlp = nn.Sequential(
            [
                nn.Dense(self.mlp_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                nn.Dense(self.embed_dim, kernel_init=orthogonal(0.01)),
            ]
        )

    def __call__(self, x: chex.Array) -> chex.Array:  # (B, S, E)
        y = self.attn(self.ln1(x))
        x = x + y
        y = self.mlp(self.ln2(x))
        return x + y


class TransformerEncoder(nn.Module):
    """Stack of TransformerEncoderBlocks with final LayerNorm."""

    num_layers: int
    embed_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.0

    def setup(self) -> None:
        self.blocks: List[TransformerEncoderBlock] = [
            TransformerEncoderBlock(
                self.embed_dim,
                self.num_heads,
                self.mlp_dim,
                self.dropout_rate,
                name=f"block_{i}",
            )
            for i in range(self.num_layers)
        ]
        self.blocks = nn.Sequential(self.blocks)
        self.pre_ln = nn.LayerNorm()
        self.post_ln = nn.LayerNorm()

    def __call__(self, x: chex.Array) -> chex.Array:  # (B, S, E)
        return self.post_ln(self.blocks(x))


class S2MPNetwork(nn.Module):
    """Masked auto-encoder network for multi-agent sequences."""

    obs_dim: int  # dimensionality of one observation vector
    action_dim: int  # dimensionality of one action vector / number of discrete actions
    n_agents: int  # number of agents N
    K: int  # number of (observation, action) pairs (timesteps)
    embed_dim: int = 64  # model dimension E (must be divisible by 2)
    num_heads: int = 4
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    latent_tokens: int = 4  # L
    mlp_dim: int = 128  # inner FF dimension in transformer
    dropout_rate: float = 0.0  # dropout rate for transformer blocks
    action_space_type: str = _DISCRETE  # "discrete" | "continuous"

    def setup(self) -> None:
        if self.embed_dim % 2 != 0:
            raise ValueError("embed_dim 必须能被 2 整除，以便拼接位置编码")

        # Encoders for obs 和 action - 启用LayerNorm稳定训练
        self.obs_encoder = nn.Sequential(
            [
                nn.LayerNorm(),  # 输入标准化
                nn.Dense(self.embed_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                nn.LayerNorm(),  # 输出标准化
            ]
        )

        # 对连续/离散动作分别处理
        use_bias = self.action_space_type == _CONTINUOUS
        self.action_encoder = nn.Sequential(
            [
                nn.LayerNorm(),  # 输入标准化
                nn.Dense(
                    self.embed_dim,
                    use_bias=use_bias,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                ),
                nn.gelu,
                nn.LayerNorm(),  # 输出标准化
            ]
        )

        # Learnable special tokens
        self.mask_token = self.param("mask_token", nn.initializers.zeros, (self.embed_dim,))
        self.encoder_latent_token = self.param(
            "encoder_latent_token", nn.initializers.zeros, (self.embed_dim,)
        )
        self.decoder_token = self.param("decoder_token", nn.initializers.zeros, (self.embed_dim,))

        # Positional embeddings (agent id & temporal id)
        half_dim = self.embed_dim // 2
        self.agent_pos_embed = self.param(
            "agent_pos_embed", nn.initializers.normal(stddev=0.02), (self.n_agents, half_dim)
        )
        self.time_pos_embed = self.param(
            "time_pos_embed", nn.initializers.normal(stddev=0.02), (self.K, half_dim)
        )

        # 位置编码后的LayerNorm
        self.pos_embed_ln = nn.LayerNorm()

        # 编码器和解码器输入前的LayerNorm
        self.encoder_input_ln = nn.LayerNorm()
        self.decoder_input_ln = nn.LayerNorm()

        # Transformer stacks
        self.encoder = TransformerEncoder(
            self.num_encoder_layers,
            self.embed_dim,
            self.num_heads,
            self.mlp_dim,
            self.dropout_rate,
            name="encoder",
        )
        self.decoder = TransformerEncoder(
            self.num_decoder_layers,
            self.embed_dim,
            self.num_heads,
            self.mlp_dim,
            self.dropout_rate,
            name="decoder",
        )

        # Reconstruction head (shared for all tokens) - 添加LayerNorm稳定重建
        self.reconstruction_head = nn.Sequential(
            [
                nn.LayerNorm(),  # 重建头输入标准化
                nn.Dense(self.mlp_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                nn.LayerNorm(),  # 中间层标准化
                nn.Dense(self.obs_dim, kernel_init=orthogonal(0.01)),
            ]
        )

    # Utility -----------------------------------------------------------------
    def _build_positional_embedding(self, N: int, K: int, for_obs_only: bool = False) -> chex.Array:
        """Pre-compute positional embedding table for either all 2*K tokens or obs-only tokens."""
        if for_obs_only:
            # 只需要 n*K 的观测 token
            seq_len = N * K
            agent_idx = jnp.repeat(jnp.arange(N), K)
            time_idx = jnp.tile(jnp.arange(K), N)
        else:
            # 需要 n*2K 的 action+obs token
            seq_len = N * K * 2
            agent_idx = jnp.repeat(jnp.arange(N), K * 2)
            time_pattern = jnp.repeat(jnp.arange(K), 2)  # (2K, ) -> action, obs, action, obs…
            time_idx = jnp.tile(time_pattern, N)

        # 处理时间索引超出预定义嵌入范围的情况
        max_time_embed = self.time_pos_embed.shape[0]
        time_idx = jnp.clip(time_idx, 0, max_time_embed - 1)

        agent_emb = self.agent_pos_embed[agent_idx]  # (seq_len, E/2)
        time_emb = self.time_pos_embed[time_idx]  # (seq_len, E/2)
        pos_emb = jnp.concatenate([agent_emb, time_emb], axis=-1)  # (seq_len, E)
        return pos_emb  # (seq_len, E)

    # -------------------------------------------------------------------------

    def __call__(
        self,
        obs_seq: chex.Array,  # (B, N, K, obs_dim)
        action_seq: chex.Array,  # (B, N, K) for discrete or (B, N, K, action_dim) for continuous
        agent_mask: chex.Array,  # (B, N) 1=keep, 0=mask
        rngs: dict = None,  # 兼容统一接口，可选参数
    ) -> chex.Array:  # -> reconstructed obs (B, N, K, obs_dim)
        # print(f"[s2mp]obs_seq: {obs_seq.shape}")
        # print(f"[s2mp]action_seq: {action_seq.shape}")
        # print(f"[s2mp]agent_mask: {agent_mask.shape}")
        B, N, K = obs_seq.shape[:3]  # 动态获取序列长度

        # 1) Encode observations & actions
        obs_emb = self.obs_encoder(obs_seq)  # (B, N, K, E)

        # 处理动作序列：支持3维离散动作 (B, N, K) 或4维连续动作 (B, N, K, action_dim)
        if action_seq.ndim == 3:
            # 3维输入，假设是离散动作索引 (B, N, K)
            if self.action_space_type == _DISCRETE:
                action_onehot = jax.nn.one_hot(action_seq.astype(jnp.int32), self.action_dim)
                action_flat = action_onehot  # (B, N, K, action_dim)
            else:
                # 连续动作但是3维，需要扩展维度
                action_flat = action_seq[..., None]  # (B, N, K, 1)
        elif action_seq.ndim == 4:
            # 4维输入 (B, N, K, action_dim)
            if self.action_space_type == _DISCRETE and action_seq.dtype in [jnp.int32, jnp.int64]:
                action_onehot = jax.nn.one_hot(action_seq.astype(jnp.int32), self.action_dim)
                action_flat = action_onehot
            else:
                # 已经是连续向量或离散 one-hot
                action_flat = action_seq
        else:
            raise ValueError(f"不支持的动作序列维度: {action_seq.ndim}，期望3维或4维")

        action_emb = self.action_encoder(action_flat)  # (B, N, K, E)

        # 2) Interleave & flatten tokens: a0, o1, a1, o2 … per agent
        tokens_pair = jnp.stack([action_emb, obs_emb], axis=3)  # (B, N, K, 2, E)
        tokens_agent = tokens_pair.reshape(B, N, K * 2, self.embed_dim)  # (B, N, 2K, E)
        tokens_flat = tokens_agent.reshape(B, N * K * 2, self.embed_dim)  # (B, 2KN, E)

        # 3) Apply agent-level mask
        keep_mask = agent_mask  # (B, N)
        keep_mask = jnp.repeat(keep_mask, K * 2, axis=1)  # (B, 2KN)
        mask_token = self.mask_token  # (E,)
        mask_token = jnp.broadcast_to(mask_token, (B, tokens_flat.shape[1], self.embed_dim))
        tokens_masked = jnp.where(keep_mask[..., None] == 1, tokens_flat, mask_token)

        # 4) Add positional embeddings
        pos_emb_full = self._build_positional_embedding(N, K, for_obs_only=False)  # (2KN, E)
        pos_emb_full = jnp.broadcast_to(pos_emb_full, (B, *pos_emb_full.shape))
        tokens_masked = tokens_masked + pos_emb_full

        # 位置编码后应用LayerNorm稳定训练
        tokens_masked = self.pos_embed_ln(tokens_masked)

        # 5) Append L latent tokens
        latent_token = self.encoder_latent_token  # (E,)
        latent_token = jnp.broadcast_to(latent_token, (B, self.latent_tokens, self.embed_dim))
        enc_input = jnp.concatenate([tokens_masked, latent_token], axis=1)  # (B, 2KN+L, E)

        # 编码器输入前应用LayerNorm
        enc_input = self.encoder_input_ln(enc_input)

        # 6) Transformer encoder
        enc_output = self.encoder(enc_input)  # (B, 2KN+L, E)
        enc_latents = enc_output[:, -self.latent_tokens :, :]  # (B, L, E)

        # 7) Prepare decoder tokens for observations
        obs_seq_len = N * K
        dec_token_embed = self.decoder_token  # (E,)
        dec_tokens = jnp.broadcast_to(
            dec_token_embed, (B, obs_seq_len, self.embed_dim)
        )  # (B, NK, E)

        pos_emb_obs = self._build_positional_embedding(N, K, for_obs_only=True)  # (NK, E)
        pos_emb_obs = jnp.broadcast_to(pos_emb_obs, (B,) + pos_emb_obs.shape)
        dec_tokens = dec_tokens + pos_emb_obs

        # Concatenate encoder latents and decoder tokens
        dec_input = jnp.concatenate([enc_latents, dec_tokens], axis=1)  # (B, L+NK, E)

        # 解码器输入前应用LayerNorm
        dec_input = self.decoder_input_ln(dec_input)

        dec_output = self.decoder(dec_input)  # (B, L+NK, E)

        dec_obs_tokens = dec_output[:, -obs_seq_len:, :]  # (B, NK, E)

        # 8) Reconstruction head
        rec_obs_flat = self.reconstruction_head(dec_obs_tokens)  # (B, NK, obs_dim)
        rec_obs = rec_obs_flat.reshape(B, N, K, self.obs_dim)  # (B, N, K, obs_dim)

        return rec_obs
