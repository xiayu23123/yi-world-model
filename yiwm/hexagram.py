"""卦象推断层: 6 爻强度 -> 64 卦 logits.

Structural priors baked in:
  * multi-relation graph convolution (R-GCN style) over four hexagram
    relations, each with its own weight matrix and a learnable softmax gate:
        ham  一爻之变  (Hamming-1, 6-regular graph)
        cuo  错卦      (every line inverted; involution)
        zong 综卦      (hexagram turned upside down; involution, 8 fixed points)
        hu   互卦      (nuclear hexagram; not an involution)
  * lower / upper 经卦 (八卦) embeddings added to every hexagram embedding.
  * a direct polarity term: agreement between sign(yao) and each hexagram's
    bit pattern. Gives the model usable structure from step 0.

All fixed tensors are registered buffers.
"""

import torch
import torch.nn as nn

from .constants import (
    BINARY_HEX, HEX_ADJ, HEX_CUO, HEX_HU, HEX_LOWER_TRIGRAM, HEX_UPPER_TRIGRAM,
    HEX_ZONG,
)


class HexagramInference(nn.Module):
    RELATIONS = ("ham", "cuo", "zong", "hu")

    def __init__(self, embed_dim: int = 64, gcn_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        n_rel = len(self.RELATIONS)

        self.hex_embed = nn.Embedding(64, embed_dim)
        self.tri_embed = nn.Embedding(8, embed_dim)
        self.query = nn.Sequential(
            nn.Linear(6, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.scale = embed_dim ** -0.5

        # one self-transform + one transform per relation, per layer
        self.self_lin = nn.ModuleList(
            nn.Linear(embed_dim, embed_dim) for _ in range(gcn_layers)
        )
        self.rel_lin = nn.ModuleList(
            nn.ModuleDict(
                {r: nn.Linear(embed_dim, embed_dim, bias=False) for r in self.RELATIONS}
            )
            for _ in range(gcn_layers)
        )
        # learnable per-layer, per-relation mixing weight (softmax over relations)
        self.rel_gate = nn.Parameter(torch.zeros(gcn_layers, n_rel))

        self.register_buffer("binary_hex", BINARY_HEX.clone())          # [64,6]
        adj = HEX_ADJ.clone()
        self.register_buffer("adj_ham", adj / adj.sum(-1, keepdim=True).clamp(min=1))
        self.register_buffer("idx_cuo", HEX_CUO.clone())               # [64]
        self.register_buffer("idx_zong", HEX_ZONG.clone())             # [64]
        self.register_buffer("idx_hu", HEX_HU.clone())                 # [64]
        self.register_buffer("lower_tri", HEX_LOWER_TRIGRAM.clone())   # [64]
        self.register_buffer("upper_tri", HEX_UPPER_TRIGRAM.clone())   # [64]

    def _relation_message(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name == "ham":
            return self.adj_ham @ x                 # weighted neighbour mean
        if name == "cuo":
            return x[self.idx_cuo]                  # permutation gather
        if name == "zong":
            return x[self.idx_zong]
        if name == "hu":
            return x[self.idx_hu]
        raise KeyError(name)

    def relation_weights(self) -> torch.Tensor:
        """[gcn_layers, n_rel] softmax gates -- for inspection."""
        return torch.softmax(self.rel_gate, dim=1)

    def hex_features(self) -> torch.Tensor:
        """Graph-smoothed hexagram embeddings, [64, embed_dim]."""
        x = (
            self.hex_embed.weight
            + self.tri_embed(self.lower_tri)
            + self.tri_embed(self.upper_tri)
        )
        for layer, (slin, rlin) in enumerate(zip(self.self_lin, self.rel_lin)):
            gate = torch.softmax(self.rel_gate[layer], dim=0)          # [n_rel]
            msg = slin(x)
            for i, r in enumerate(self.RELATIONS):
                msg = msg + gate[i] * rlin[r](self._relation_message(r, x))
            x = x + torch.relu(msg)
        return x

    def forward(self, yao: torch.Tensor) -> torch.Tensor:
        """[batch, 6] -> [batch, 64] logits."""
        q = self.query(yao)                              # [B, E]
        h = self.hex_features()                          # [64, E]
        sim = self.scale * (q @ h.t())                   # [B, 64]
        polar = yao @ (2.0 * self.binary_hex.t() - 1.0)  # [B, 64], sign agreement
        return sim + self.alpha * polar
