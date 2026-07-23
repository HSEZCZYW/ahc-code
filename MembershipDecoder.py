import torch
import torch.nn as nn

from torch import Tensor


class MembershipDecoder(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_z: Tensor, edge_z: Tensor):


        feat = torch.cat([
            node_z,
            edge_z,
            node_z * edge_z,
            torch.abs(node_z - edge_z),
        ], dim=-1)
        return self.mlp(feat).squeeze(-1)
