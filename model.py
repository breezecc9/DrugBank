import torch
from torch import nn
from torch_geometric.utils import softmax, scatter
from typing import Literal


class AttnGIN(nn.Module):
    def __init__(self, d_model, dp_r, heads):
        super().__init__()
        self.d_model = d_model
        self.dp_r = dp_r
        self.heads = heads

        assert d_model % heads == 0, (
            f"d_model {d_model} must be divisible by heads {heads}"
        )
        self.head_dim = d_model // heads

        self.msg_proj = nn.Linear(d_model * 2, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_a = nn.Dropout(dp_r)
        self.dropout_o = nn.Dropout(dp_r)

    def forward(self, x, edge_index, edge_attr):
        # E:edge num, M:model dim, H:head num, D:head dim, N:node num
        x_j = x[edge_index[0]]  # E,M
        x_i = x[edge_index[1]]  # E,M
        msg_input = self.msg_proj(torch.cat([x_j, edge_attr], dim=-1))  # E,2M -> E,M

        q: torch.Tensor = self.q_proj(x_i).view(
            -1, self.heads, self.head_dim
        )  # E,M -> E,H,D
        k: torch.Tensor = self.k_proj(msg_input).view(
            -1, self.heads, self.head_dim
        )  # E,M -> E,H,D
        v: torch.Tensor = self.v_proj(msg_input).view(
            -1, self.heads, self.head_dim
        )  # E,M -> E,H,D
        attn_score = torch.einsum("ehd,ehd->eh", q, k) / (self.head_dim**0.5)  # E,H,D
        alpha = softmax(attn_score, edge_index[1], dim=0)
        alpha = self.dropout_a(alpha)
        weighted_v = torch.einsum("eh,ehd->ehd", alpha, v).view(
            -1, self.d_model
        )  # E,H,D -> E,M
        weighted_v = self.out_proj(weighted_v)
        out = scatter(weighted_v, edge_index[1], dim=0, reduce="sum")  # N,M
        return self.dropout_o(out)


class FFN(nn.Module):
    def __init__(self, d_model: int = 256, d_ff: int | None = None, dp_r: float = 0.1):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = d_model * 4
        self.d_ff = d_ff
        self.up1 = nn.Linear(d_model, d_ff, False)
        self.up2 = nn.Linear(d_model, d_ff, False)
        self.down = nn.Linear(d_ff, d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dp_r)

    def forward(self, x):
        x = self.act(self.up1(x)) * self.up2(x)
        x = self.down(x)
        return self.dropout(x)


class AttnGINTransformerLayer(nn.Module):
    def __init__(self, d_model, dp_r, heads):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.dp_r = dp_r

        self.LN_e = nn.LayerNorm(d_model)
        self.LN_n = nn.LayerNorm(d_model)
        self.attGIN = AttnGIN(d_model, dp_r, heads)

        self.LN_f = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, dp_r=dp_r)

    def forward(self, node, edge_index, edge_attr):
        h_n = self.LN_n(node)
        h_e = self.LN_e(edge_attr)
        h = self.attGIN(h_n, edge_index, h_e)
        node = node + h
        h = self.LN_f(node)
        h = self.ffn(h)
        return node + h


class AttnReadout(nn.Module):
    def __init__(self, d_model, heads, dp_r):
        super().__init__()
        self.d_model = d_model
        self.dp_r = dp_r

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.heads = heads
        assert d_model % heads == 0, (
            f"d_model {d_model} must be divisible by heads {heads}"
        )
        self.head_dim = d_model // heads
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_a = nn.Dropout(self.dp_r)
        self.dropout_o = nn.Dropout(self.dp_r)

    def forward(self, node, batch_index, graph_attr):
        # N:node num, B:batch size, M:model dim, H:head num, D:head dim
        q: torch.Tensor = self.q_proj(graph_attr[batch_index]).view(
            -1, self.heads, self.head_dim
        )  # N,H,D
        k: torch.Tensor = self.k_proj(node).view(-1, self.heads, self.head_dim)  # N,H,D
        v: torch.Tensor = self.v_proj(node).view(-1, self.heads, self.head_dim)  # N,H,D
        attn_score = torch.einsum("nhd,nhd->nh", q, k) / (self.head_dim**0.5)  # N,H
        alpha = softmax(attn_score, batch_index, dim=0)  # N,H
        alpha = self.dropout_a(alpha)  # N,H
        weighted_v = torch.einsum("nh,nhd->nhd", alpha, v).view(-1, self.d_model)  # N,M
        weighted_v = self.out_proj(weighted_v)
        out = scatter(weighted_v, batch_index, dim=0, reduce="sum")  # B,M
        return self.dropout_o(out)


class AttnReadoutTransformerLayer(nn.Module):
    def __init__(self, d_model, heads, dp_r):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.dp_r = dp_r

        self.LN_n = nn.LayerNorm(d_model)
        self.LN_g = nn.LayerNorm(d_model)
        self.Readout = AttnReadout(d_model, heads, dp_r)
        self.LN_f = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model=d_model, dp_r=dp_r)

    def forward(self, node, batch_index, graph_attr):
        h_n = self.LN_n(node)
        h_g = self.LN_g(graph_attr)
        h = self.Readout(h_n, batch_index, h_g)
        graph_attr = graph_attr + h
        h = self.LN_f(graph_attr)
        h = self.ffn(h)
        return h + graph_attr


class AttnGINTFEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        graph_dim: int,
        d_model: int,
        block_num: int,
        dp_r: float,
        heads: int,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.graph_dim = graph_dim
        self.d_model = d_model
        self.block_num = block_num
        self.heads = heads
        self.dp_r = dp_r

        self.node_proj = (
            nn.Linear(node_dim, d_model) if node_dim != d_model else nn.Identity()
        )
        self.edge_proj = (
            nn.Linear(edge_dim, d_model) if edge_dim != d_model else nn.Identity()
        )
        self.graph_proj = (
            nn.Linear(graph_dim, d_model) if graph_dim != d_model else nn.Identity()
        )

        self.ffn_g = FFN(d_model, dp_r=dp_r)

        self.attn_gin_tfl_list = nn.ModuleList(
            [
                AttnGINTransformerLayer(d_model, dp_r=dp_r, heads=heads)
                for _ in range(block_num)
            ]
        )

        self.attn_readout_tfl = AttnReadoutTransformerLayer(
            d_model, dp_r=dp_r, heads=heads
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, batch_data):
        node, edge_index, edge_attr, batch_index, graph_attr = (
            batch_data.x,
            batch_data.edge_index,
            batch_data.edge_attr,
            batch_data.batch,
            batch_data.graph_attr,
        )
        node = self.node_proj(node)
        edge_attr = self.edge_proj(edge_attr)
        graph_attr = self.graph_proj(graph_attr)
        graph_attr = self.ffn_g(graph_attr)

        for layer in self.attn_gin_tfl_list:
            node = layer(node, edge_index, edge_attr)

        out = self.attn_readout_tfl(node, batch_index, graph_attr)
        return self.final_norm(out)


class BModel(nn.Module):
    pass


class BClassifier(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, drug1, drug2) -> torch.Tensor:
        diff = torch.abs(drug1 - drug2)
        mul = drug1 * drug2
        fuse_feat = torch.cat([drug1, drug2, diff, mul], dim=-1)
        return self.mlp(fuse_feat)


class MClassifier(nn.Module):
    def __init__(self, d_model: int = 256, class_num: int = 2, dp_r: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.class_num = class_num
        self.drug1_proj = nn.Linear(d_model, d_model)
        self.drug2_proj = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.BatchNorm1d(d_model * 2),
            FFN(d_model * 2, dp_r=dp_r),
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            FFN(d_model, dp_r=dp_r),
            nn.Linear(d_model, class_num),
        )

    def forward(self, d1, d2):
        d1 = self.drug1_proj(d1)
        d2 = self.drug2_proj(d2)
        x = torch.cat([d1, d2, torch.abs(d1 - d2), d1 * d2], dim=-1)
        return self.mlp(x)


class EarlyStop:
    def __init__(
        self,
        patience: int = 5,
        mode: Literal["max", "min"] = "max",
        min_delta: float = 1e-4,
    ):
        self.patience = patience
        self.mode = mode

        self.min_delta = min_delta
        self.counter = 0
        self.best_metric_val = None
        self.early_stop = False

    def __call__(self, metric_value):
        if self.best_metric_val is None:
            is_improved = True
        else:
            if self.mode == "min":
                is_improved = metric_value < self.best_metric_val - self.min_delta
            else:
                is_improved = metric_value > self.best_metric_val + self.min_delta

        if is_improved:
            self.counter = 0
            self.best_metric_val = metric_value
            self.early_stop = False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return is_improved

    def state_dict(self):
        state = {
            "counter": self.counter,
            "best_metric_val": self.best_metric_val,
            "early_stop": self.early_stop,
        }
        return state

    def load_state_dict(self, state_dict):
        self.counter = state_dict["counter"]
        self.best_metric_val = state_dict["best_metric_val"]
        self.early_stop = state_dict["early_stop"]


__all__ = ["AttnGINTFEncoder", "MClassifier", "EarlyStop", "BClassifier"]
