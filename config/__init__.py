from .config import BaseConfig
from .BN import *


class default_1(BaseConfig):
    data_source = "drugbank"
    split_type = "random"
    epochs = 200
    node_dim = 93
    edge_dim = 17
    graph_dim = 1051
    d_model = 128
    lr = 1e-3
    heads = 8
    dp_r = 0.1
    seed = 42
    block_num = 5
    class_num = 2
    drug_batch_size = 2048
    itc_batch_size =20480
    label_smoothing = 0.1
    weight_decay = 5e-4
    patience = 10
    min_delta = 1e-3
    num_workers = 0


__all__ = ["BaseConfig"]
