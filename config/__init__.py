from .config import BaseConfig


class default(BaseConfig):
    split_type = "random"
    epochs = 200
    node_dim = 93
    edge_dim = 17
    graph_dim = 1024
    d_model = 128
    lr = 1e-3
    heads = 8
    dp_r = 0.1
    seed = 42
    block_num = 5
    drug_batch_size = 2048
    itc_batch_size = 20480
    weight_decay = 5e-4
    patience = 10
    min_delta = 5e-4
    num_workers = 0

    label_smoothing = 0.1
    class_num = 87

    scaler = False


__all__ = ["BaseConfig"]
