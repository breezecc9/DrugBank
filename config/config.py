from typing import ClassVar, Literal


class BaseConfig:
    _required_fields = {
        "data_source",
        "split_type",
        "epochs",
        "node_dim",
        "edge_dim",
        "graph_dim",
        "d_model",
        "lr",
        "heads",
        "dp_r",
        "seed",
        "block_num",
        "class_num",
        "drug_batch_size",
        "itc_batch_size",
        "label_smoothing",
        "weight_decay",
        "min_delta",
        "patience",
        "num_workers",
    }
    __name__: str
    data_source: ClassVar[Literal["drugbank", "twosides"]]
    split_type: ClassVar[Literal["random", "cluster"]]
    epochs: ClassVar[int]
    node_dim: ClassVar[int]
    edge_dim: ClassVar[int]
    graph_dim: ClassVar[int]
    d_model: ClassVar[int]
    lr: ClassVar[float]
    heads: ClassVar[int]
    dp_r: ClassVar[float]
    seed: ClassVar[int]
    block_num: ClassVar[int]
    class_num: ClassVar[int]
    drug_batch_size: ClassVar[int]
    itc_batch_size: ClassVar[int]
    label_smoothing: ClassVar[float]
    weight_decay: ClassVar[float]
    patience: ClassVar[int]
    min_delta: ClassVar[float]
    num_workers: ClassVar[int]

    @classmethod
    def __init_subclass__(cls):
        for field in cls._required_fields:
            if field not in cls.__dict__:
                print(field)
                raise NotImplementedError(
                    f"Subclass {cls.__name__} must explicitly set attribute: {field}"
                )
