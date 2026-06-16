import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from model import EarlyStop, MClassifier, AttnGINTFEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from config import BaseConfig
import config
from process_data import (
    DrugDataset,
    InteractionDataset,
    Timer,
    itc_collate_fn,
    drug_collate_fn,
)


def train(cfg: BaseConfig):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = os.path.join("./split_data", cfg.split_type + "-" + str(cfg.seed))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    drug_set = DrugDataset(root)
    train_itc = InteractionDataset(root, "train", "ft")
    val_itc = InteractionDataset(root, "val", "ft")
    drug_loader = DataLoader(
        drug_set,
        collate_fn=drug_collate_fn,
        batch_size=cfg.drug_batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )
    train_loader = DataLoader(
        train_itc,
        collate_fn=itc_collate_fn,
        batch_size=cfg.itc_batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_itc,
        collate_fn=itc_collate_fn,
        batch_size=cfg.itc_batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )

    encoder = AttnGINTFEncoder(
        cfg.node_dim,
        cfg.edge_dim,
        cfg.graph_dim,
        cfg.d_model,
        cfg.block_num,
        cfg.dp_r,
        cfg.heads,
    )
    classifier = MClassifier(cfg.d_model, cfg.class_num, cfg.dp_r).to(device)
    optimizer = AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, cfg.epochs, eta_min=0.00001)
    criterion = CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    early_stop = EarlyStop(patience=cfg.patience, mode="max", min_delta=cfg.min_delta)
