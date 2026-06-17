import os
import random

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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


def _train_one_epoch(
    encoder,
    classifier,
    drug_loader,
    itc_loader,
    optimizer,
    criterion,
    device,
    scaler=None,
):

    encoder.train()
    classifier.train()

    total_loss = 0.0
    total_acc = 0.0
    batch_counter = 0
    for d1, d2, labels in itc_loader:
        batch_counter += 1
        d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                all_drugs = torch.cat(
                    [encoder(drugs.to(device)) for drugs in drug_loader]
                )
                logits = classifier(all_drugs[d1], all_drugs[d2])
                logits = torch.clamp(logits, min=-8.0, max=8.0)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(classifier.parameters()), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            all_drugs = torch.cat([encoder(drugs.to(device)) for drugs in drug_loader])
            logits = classifier(all_drugs[d1], all_drugs[d2])
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean()

        total_loss += loss.item()
        total_acc += acc.item()
    avg_loss = total_loss / len(itc_loader)
    avg_acc = total_acc / len(itc_loader)
    return avg_loss, avg_acc


def _val_one_epoch(
    encoder,
    classifier,
    drug_loader,
    itc_loader,
    criterion,
    device,
):
    encoder.eval()
    classifier.eval()

    val_loss = 0.0
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        all_drugs = torch.cat([encoder(drugs.to(device)) for drugs in drug_loader])

        for d1, d2, labels in itc_loader:
            d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)

            logits = classifier(all_drugs[d1], all_drugs[d2])
            logits = torch.clamp(logits, min=-8.0, max=8.0)
            loss = criterion(logits, labels)

            prob = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            val_loss += loss.item()
            all_probs.append(prob.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    avg_loss = val_loss / len(itc_loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    return (avg_loss, acc, f1, auc, prec, rec)


def train(cfg: BaseConfig):

    task_name = cfg.__name__
    epochs = cfg.epochs
    seed = cfg.seed
    lr = cfg.lr / 10

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_root = os.path.join("./split-data", cfg.split_type + "-" + str(seed))
    output_root = os.path.join("./task", task_name)

    os.makedirs(output_root, exist_ok=True)

    model_path = os.path.join(output_root, "fine-tuned.pt")
    ft_res_path = os.path.join(output_root, "fine-tune-metric.csv")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    drug_set = DrugDataset(input_root)
    train_itc = InteractionDataset(input_root, "train", "fine-tune")
    val_itc = InteractionDataset(input_root, "val", "fine-tune")

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
        lr=lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=0.00001)
    criterion = CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    early_stop = EarlyStop(patience=cfg.patience, mode="max", min_delta=cfg.min_delta)
    scaler = torch.GradScaler() if torch.cuda.is_available() and cfg.scaler else None

    result = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "prec": [],
        "rec": [],
        "f1": [],
        "auc": [],
        "elapsed": [],
    }

    total_timer = 0

    for epoch in range(0, epochs):
        current_epoch = epoch + 1

        with Timer() as timer:
            train_loss, train_acc = _train_one_epoch(
                encoder,
                classifier,
                drug_loader,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
            )
        total_timer += timer.elapsed
        result["train_loss"].append(train_loss)
        result["train_acc"].append(train_acc)

        with Timer() as timer:

            val_loss, val_acc, val_f1, val_auc, val_prec, val_rec = _val_one_epoch(
                encoder,
                classifier,
                drug_loader,
                val_loader,
                criterion,
                device,
            )
        total_timer += timer.elapsed
        result["val_loss"].append(val_loss)
        result["val_acc"].append(val_acc)
        result["f1"].append(val_f1)
        result["auc"].append(val_auc)
        result["prec"].append(val_prec)
        result["rec"].append(val_rec)
        result["elapsed"].append(total_timer)


        scheduler.step()
        is_improved = early_stop(val_auc)
 
        if is_improved:
            torch.save(
                {
                    "epoch": current_epoch,
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                },
                model_path,
            )
      
        else:
            pass

        if early_stop.early_stop:

            break
    pd.DataFrame(result).to_csv(ft_res_path, index=False)
