import os
import random
from typing import Literal
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import BCEWithLogitsLoss
import numpy as np
import torch
from torch.utils.data import DataLoader
from process_data import (
    DrugDataset,
    InteractionDataset,
    Timer,
    itc_collate_fn,
    drug_collate_fn,
)
from model import EarlyStop, BClassifier, AttnGINTFEncoder
from config import BaseConfig
import config
from custom_printer import train_ptr as ptr, ptr_color


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
    total_batch = len(itc_loader)
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
                logits = logits.squeeze(-1)
                logits = torch.clamp(logits, min=-8.0, max=8.0)
                loss = criterion(logits, labels.float())

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
            logits = logits.squeeze(-1)
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()

        prob = torch.sigmoid(logits)
        preds = (prob > 0.5).long()
        acc = (preds == labels).float().mean()

        total_loss += loss.item()
        total_acc += acc.item()
        ptr.w_flush(
            "train",
            f"[Batch:{batch_counter}/{total_batch}]  Loss:{loss.item():.5f}  Acc:{acc.item():.5f}",
        )
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
            logits = logits.squeeze(-1)
            logits = torch.clamp(logits, min=-8.0, max=8.0)
            loss = criterion(logits, labels.float())

            prob = torch.sigmoid(logits)
            preds = (prob > 0.5).long()

            val_loss += loss.item()
            all_probs.append(prob.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    avg_loss = val_loss / len(itc_loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    prec = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    return (avg_loss, acc, f1, auc, prec, rec)


def _train(
    cfg: BaseConfig,
):

    task_name = cfg.__name__
    epochs = cfg.epochs
    seed = cfg.seed

    # torch.autograd.set_detect_anomaly(True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    input_root = os.path.join("./split-data", cfg.split_type + "-" + str(seed))

    output_root = os.path.join("./task", task_name)

    os.makedirs(output_root, exist_ok=True)

    model_path = os.path.join(output_root, "pre-trained.pt")
    pt_res_path = os.path.join(output_root, "pre-train-metric.csv")

    drug_set = DrugDataset(input_root)
    train_itc = InteractionDataset(input_root, "train", "pre-train")
    val_itc = InteractionDataset(input_root, "val", "pre-train")

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
    ).to(device)

    classifier = BClassifier(cfg.d_model, cfg.dp_r).to(device)
    optimizer = AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=0.00001)
    criterion = BCEWithLogitsLoss(reduction="mean")
    early_stop = EarlyStop(patience=cfg.patience, mode="max", min_delta=cfg.min_delta)
    scaler = torch.GradScaler() if torch.cuda.is_available() and cfg.scaler else None
    ptr.set_value_batch(
        {
            "name": task_name,
            "encoder": type(encoder).__name__,
            "classifier": type(classifier).__name__,
            "lr": {"CLR": cfg.lr, "LR": cfg.lr},
            "device": device,
            "epoch": f"0/{cfg.epochs}",
            "elapsed": 0,
            "early_stop": f"0/{early_stop.patience}",
            "state": "pending",
            "stage": "pre-train",
        }
    )

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
        ptr.w_flush("epoch", f"{current_epoch}/{epochs}")
        with Timer() as timer:
            ptr.w_flush("state", "training", ptr_color.training)
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
        ptr.write(
            "train",
            f"loss={train_loss:.5f}  acc={train_acc:.5f}  ({timer.elapsed:.5f} s)",
        )
        ptr.w_flush(
            "elapsed",
            f"{total_timer:.5f}",
        )

        with Timer() as timer:
            ptr.w_flush("state", "valdating", ptr_color.validating)
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
        ptr.write(
            "val",
            f"loss={val_loss:.5f}  acc={val_acc:.5f}  f1_score={val_f1:.5f}  auc={val_auc:.5f}  precision={val_prec:.5f}  recall={val_rec:.5f} ({timer.elapsed:.5f} s)",
        )
        ptr.write(
            "elapsed",
            f"{total_timer:.5f}",
        )

        scheduler.step()
        is_improved = early_stop(val_auc)
        ptr.write("state", "waiting", ptr_color.pending)
        ptr.w_flush(
            "lr", {"CLR": f"{optimizer.param_groups[0]['lr']:.7f}", "LR": cfg.lr}
        )
        if is_improved:
            torch.save(
                {
                    "epoch": current_epoch,
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                },
                model_path,
            )
            ptr.write(
                "early_stop",
                f"{early_stop.counter}/{early_stop.patience}",
                ptr_color.info,
            )
            ptr.scroll(
                "info",
                f"[{current_epoch}/{epochs}] Model performance improved",
                ptr_color.notice,
            )
            ptr.write(
                "best",
                f"[{current_epoch}/{epochs}] loss={val_loss:.5f}  acc={val_acc:.5f}  f1_score={val_f1:.5f}  auc={val_auc:.5f}  precision={val_prec:.5f}  recall={val_rec:.5f}  ({timer.elapsed:.5f} s)",
                ptr_color.pending,
            )
            ptr.scl_flush(
                "info",
                f"[{current_epoch}/{epochs}] best model improved → saved best.pt and evaluate.csv",
                ptr_color.notice,
            )
        else:
            ptr.scroll(
                "info",
                f"[{current_epoch}/{epochs}] Model performance not improved",
                ptr_color.warning,
            )
            ptr.w_flush(
                "early_stop",
                f"{early_stop.counter}/{early_stop.patience}",
                ptr_color.warning,
            )

        if early_stop.early_stop:
            ptr.write(
                "early_stop",
                f"{early_stop.counter}/{early_stop.patience}",
                ptr_color.error,
            )
            ptr.scroll(
                "info",
                f"[{current_epoch}/{epochs}] Early stopping triggered",
                ptr_color.warning,
            )
            ptr.w_flush("state", "finished", ptr_color.done)
            break
    pd.DataFrame(result).to_csv(pt_res_path, index=False)


def _test(cfg: BaseConfig):

    task_name = cfg.__name__
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = os.path.join("./split-data", cfg.split_type + "-" + str(cfg.seed))

    drug_set = DrugDataset(root)
    test_itc = InteractionDataset(root, "test", "pre-train")

    drug_loader = DataLoader(
        drug_set,
        collate_fn=drug_collate_fn,
        batch_size=cfg.drug_batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_itc,
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
    ).to(device)
    classifier = BClassifier(cfg.d_model, cfg.dp_r).to(device)
    criterion = BCEWithLogitsLoss(reduction="mean")

    base_dir = os.path.join("./task", task_name)
    best_path = os.path.join(base_dir, "pre-trained.pt")
    eval_path = os.path.join(base_dir, "eval.csv")

    if not os.path.exists(best_path):
        return

    best_model = torch.load(best_path, weights_only=False)

    encoder.load_state_dict(best_model["encoder"])
    classifier.load_state_dict(best_model["classifier"])

    loss, acc, f1, auc, prec, rec = _val_one_epoch(
        encoder,
        classifier,
        drug_loader,
        test_loader,
        criterion,
        device,
    )
    pd.DataFrame(
        {"loss": loss, "acc": acc, "f1": f1, "auc": auc, "prec": prec, "rec": rec},index=[0]
    ).to_csv(eval_path, index=False)


def pre_train(configs: list[str], mode: Literal["train", "test", "all"]):
    for cfg in configs:
        if mode in ("train", "all"):
            _train(getattr(config, cfg))
        if mode in ("test", "all"):
            _test(getattr(config, cfg))


__all__ = ["pre_train"]
