import os
import random
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
from torch.nn import CrossEntropyLoss
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

    train_loss = 0.0
    train_acc = 0.0
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
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(classifier.parameters()), max_norm=1.0
            )
            optimizer.step()

        preds = torch.argmax(logits, dim=1)

        acc = (preds == labels).float().mean()

        train_loss += loss.item()
        train_acc += acc.item()
        ptr.w_flush(
            "train",
            f"[Batch:{batch_counter}/{total_batch}] loss:{loss.item():.5f},acc:{acc.item():.5f}",
        )
    avg_train_loss = train_loss / len(itc_loader)
    avg_train_acc = train_acc / len(itc_loader)
    return avg_train_loss, avg_train_acc


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

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        all_drugs = torch.cat([encoder(drugs.to(device)) for drugs in drug_loader])

        for d1, d2, labels in itc_loader:
            d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)
            logits = classifier(all_drugs[d1], all_drugs[d2])
            logits = torch.clamp(logits, min=-8.0, max=8.0)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=-1)
            prob = torch.softmax(logits, dim=-1)

            val_loss += loss.item()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(prob.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    avg_loss = val_loss / len(itc_loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    pos_probs = all_probs[:, 1]
    auc = roc_auc_score(all_labels, pos_probs)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    return (avg_loss, acc, f1, auc, precision, recall)


def _train(
    config: BaseConfig,
    history=None,
):
    torch.autograd.set_detect_anomaly(True)

    name = config.__name__
    data_source = config.data_source
    split_type = config.split_type
    epochs = config.epochs
    node_dim = config.node_dim
    edge_dim = config.edge_dim
    graph_dim = config.graph_dim
    d_model = config.d_model
    lr = config.lr
    heads = config.heads
    dp_r = config.dp_r
    weight_decay = config.weight_decay
    patience = config.patience
    min_delta = config.min_delta
    seed = config.seed
    block_num = config.block_num
    drug_batch_size = config.drug_batch_size
    itc_batch_size = config.itc_batch_size
    label_smoothing = config.label_smoothing
    num_workers = config.num_workers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_epoch = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_itc_generator = torch.Generator()
    train_itc_generator.manual_seed(seed)
    root = os.path.join(
        "./split_data", data_source + "-" + split_type + "-" + str(seed)
    )
    base_dir = os.path.join("./checkpoints", name)
    os.makedirs(base_dir, exist_ok=True)
    best_path = os.path.join(base_dir, "best.pt")
    evaluate_path = os.path.join(base_dir, "evaluate.csv")
    drug_set = DrugDataset(root)
    train_itc = InteractionDataset(root, "train", "pre")
    val_itc = InteractionDataset(root, "val", "pre")
    drug_loader = DataLoader(
        drug_set,
        collate_fn=drug_collate_fn,
        batch_size=drug_batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    train_loader = DataLoader(
        train_itc,
        collate_fn=itc_collate_fn,
        batch_size=itc_batch_size,
        num_workers=num_workers,
        shuffle=True,
        generator=train_itc_generator,
    )

    val_loader = DataLoader(
        val_itc,
        collate_fn=itc_collate_fn,
        batch_size=itc_batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    encoder = AttnGINTFEncoder(
        node_dim, edge_dim, graph_dim, d_model, block_num, dp_r, heads
    ).to(device)

    classifier = BClassifier(d_model, dp_r).to(device)
    optimizer = AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=0.00001)
    criterion = CrossEntropyLoss(label_smoothing=label_smoothing)
    early_stop = EarlyStop(patience=patience, mode="max", min_delta=min_delta)
    # scaler = torch.GradScaler() if torch.cuda.is_available() else None
    scaler = None
    ptr.set_value_batch(
        {
            "name": name,
            "epochs": epochs,
            "encoder": type(encoder).__name__,
            "classifier": type(classifier).__name__,
            "lr": lr,
            "data_source": data_source,
            "split_type": split_type,
            "seed": seed,
            "device": device,
            "resume": "False",
            "epoch": f"0/{epochs}",
            "current_lr": lr,
            "elapsed": 0,
            "early_stop": f"0/{early_stop.patience}",
            "state": "pending",
        }
    )

    result = {
        "train_loss": [],
        "train_acc": [],
        "train_timer": [],
        "val_loss": [],
        "val_acc": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "auc": [],
        "val_timer": [],
    }

    total_timer = 0

    if history is not None:
        start_epoch = history["epoch"]
        encoder.load_state_dict(history["encoder"])
        classifier.load_state_dict(history["classifier"])
        optimizer.load_state_dict(history["optimizer"])
        scheduler.load_state_dict(history["scheduler"])
        early_stop.load_state_dict(history["early_stop"])
        if scaler is not None and history["scaler"] is not None:
            scaler.load_state_dict(history["scaler"])

        if (
            "train_itc_generator" in history
            and history["train_itc_generator"] is not None
        ):
            train_itc_generator.set_state(history["train_itc_generator"])

        if torch.cuda.is_available() and history["cuda_random"] is not None:
            torch.cuda.set_rng_state_all(history["cuda_random"])
        torch.random.set_rng_state(history["torch_random"])
        np.random.set_state(history["numpy_random"])
        random.setstate(history["python_random"])
        result = history["result"].to_dict(orient="list")
        total_timer = sum(result["train_timer"]) + sum(result["val_timer"])

        ptr.write("epoch", f"{start_epoch}/{epochs}")
        ptr.write("resume", "True", ptr_color.flag)
        ptr.write("current_lr", f"{optimizer.param_groups[0]['lr']:.7f}")
        ptr.scl_flush(
            "info",
            f"Checkpoint loaded. Resuming from epoch {start_epoch + 1}.",
            ptr_color.notice,
        )

    if early_stop.early_stop:
        ptr.scroll(
            "info",
            f"Early stop already triggered at epoch {start_epoch}/{epochs} — nothing to resume",
            ptr_color.warning,
        )
        ptr.write(
            "early_stop",
            f"{early_stop.counter}/{early_stop.patience}",
            ptr_color.error,
        )
        ptr.w_flush("state", "finished", ptr_color.done)
        return

    if start_epoch >= epochs:
        ptr.scroll(
            "info",
            f"Experiment already finished at epoch {start_epoch}/{epochs} — nothing to resume",
            ptr_color.warning,
        )
        ptr.w_flush("state", "finished", ptr_color.done)
        return

    for epoch in range(start_epoch, epochs):
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
        result["train_timer"].append(timer.elapsed)
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
            val_loss, val_acc, val_f1_score, val_auc, val_precision, val_recall = (
                _val_one_epoch(
                    encoder,
                    classifier,
                    drug_loader,
                    val_loader,
                    criterion,
                    device,
                )
            )
        total_timer += timer.elapsed
        result["val_loss"].append(val_loss)
        result["val_acc"].append(val_acc)
        result["f1_score"].append(val_f1_score)
        result["auc"].append(val_auc)
        result["val_timer"].append(timer.elapsed)
        result["precision"].append(val_precision)
        result["recall"].append(val_recall)
        ptr.write(
            "val",
            f"loss={val_loss:.5f}  acc={val_acc:.5f}  f1_score={val_f1_score:.5f}  auc={val_auc:.5f}  precision={val_precision:.5f} recall={val_recall:.5f} ({timer.elapsed:.5f} s)",
        )
        ptr.write(
            "elapsed",
            f"{total_timer:.5f}",
        )

        scheduler.step()
        is_improved = early_stop(val_f1_score)
        ptr.write("state", "waiting", ptr_color.pending)
        ptr.w_flush("current_lr", f"{optimizer.param_groups[0]['lr']:.7f}")
        if is_improved:
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                },
                best_path,
            )
            pd.DataFrame(
                {
                    "epoch": [current_epoch],
                    "loss": [val_loss],
                    "acc": [val_acc],
                    "precision": [val_precision],
                    "recall": [val_recall],
                    "f1_score": [val_f1_score],
                    "auc": [val_auc],
                }
            ).to_csv(evaluate_path, index=False)
            _save_checkpoint(
                current_epoch,
                encoder,
                classifier,
                optimizer,
                scheduler,
                early_stop,
                scaler,
                train_itc_generator,
                result,
                base_dir,
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
                f"[{current_epoch}/{epochs}] loss={val_loss:.5f}  acc={val_acc:.5f}  f1_score={val_f1_score:.5f}  auc={val_auc:.5f}  precision={val_precision:.5f} recall={val_recall:.5f}  ({timer.elapsed:.5f} s)",
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

        if current_epoch % 20 == 0:
            _save_checkpoint(
                current_epoch,
                encoder,
                classifier,
                optimizer,
                scheduler,
                early_stop,
                scaler,
                train_itc_generator,
                result,
                base_dir,
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
            _save_checkpoint(
                current_epoch,
                encoder,
                classifier,
                optimizer,
                scheduler,
                early_stop,
                scaler,
                train_itc_generator,
                result,
                base_dir,
            )
            break


def _save_checkpoint(
    epoch,
    encoder,
    classifier,
    optimizer,
    scheduler,
    early_stop,
    scaler,
    train_itc_generator,
    result,
    base_dir,
):
    history_path = os.path.join(base_dir, "history.pt")
    result_path = os.path.join(base_dir, "result.csv")
    checkpoint = {
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "classifier": classifier.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "early_stop": early_stop.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "cuda_random": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
        "torch_random": torch.random.get_rng_state(),
        "numpy_random": np.random.get_state(),
        "python_random": random.getstate(),
        "train_itc_generator": train_itc_generator.get_state(),
    }
    torch.save(checkpoint, history_path)
    pd.DataFrame(result).to_csv(result_path, index=False)
    ptr.scl_flush(
        "info",
        f"[Epoch:{epoch}]  checkpoint saved (history.pt + result.csv)",
        ptr_color.notice,
    )


def resume_training(config_class_name: str):
    cfg = getattr(config, config_class_name)
    history_path = os.path.join("./checkpoints", cfg.__name__, "history.pt")
    result_path = os.path.join("./checkpoints", cfg.__name__, "result.csv")
    history = torch.load(history_path, weights_only=False)
    result = pd.read_csv(result_path)
    history["result"] = result
    _train(cfg, history=history)


def run_training(config_class_name: str):
    cfg = getattr(config, config_class_name)
    _train(cfg)


def train_all(configs: list[str]):
    for cfg in configs:
        run_training(cfg)


__all__ = ["resume_training", "run_training", "train_all"]
