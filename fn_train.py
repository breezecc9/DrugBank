import os
import config
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
from config import BaseConfig
from process_data import (
    DrugDataset,
    InteractionDataset,
    Timer,
    itc_collate_fn,
    drug_collate_fn,
)
from pretty_printer import pt_printer as ptr, PtrColor


def _train_one_epoch(
    encoder,
    classifier,
    drug_loader,
    itc_loader,
    optimizer,
    criterion,
    device,
):
    encoder.eval()
    classifier.train()
    total_batch = len(itc_loader)
    total_loss = 0.0
    total_acc = 0.0
    batch_counter = 0

    with torch.no_grad():
        all_drugs = torch.cat([encoder(drugs.to(device)) for drugs in drug_loader])

    for d1, d2, labels in itc_loader:
        batch_counter += 1
        d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)

        optimizer.zero_grad()
        # all_drugs = torch.cat([encoder(drugs.to(device)) for drugs in drug_loader])
        logits = classifier(all_drugs[d1], all_drugs[d2])
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean()

        total_loss += loss.item()
        total_acc += acc.item()

        ptr.w_flush(
            "train",
            {
                "batch": f"{batch_counter}/{total_batch}",
                "loss": f"{loss.item():.5f}",
                "acc": f"{acc.item():.5f}",
                "elapsed": "--",
            },
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


def _train(cfg: BaseConfig):

    task_name = cfg.__name__
    epochs = cfg.epochs
    # lr = cfg.lr / 10
    lr = cfg.lr
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ptr.write("name", task_name)
    ptr.write("stage", "fine-tune")
    ptr.write("lr", f"{lr:.7f}/{lr:.7f}")
    ptr.write("state", "waiting", PtrColor.pending)
    ptr.w_flush("device", device)

    input_root = os.path.join("./split-data", cfg.split_type + "-" + str(cfg.seed))
    output_root = os.path.join("./task", task_name)

    model_path = os.path.join(output_root, "fine-tune.pt")
    ft_res_path = os.path.join(output_root, "fine-tune-metric.csv")

    pre_train_model_path = os.path.join(output_root, "pre-train.pt")
    os.makedirs(output_root, exist_ok=True)

    if not os.path.exists(pre_train_model_path):
        ptr.scl_flush(
            "info",
            f"Failed to load pre-trained model weights from: {pre_train_model_path}",
        )
        return

    pt_weights = torch.load(pre_train_model_path, weights_only=False)

    ptr.scl_flush(
        "info", f"loaded pre-trained model weights from: {pre_train_model_path}"
    )

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
    ).to(device)

    encoder.load_state_dict(pt_weights["encoder"])

    for param in encoder.parameters():
        param.requires_grad = False

    classifier = MClassifier(cfg.d_model, cfg.class_num, cfg.dp_r).to(device)
    optimizer = torch.optim.AdamW(
        list(classifier.parameters()),
        lr=lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=0.00001
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    early_stop = EarlyStop(patience=cfg.patience, mode="max", min_delta=cfg.min_delta)

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
            ptr.w_flush("state", "training", PtrColor.training)
            train_loss, train_acc = _train_one_epoch(
                encoder,
                classifier,
                drug_loader,
                train_loader,
                optimizer,
                criterion,
                device,
            )
        total_timer += timer.elapsed
        result["train_loss"].append(train_loss)
        result["train_acc"].append(train_acc)
        ptr.write(
            "train",
            {
                "batch": "--",
                "loss": f"{train_loss:.5f}",
                "acc": f"{train_acc:.5f}",
                "elapsed": f"{timer.elapsed:.5f}",
            },
        )
        ptr.w_flush(
            "elapsed",
            f"{total_timer:.5f}",
        )
        with Timer() as timer:
            ptr.w_flush("state", "valdating", PtrColor.validating)
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
            {
                "loss": f"{val_loss:.5f}",
                "acc": f"{val_acc:.5f}",
                "f1": f"{val_f1:.5f}",
                "auc": f"{val_auc:.5f}",
                "prec": f"{val_prec:.5f}",
                "rec": f"{val_rec:.5f}",
                "elapsed": f"{timer.elapsed:.5f}",
            },
        )
        ptr.write(
            "elapsed",
            f"{total_timer:.5f}",
        )
        ptr.write("state", "waiting", PtrColor.pending)
        scheduler.step()
        is_improved = early_stop(val_f1)
        ptr.w_flush("lr", f"{optimizer.param_groups[0]['lr']:.7f}/{lr:.7f}")
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
                PtrColor.info,
            )
            ptr.write(
                "best",
                {
                    "loss": f"{val_loss:.5f}",
                    "acc": f"{val_acc:.5f}",
                    "f1": f"{val_f1:.5f}",
                    "auc": f"{val_auc:.5f}",
                    "prec": f"{val_prec:.5f}",
                    "rec": f"{val_rec:.5f}",
                },
                PtrColor.pending,
            )
            ptr.scl_flush(
                "info",
                f"[Epoch {current_epoch}/{epochs}] new best model saved to {model_path}",
                PtrColor.flag,
            )
        else:
            ptr.scroll(
                "info",
                f"[{current_epoch}/{epochs}] performance not improved",
                PtrColor.warning,
            )
            ptr.w_flush(
                "early_stop",
                f"{early_stop.counter}/{early_stop.patience}",
                PtrColor.warning,
            )

        if early_stop.early_stop:
            ptr.write(
                "early_stop",
                f"{early_stop.counter}/{early_stop.patience}",
                PtrColor.error,
            )
            ptr.scroll(
                "info",
                f"[{current_epoch}/{epochs}] early stopping triggered",
                PtrColor.error,
            )
            ptr.w_flush("state", "done", PtrColor.done)
            break
    pd.DataFrame(result).to_csv(ft_res_path, index=False)
    ptr.scroll(
        "info",
        f"{task_name} fine-tune already completed",
        PtrColor.notice,
    )
    ptr.scl_flush(
        "info",
        f"{task_name} fine-tune already completed",
        PtrColor.notice,
    )


def _test(cfg: BaseConfig):

    task_name = cfg.__name__
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = os.path.join("./split-data", cfg.split_type + "-" + str(cfg.seed))

    base_dir = os.path.join("./task", task_name)
    best_path = os.path.join(base_dir, "fine-tuned.pt")
    eval_path = os.path.join(base_dir, "fine-tuned-eval.csv")

    drug_set = DrugDataset(root)
    test_itc = InteractionDataset(root, "test", "fine-tune")

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
    classifier = MClassifier(cfg.d_model, class_num=cfg.class_num, dp_r=cfg.dp_r).to(
        device
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    ptr.write("name", cfg.__name__)
    ptr.write("stage", "fine-tune")
    ptr.write("state", "test", PtrColor.flag)
    ptr.w_flush("device", device)

    if not os.path.exists(best_path):
        ptr.scl_flush("info", f"best_path: {best_path} not existed")
        return

    best_model = torch.load(best_path, weights_only=False)
    ptr.scl_flush("info", f"loaded best model from:{best_path}")

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
    ptr.w_flush(
        "test",
        {
            "loss": f"{loss:.5f}",
            "acc": f"{acc:.5f}",
            "f1": f"{f1:.5f}",
            "auc": f"{auc:.5f}",
            "prec": f"{prec:.5f}",
            "rec": f"{rec:.5f}",
        },
    )
    pd.DataFrame(
        {"loss": loss, "acc": acc, "f1": f1, "auc": auc, "prec": prec, "rec": rec},
        index=[0],
    ).to_csv(eval_path, index=False)
    ptr.scl_flush("info", f"eval data saved to {eval_path}", color=PtrColor.flag)


def fine_tune(configs: list[str]):
    for cfg in configs:
        _train(getattr(config, cfg))
        _test(getattr(config, cfg))


__all__ = ["fine_tune"]
