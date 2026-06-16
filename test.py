import os
import pandas as pd
import torch
from config import BaseConfig
import config
from process_data import  DrugDataset, InteractionDataset, drug_collate_fn, itc_collate_fn
from model import AttnGINTFEncoder, BClassifier
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _test(config: BaseConfig):
    name = config.__name__
    data_source = config.data_source
    split_type = config.split_type
    node_dim = config.node_dim
    edge_dim = config.edge_dim
    graph_dim = config.graph_dim
    d_model = config.d_model
    heads = config.heads
    dp_r = config.dp_r
    seed = config.seed
    block_num = config.block_num
    class_num = config.class_num
    drug_batch_size = config.drug_batch_size
    itc_batch_size = config.itc_batch_size
    label_smoothing = config.label_smoothing
    num_workers = config.num_workers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = os.path.join(
        "./split_data", data_source + "-" + split_type + "-" + str(seed)
    )
    drug_set = DrugDataset(root)
    test_itc = InteractionDataset(root, "test", "pre")
    drug_loader = DataLoader(
        drug_set,
        collate_fn=drug_collate_fn,
        batch_size=drug_batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_itc,
        collate_fn=itc_collate_fn,
        batch_size=itc_batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    encoder = AttnGINTFEncoder(
        node_dim, edge_dim, graph_dim, d_model, block_num, dp_r, heads
    ).to(device)
    classifier = BClassifier(d_model, dp_r).to(device)
    criterion = CrossEntropyLoss(label_smoothing=label_smoothing)
    base_dir = os.path.join("./checkpoints", name)
    best_path = os.path.join(base_dir, "best.pt")
    evaluate_path = os.path.join(base_dir, "evaluate.csv")
    cm_path = os.path.join(base_dir, "cm.csv")
    evaluate: dict
    if os.path.exists(evaluate_path):
        evaluate = pd.read_csv(evaluate_path).to_dict(orient="list")
        evaluate["epoch"].append(-1)
    else:
        evaluate = {"epoch": [-1], "loss": [], "acc": [], "f1_score": [], "auc": []}
    if not os.path.exists(best_path):
        return
    best_model = torch.load(best_path, weights_only=False)
    encoder.load_state_dict(best_model["encoder"])
    classifier.load_state_dict(best_model["classifier"])

    encoder.eval()
    classifier.eval()

    test_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        all_drugs = torch.cat([encoder(drugs.to(device)) for drugs in drug_loader])

        for d1, d2, labels in test_loader:
            d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)
            logits = classifier(all_drugs[d1], all_drugs[d2])
            logits = torch.clamp(logits, min=-30, max=30)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=-1)
            prob = torch.softmax(logits, dim=-1)

            test_loss += loss.item()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(prob.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()

    evaluate["loss"].append(test_loss / len(test_loader))
    evaluate["acc"].append(accuracy_score(all_labels, all_preds))
    evaluate["f1_score"].append(
        f1_score(all_labels, all_preds, average="macro", zero_division=0)
    )
    pos_probs = all_probs[:, 1]
    evaluate["auc"].append(
        roc_auc_score(all_labels, pos_probs)
    )
    evaluate["precision"].append(
        precision_score(all_labels, all_preds, average="binary",zero_division=0)
    )
    evaluate["recall"].append(recall_score(all_labels, all_preds, average="binary",zero_division=0))
    pd.DataFrame(evaluate).to_csv(evaluate_path, index=False)
    cm = confusion_matrix(all_labels, all_preds)
    pd.DataFrame(
        cm,
        index=[f"True_{i}" for i in range(class_num)],
        columns=[f"Pred_{i}" for i in range(class_num)],
    ).to_csv(cm_path)


def run_testing(config_class_name: str):
    _test(getattr(config, config_class_name))


def test_all(configs: list[str]):
    for cfg in configs:
        run_testing(cfg)


__all__ = ["run_testing", "test_all"]
