import time
import pandas as pd
import torch
from process_data import (
    DrugDataset,
    InteractionDataset,
    drug_collate_fn,
    itc_collate_fn,
)
from torch.utils.data import DataLoader
from collections import Counter


def test_drug_num_workers(num_workers):

    drug_set = DrugDataset("./split_data/drugbank-random-42")
    drug_loader = DataLoader(
        drug_set,
        batch_size=2048,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=drug_collate_fn,
    )
    start = time.time()
    for drug in drug_loader:
        pass
    print(f"num_workers={num_workers}: {time.time() - start:.2f}s")


def test_itc_num_workers(num_workers):

    start = time.time()
    itc_set = InteractionDataset("./split_data/drugbank-random-42", "train")
    itc_loader = DataLoader(
        itc_set,
        batch_size=20480,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=itc_collate_fn,
    )

    for itc in itc_loader:
        pass
    print(f"num_workers={num_workers}: {time.time() - start:.2f}s")


def has_nan_inf():
    drug_set = DrugDataset("./split_data/drugbank-random-42")
    drug_loader = DataLoader(
        drug_set,
        batch_size=2048,
        num_workers=0,
        shuffle=False,
        collate_fn=drug_collate_fn,
    )
    for drug in drug_loader:
        x_has_nan = torch.isnan(drug.x).any().item()
        edge_attr_has_nan = torch.isnan(drug.edge_attr).any().item()
        graph_attr_has_nan = torch.isnan(drug.graph_attr).any().item()
        x_has_inf = torch.isinf(drug.x).any().item()
        edge_attr_has_inf = torch.isinf(drug.edge_attr).any().item()
        graph_attr_has_inf = torch.isinf(drug.graph_attr).any().item()

        print("x_has_nan", x_has_nan)
        print("edge_attr_has_nan", edge_attr_has_nan)
        print("graph_attr_has_nan", graph_attr_has_nan)
        print("x_has_inf", x_has_inf)
        print("edge_attr_has_inf", edge_attr_has_inf)
        print("graph_attr_has_inf", graph_attr_has_inf)


if __name__ == "__main__":
    x = pd.read_csv("./data/drugbank.tab", sep="\t")
    itc = x[["ID1", "ID2", "Y"]]
    raw = {}
    existed = {}
    for d1, d2, label in zip(itc["ID1"], itc["ID2"], itc["Y"]):
        pair = frozenset({d1, d2})
        if(pair in raw):
            existed[pair] = label
            continue
        raw[frozenset({d1, d2})] = label
    print(raw)
    print(existed)