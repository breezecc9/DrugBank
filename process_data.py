import math
import os.path
import random
from typing import Literal
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    Descriptors,
    rdMolDescriptors,
    Lipinski,
    GraphDescriptors,
    ValenceType,
)
from rdkit.ML.Cluster import Butina
from torch_geometric.data import Data, InMemoryDataset, Batch
from rdkit.Chem import rdCIPLabeler
import time
from collections import defaultdict


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


class DrugDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
    ):
        super().__init__(root, transform, pre_transform)
        self._data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )

    @property
    def processed_file_names(self):
        return ["drug.pt"]

    @property
    def raw_file_names(self):
        return ["drug.csv"]

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        data_list = []
        for smile in df["smile"]:
            mol = smiles_to_graph(smile)
            data_list.append(mol)
        self._data, self.slices = self.collate(data_list)
        torch.save((self._data, self.slices), self.processed_paths[0])


class InteractionDataset(Dataset):
    def __init__(
        self,
        root,
        type: Literal["train", "val", "test"] = "train",
        stage: Literal["pre", "ft"] = "pre",
    ):
        super().__init__()
        cache_key = f"{type}_{stage}_itc.pt"
        cache_file_path = os.path.join(root, "processed", cache_key)
        if not os.path.exists(cache_file_path):
            os.makedirs(os.path.join(root, "processed"), exist_ok=True)
            df = pd.read_csv(os.path.join(root, "raw", f"{type}_{stage}_itc.csv"))
            drug1 = torch.tensor(df["drug1"].values, dtype=torch.long)
            drug2 = torch.tensor(df["drug2"].values, dtype=torch.long)
            label = torch.tensor(df["label"].values, dtype=torch.long)
            torch.save((drug1, drug2, label), cache_file_path)

        drug1, drug2, label = torch.load(cache_file_path, weights_only=False)
        self.drug1 = drug1.share_memory_()
        self.drug2 = drug2.share_memory_()
        self.label = label.share_memory_()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.drug1[idx], self.drug2[idx], self.label[idx]


def split_data(
    data_source: Literal["drugbank", "twosides"] = "drugbank",
    split_type: Literal["random", "cluster"] = "random",
    ratio_tuple: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed=42,
):
    save_dir = os.path.join(
        "./split_data", data_source + "-" + split_type + "-" + str(seed), "raw"
    )
    os.makedirs(save_dir, exist_ok=True)
    if data_source == "drugbank":
        if split_type == "random":
            _split_drugbank_random(
                pd.read_csv("./data/drugbank.tab", sep="\t"),
                ratio_tuple,
                seed,
                save_dir,
            )
        else:
            raise TypeError()
    else:
        raise TypeError()


def _split_drugbank_random(df: pd.DataFrame, ratio_tuple, seed, save_dir):
    drug1 = df[["ID1", "X1"]].drop_duplicates(keep="first")
    drug2 = df[["ID2", "X2"]].drop_duplicates(keep="first")
    drug1.columns = ["id", "smile"]
    drug2.columns = ["id", "smile"]
    drug = (
        pd.concat([drug1, drug2])
        .drop_duplicates(subset=["id", "smile"], keep="first")
        .reset_index(drop=True)
    )
    id_map = {id: idx for idx, id in enumerate(drug["id"])}
    itc = df[["ID1", "ID2", "Y"]].drop_duplicates(keep="first").reset_index(drop=True)
    itc.columns = ["drug1", "drug2", "label"]
    itc["drug1"] = itc["drug1"].map(id_map)
    itc["drug2"] = itc["drug2"].map(id_map)
    itc["label"] = itc["label"] - 1
    unique_pairs = set()
    unique_itc = {"drug1": [], "drug2": [], "label": []}
    for d1, d2, label in zip(itc["drug1"], itc["drug2"], itc["label"]):
        pair = frozenset({d1, d2})
        if pair not in unique_pairs:
            unique_pairs.add(pair)
            unique_itc["drug1"].append(d1)
            unique_itc["drug2"].append(d2)
            unique_itc["label"].append(label)
    itc = pd.DataFrame(unique_itc)

    temp, test = train_test_split(
        itc, train_size=1 - ratio_tuple[2], random_state=seed, stratify=itc["label"]
    )

    train, val = train_test_split(
        temp,
        train_size=ratio_tuple[0] / (ratio_tuple[0] + ratio_tuple[1]),
        random_state=seed,
        stratify=temp["label"],
    )
    os.makedirs(save_dir, exist_ok=True)

    drug["smile"].reset_index().rename(columns={"index": "id"}).to_csv(
        os.path.join(save_dir, "drug.csv"),
        index=False,
    )
    itc.to_csv(
        os.path.join(save_dir, "all_itc.csv"),
        index=False,
    )
    train.to_csv(
        os.path.join(save_dir, "train_itc.csv"),
        index=False,
    )
    test.to_csv(
        os.path.join(save_dir, "test_itc.csv"),
        index=False,
    )
    val.to_csv(
        os.path.join(save_dir, "val_itc.csv"),
        index=False,
    )


def _one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]


ELECTRONEG = {
    "H": 2.20,
    "Li": 0.98,
    "B": 2.04,
    "C": 2.55,
    "N": 3.04,
    "O": 3.44,
    "F": 3.98,
    "Na": 0.93,
    "Mg": 1.31,
    "Al": 1.61,
    "Si": 1.90,
    "P": 2.19,
    "S": 2.58,
    "Cl": 3.16,
    "K": 0.82,
    "Ca": 1.00,
    "Ti": 1.54,
    "Cr": 1.66,
    "Fe": 1.83,
    "Co": 1.88,
    "Cu": 1.90,
    "Zn": 1.65,
    "Ga": 1.81,
    "As": 2.18,
    "Se": 2.55,
    "Br": 2.96,
    "Sr": 0.95,
    "Tc": 1.90,
    "Ag": 1.93,
    "Sb": 2.05,
    "I": 2.66,
    "La": 1.10,
    "Gd": 1.20,
    "Pt": 2.28,
    "Au": 2.54,
    "Hg": 2.00,
    "Bi": 2.02,
    "Ra": 0.90,
}


def _atom_features(atom):
    features = []

    # 1. Atom symbol (38)
    features += _one_hot_encoding(
        atom.GetSymbol(),
        [
            "H",
            "Li",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "K",
            "Ca",
            "Ti",
            "Cr",
            "Fe",
            "Co",
            "Cu",
            "Zn",
            "Ga",
            "As",
            "Se",
            "Br",
            "Sr",
            "Tc",
            "Ag",
            "Sb",
            "I",
            "La",
            "Gd",
            "Pt",
            "Au",
            "Hg",
            "Bi",
            "Ra",
        ],
    )

    # 2. Degree (7)
    features += _one_hot_encoding(
        atom.GetDegree(),
        [0, 1, 2, 3, 4, 5, 6],
    )

    # 3. Total hydrogens (5)
    features += _one_hot_encoding(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4, 5],
    )

    # 4. Formal charge (5)
    features += _one_hot_encoding(
        atom.GetFormalCharge(),
        [-2, -1, 0, 1, 2],
    )

    # 5. Aromaticity (1)
    features.append(int(atom.GetIsAromatic()))

    # 6. In ring (1)
    features.append(int(atom.IsInRing()))

    # 7. Hybridization (7)
    hyb_list = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.OTHER,
    ]
    features += _one_hot_encoding(atom.GetHybridization(), hyb_list)
    # 8. Chiral tag (3)
    features += _one_hot_encoding(
        atom.GetChiralTag(),
        [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ],
    )

    # 9. Atomic mass (1)
    features.append(atom.GetMass() / 100.0)
    # 10. 显式价态 8维
    features += _one_hot_encoding(
        atom.GetValence(ValenceType.EXPLICIT), [0, 1, 2, 3, 4, 5, 6, 7]
    )
    # 11. 隐式价态 8维
    features += _one_hot_encoding(
        atom.GetValence(ValenceType.IMPLICIT), [0, 1, 2, 3, 4, 5, 6, 7]
    )
    # 12. 是否杂原子 1维 (C/H=0，其余=1)
    symbol = atom.GetSymbol()
    features.append(0 if symbol in ("C", "H") else 1)
    # 13. 原子电负性 1维 (归一化)
    features.append(ELECTRONEG.get(symbol, 2.5) / 4.0)
    # 14. Gasteiger 部分电荷 1维
    try:
        charge = float(atom.GetProp("_GasteigerCharge"))
    except:
        charge = 0.0
    if math.isnan(charge) or math.isinf(charge):
        charge = 0.0
    features.append(charge)

    features.append(int(symbol in ("O", "N", "S") and atom.GetTotalNumHs() > 0))

    features.append(int(symbol in ("O", "N", "S", "F")))

    # 17. CIP 手性标签 (3维: R, S, None)
    try:
        cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "None"
    except:
        cip = "None"
    features += _one_hot_encoding(cip, ["R", "S", "None"])

    return features


def _bond_features(bond):
    bond_type = bond.GetBondType()

    features = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]

    # Bond stereochemistry
    features += _one_hot_encoding(
        bond.GetStereo(),
        [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
        ],
    )

    # 1. 浮点键级 1维
    features.append(bond.GetBondTypeAsDouble())
    # 2. 是否芳香键（二次强化）1维
    features.append(int(bond.GetIsAromatic()))
    # 3. 共轭环键 1维
    features.append(int(bond.GetIsConjugated() and bond.IsInRing()))
    # 4. 键在 ≤6 元环内 (1维)
    features.append(int(any(bond.IsInRingSize(s) for s in range(3, 7))))
    # 5. 两端原子形式电荷差 (1维)
    f1 = bond.GetBeginAtom().GetFormalCharge()
    f2 = bond.GetEndAtom().GetFormalCharge()
    features.append(f1 - f2)
    # 6. 键是否连接杂原子 (1维)
    sym1 = bond.GetBeginAtom().GetSymbol()
    sym2 = bond.GetEndAtom().GetSymbol()
    # 7. 桥键特征：两端原子环状态不同 (1维)
    r1 = bond.GetBeginAtom().IsInRing()
    r2 = bond.GetEndAtom().IsInRing()
    features.append(int(r1 != r2))
    features.append(0 if sym1 in ("C", "H") and sym2 in ("C", "H") else 1)

    return features


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol.UpdatePropertyCache(strict=False)
    AllChem.ComputeGasteigerCharges(mol)
    # Node features
    x = []
    for atom in mol.GetAtoms():
        x.append(_atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)
    # 分配 CIP 标签（用于手性特征）
    try:
        rdCIPLabeler.AssignCIPLabels(mol)
    except:
        pass

    # Edge features
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = _bond_features(bond)

        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bf)
        edge_attr.append(bf)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # ========== 全局特征 ==========
    gw = Descriptors.MolWt(mol) / 500.0
    logp = Descriptors.MolLogP(mol) / 10.0
    tpsa = Descriptors.TPSA(mol) / 250.0
    hdonor = Descriptors.NumHDonors(mol) / 10.0
    haccept = Descriptors.NumHAcceptors(mol) / 10.0
    rot_bond = Descriptors.NumRotatableBonds(mol) / 20.0
    ring_num = rdMolDescriptors.CalcNumRings(mol) / 10.0

    # 重原子数
    heavy_atom = Descriptors.HeavyAtomCount(mol) / 50.0
    # 芳香环数量
    aromatic_ring = rdMolDescriptors.CalcNumAromaticRings(mol) / 10.0
    # 脂肪环数量
    aliphatic_ring = rdMolDescriptors.CalcNumAliphaticRings(mol) / 10.0
    # 摩尔折射率
    mr = Descriptors.MolMR(mol) / 100.0
    # 分子柔性指数
    total_bonds = mol.GetNumBonds()
    frac_rot = Descriptors.NumRotatableBonds(mol) / max(1, total_bonds)
    # 卤素原子总数
    halogens = (
        sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I")) / 10.0
    )
    # 氧原子数、氮原子数
    o_count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O") / 10.0
    n_count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N") / 10.0
    # 分子复杂度 (BertzCT)
    complexity = GraphDescriptors.BertzCT(mol) / 1000.0
    # 不饱和碳原子比例
    unsaturated_c = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[C]=[C]")))
    unsat_c_ratio = unsaturated_c / max(1, Descriptors.HeavyAtomCount(mol))
    # Fsp3 (sp3杂化碳比例)
    fsp3 = Lipinski.FractionCSP3(mol) if total_bonds > 0 else 0.0
    # 正/负电荷数
    pos_charge = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() > 0) / 5.0
    neg_charge = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() < 0) / 5.0
    # 刚性键比例
    rigid_bonds = total_bonds - Descriptors.NumRotatableBonds(mol)
    rigid_ratio = rigid_bonds / max(1, total_bonds)
    # Kappa 形状指数 (归一化)
    kappa1 = GraphDescriptors.Kappa1(mol) / 20.0
    kappa2 = GraphDescriptors.Kappa2(mol) / 20.0
    kappa3 = GraphDescriptors.Kappa3(mol) / 20.0
    # Chi 分子连接性指数 (归一化)
    chi0v = GraphDescriptors.Chi0v(mol) / 10.0
    chi1v = GraphDescriptors.Chi1v(mol) / 10.0
    chi2v = GraphDescriptors.Chi2v(mol) / 10.0

    gen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp = gen.GetFingerprint(mol)
    tensor_fp = torch.tensor(np.array(fp), dtype=torch.float).unsqueeze(0)
    graph_attr = [
        gw,
        logp,
        tpsa,
        hdonor,
        haccept,
        rot_bond,
        ring_num,
        heavy_atom,
        aromatic_ring,
        aliphatic_ring,
        mr,
        frac_rot,
        halogens,
        o_count,
        n_count,
        complexity,
        unsat_c_ratio,
        fsp3,
        pos_charge,
        neg_charge,
        rigid_ratio,
        kappa1,
        kappa2,
        kappa3,
        chi0v,
        chi1v,
        chi2v,
    ]
    graph_attr = torch.tensor(graph_attr, dtype=torch.float).unsqueeze(0)
    graph_attr = torch.cat([graph_attr, tensor_fp], dim=1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, graph_attr=graph_attr)


def drug_collate_fn(batch):
    return Batch.from_data_list(batch)


def itc_collate_fn(batch):
    drug1, drug2, label = zip(*batch)
    return torch.stack(drug1), torch.stack(drug2), torch.stack(label)


def constrained_sample(
    pool: set, deg: dict, cluster_labels: dict, n_sample: int, k: float = 0.6, seed=42
) -> set:
    rng = np.random.RandomState(seed)
    norm_pool = list(pool)

    target_cluster_dist = defaultdict(float)
    for d1, d2 in norm_pool:
        target_cluster_dist[cluster_labels[d1]] += 0.5
        target_cluster_dist[cluster_labels[d2]] += 0.5
    total = sum(target_cluster_dist.values())
    target_dist = {c: v / total for c, v in target_cluster_dist.items()}

    drug_list = list(deg.keys())
    drug_to_idx = {d: i for i, d in enumerate(drug_list)}
    probs = np.array([1.0 / (deg[d] ** k) for d in drug_list])
    probs /= probs.sum()

    drug2pairs = defaultdict(list)
    for pair in norm_pool:
        drug2pairs[pair[0]].append(pair)
        drug2pairs[pair[1]].append(pair)

    avg_count = 2 * n_sample / len(drug_list)
    max_allowed = avg_count * 1.5

    drug_count = np.zeros(len(drug_list), dtype=np.int32)
    cluster_count = defaultdict(int)
    sampled_set = set()  # 存储规范化 tuple

    active_mask = np.ones(len(drug_list), dtype=bool)
    active_probs = probs.copy()

    while len(sampled_set) < n_sample:
        if not active_mask.any():
            break

        d1_idx = rng.choice(len(drug_list), p=active_probs / active_probs.sum())
        d1 = drug_list[d1_idx]

        candidates = drug2pairs.get(d1, [])
        if not candidates:
            continue

        # 批量过滤：排除已在sampled中的 + d2已饱和的
        valid_pairs = []
        for pair in candidates:
            if pair in sampled_set:
                continue
            d2 = pair[1] if pair[0] == d1 else pair[0]
            d2_idx = drug_to_idx[d2]
            if drug_count[d2_idx] >= max_allowed:
                continue
            valid_pairs.append(pair)

        if not valid_pairs:
            # 如果该drug所有候选都无效，标记为非活跃
            active_mask[d1_idx] = False
            active_probs[d1_idx] = 0
            continue

        # 向量化打分替代 Python 循环
        vp_arr = np.array(valid_pairs)
        d2s = np.where(vp_arr[:, 0] == d1, vp_arr[:, 1], vp_arr[:, 0])

        c1 = cluster_labels[d1]
        c2s = [cluster_labels[d2] for d2 in d2s]

        current_total = len(sampled_set) * 2 + 2
        c1_ratio = (cluster_count[c1] + 1) / current_total
        c2_ratios = np.array([(cluster_count[c] + 1) / current_total for c in c2s])

        scores = -np.abs(c1_ratio - target_dist[c1]) - np.abs(
            c2_ratios - np.array([target_dist[c] for c in c2s])
        )

        best_idx = np.argmax(scores)
        best_pair = valid_pairs[best_idx]

        # 更新状态
        sampled_set.add(best_pair)
        d2 = best_pair[1] if best_pair[0] == d1 else best_pair[0]

        d1_i, d2_i = drug_to_idx[d1], drug_to_idx[d2]
        drug_count[d1_i] += 1
        drug_count[d2_i] += 1
        cluster_count[cluster_labels[d1]] += 1
        cluster_count[cluster_labels[d2]] += 1

        # 检查是否需要禁用药物
        if drug_count[d1_i] >= max_allowed:
            active_mask[d1_i] = False
            active_probs[d1_i] = 0
        if drug_count[d2_i] >= max_allowed:
            active_mask[d2_i] = False
            active_probs[d2_i] = 0

    return sampled_set


def fingerprint_balanced_negative_sampling(
    fp_dict,
    pending_pairs,
    neg_pairs,
    cluster_labels,
    pos_neg_ratio=1.0,
    difficulty_ratio=(0.3, 0.4, 0.3),
    k=0.6,
    seed=42,
):
    deg = defaultdict(int)
    for a, b in pending_pairs:
        deg[a] += 1
        deg[b] += 1
    for d in list(fp_dict.keys()):
        if d not in deg:
            deg[d] = 1

    easy_pool, medium_pool, hard_pool = split_with_sim(fp_dict, neg_pairs)

    total_neg = int(len(pending_pairs) * pos_neg_ratio)
    n_easy = int(total_neg * difficulty_ratio[0])
    n_medium = int(total_neg * difficulty_ratio[1])
    n_hard = total_neg - n_easy - n_medium

    neg_easy = constrained_sample(easy_pool, deg, cluster_labels, n_easy, k, seed)

    neg_medium = constrained_sample(medium_pool, deg, cluster_labels, n_medium, k, seed)
    neg_hard = constrained_sample(hard_pool, deg, cluster_labels, n_hard, k, seed)

    return neg_easy | neg_medium | neg_hard


def get_all_pairs(id_list: list) -> set:
    all_pairs = set()
    for i in range(len(id_list)):
        for j in range(i + 1, len(id_list)):
            all_pairs.add(tuple(sorted((id_list[i], id_list[j]))))
    return all_pairs


def filte_false_neg(fp_dict: dict, pairs: set, sim_threshold: float) -> set:
    remaining = set()
    for pair in pairs:
        d1, d2 = pair
        sim = DataStructs.TanimotoSimilarity(fp_dict[d1], fp_dict[d2])
        if sim >= sim_threshold:
            continue
        remaining.add(pair)
    return remaining


def generate_fp(drug_dict: dict) -> dict:
    fp_dict = {}
    morgan_gen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    for id, smile in drug_dict.items():
        mol = Chem.MolFromSmiles(smile)
        fp_dict[id] = morgan_gen.GetFingerprint(mol)
    return fp_dict


def split_with_sim(
    fp_dict: dict,
    pairs: set,
):
    pairs_with_sim = []
    for pair in pairs:
        d1, d2 = pair
        sim = DataStructs.TanimotoSimilarity(fp_dict[d1], fp_dict[d2])
        pairs_with_sim.append((sim, pair))
    pairs_with_sim.sort(key=lambda x: x[0])
    n_pairs = len(pairs_with_sim)
    t1 = int(n_pairs * 0.3)
    t2 = int(n_pairs * 0.7)
    easy_pool = pairs_with_sim[:t1]
    medium_pool = pairs_with_sim[t1:t2]
    hard_pool = pairs_with_sim[t2:]
    easy_pool = [pair for _, pair in easy_pool]
    medium_pool = [pair for _, pair in medium_pool]
    hard_pool = [pair for _, pair in hard_pool]
    return set(easy_pool), set(medium_pool), set(hard_pool)


def butina_cluster(id_list: list, fp_dict: dict, dist_threshold=0.6):
    n = len(id_list)
    dists = []
    for i in range(1, n):
        for j in range(i):
            dist = 1 - DataStructs.TanimotoSimilarity(
                fp_dict[id_list[i]], fp_dict[id_list[j]]
            )
            dists.append(dist)
    cs = Butina.ClusterData(dists, n, dist_threshold, isDistData=True)
    cluster_labels = dict()
    for cluster_id, idxs in enumerate(cs):
        for idx in idxs:
            cluster_labels[id_list[idx]] = cluster_id
    return cluster_labels


def random_neg_sampling(neg_pairs_pool: set, num_sample: int, seed: int):
    random.seed(seed)
    if num_sample >= len(neg_pairs_pool):
        return set(neg_pairs_pool)
    return set(random.sample(list(neg_pairs_pool), num_sample))


def generate_pre_neg_sample(
    data_source,
    split_type,
    seed=42,
    dist_threshold=0.6,
    sim_threshold=0.7,
    pos_neg_ratio=1,
    difficulty_ratio=(0.3, 0.4, 0.3),
    k=0.6,
):
    base_dir = os.path.join(
        "./split_data", data_source + "-" + split_type + "-" + str(seed), "raw"
    )

    drugs = pd.read_csv(os.path.join(base_dir, "drug.csv"))
    all_itc = pd.read_csv(os.path.join(base_dir, "all_itc.csv"))
    train_itc = pd.read_csv(os.path.join(base_dir, "train_itc.csv"))
    val_itc = pd.read_csv(os.path.join(base_dir, "val_itc.csv"))
    test_itc = pd.read_csv(os.path.join(base_dir, "test_itc.csv"))

    drug_dict = {i: s for i, s in zip(drugs["id"], drugs["smile"])}

    all_pos_pairs = {
        tuple(sorted((d1, d2))) for d1, d2 in zip(all_itc["drug1"], all_itc["drug2"])
    }
    train_pos_pairs = {
        tuple(sorted((d1, d2)))
        for d1, d2 in zip(train_itc["drug1"], train_itc["drug2"])
    }

    all_pairs = get_all_pairs(list(drug_dict.keys()))
    all_neg_pairs = all_pairs - all_pos_pairs
    fp_dict = generate_fp(drug_dict)
    fn_filted_neg_pairs = filte_false_neg(fp_dict, all_neg_pairs, sim_threshold)
    cluster_labels = butina_cluster(list(drug_dict.keys()), fp_dict, dist_threshold)
    train_neg_pairs = fingerprint_balanced_negative_sampling(
        fp_dict,
        train_pos_pairs,
        fn_filted_neg_pairs,
        cluster_labels,
        pos_neg_ratio,
        difficulty_ratio,
        k,
        seed,
    )
    rest_neg_pairs = fn_filted_neg_pairs - train_neg_pairs
    val_neg_pairs = random_neg_sampling(rest_neg_pairs, len(val_itc), seed)
    rest_neg_pairs = rest_neg_pairs - val_neg_pairs
    test_neg_pairs = random_neg_sampling(rest_neg_pairs, len(test_itc), seed)

    train_neg = pd.DataFrame(
        list(train_neg_pairs),
        columns=["drug1", "drug2"],
    )
    val_neg = pd.DataFrame(
        list(val_neg_pairs),
        columns=["drug1", "drug2"],
    )
    test_neg = pd.DataFrame(
        list(test_neg_pairs),
        columns=["drug1", "drug2"],
    )
    train_itc["label"] = 1
    val_itc["label"] = 1
    test_itc["label"] = 1
    train_neg["label"] = 0
    val_neg["label"] = 0
    test_neg["label"] = 0

    train = pd.concat([train_itc, train_neg], axis=0, ignore_index=True)
    val = pd.concat([val_itc, val_neg], axis=0, ignore_index=True)
    test = pd.concat([test_itc, test_neg], axis=0, ignore_index=True)
    train.to_csv(os.path.join(base_dir, "train_pre_itc.csv"), index=False)
    val.to_csv(os.path.join(base_dir, "val_pre_itc.csv"), index=False)
    test.to_csv(os.path.join(base_dir, "test_pre_itc.csv"), index=False)
    print(
        f"数据集生成完成：训练集{len(train)}条，验证集{len(val)}条，测试集{len(test)}条"
    )


__all__ = ["Timer", "split_data", "itc_collate_fn", "drug_collate_fn"]

if __name__ == "__main__":
    # split_data()
    generate_pre_neg_sample("drugbank", "random", seed=42)
