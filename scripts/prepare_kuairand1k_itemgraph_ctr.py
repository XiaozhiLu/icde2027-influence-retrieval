import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import struct
import random
import argparse
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim



class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, users, pos_items, neg_items):
        pu = self.user_emb(users)
        qi = self.item_emb(pos_items)
        qj = self.item_emb(neg_items)

        pos_scores = torch.sum(pu * qi, dim=1)
        neg_scores = torch.sum(pu * qj, dim=1)
        return pos_scores, neg_scores


def build_user_train_dict(train_pairs):
    d = defaultdict(set)
    for u, i in train_pairs:
        d[u].add(i)
    return d


def sample_bpr_batch(train_pairs, user_train_dict, n_items, batch_size):
    users, pos_items, neg_items = [], [], []
    for _ in range(batch_size):
        u, i = train_pairs[random.randint(0, len(train_pairs) - 1)]
        while True:
            j = random.randint(0, n_items - 1)
            if j not in user_train_dict[u]:
                break
        users.append(u)
        pos_items.append(i)
        neg_items.append(j)

    return (
        torch.LongTensor(users),
        torch.LongTensor(pos_items),
        torch.LongTensor(neg_items),
    )


def train_bprmf(train_pairs, n_users, n_items, dim=64, epochs=20, batch_size=2048, lr=1e-3, reg=1e-4, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BPRMF(n_users, n_items, dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    user_train_dict = build_user_train_dict(train_pairs)
    steps_per_epoch = max(1, len(train_pairs) // batch_size)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            users, pos_items, neg_items = sample_bpr_batch(train_pairs, user_train_dict, n_items, batch_size)
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            optimizer.zero_grad()
            pos_scores, neg_scores = model(users, pos_items, neg_items)

            bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12).mean()
            reg_loss = (
                model.user_emb(users).pow(2).sum(dim=1).mean() +
                model.item_emb(pos_items).pow(2).sum(dim=1).mean() +
                model.item_emb(neg_items).pow(2).sum(dim=1).mean()
            )

            loss = bpr_loss + reg * reg_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[BPR Epoch {epoch:02d}] loss = {total_loss / steps_per_epoch:.6f}")

    return model.cpu()



class LogisticCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return torch.sigmoid(self.a * x + self.b)


def fit_calibrator(raw_scores, labels, epochs=300, lr=0.05):
    x = torch.tensor(raw_scores, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    model = LogisticCalibrator()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        p = model(x)
        loss = nn.BCELoss()(p, y)
        loss.backward()
        optimizer.step()

    return float(model.a.detach().cpu().item()), float(model.b.detach().cpu().item())


def fit_calibrator_with_sampling(raw_scores, labels, max_samples=200000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(labels)
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        raw_scores = raw_scores[idx]
        labels = labels[idx]
    return fit_calibrator(raw_scores, labels, epochs=300, lr=0.05)


def save_embedding_bin(path, arr):
    arr = np.asarray(arr, dtype=np.float32)
    n, d = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("i", n))
        f.write(struct.pack("i", d))
        f.write(arr.tobytes())


def save_edges(path, edges):
    with open(path, "w", encoding="utf-8") as f:
        for a, b in edges:
            f.write(f"{a} {b}\n")


def save_eval_impressions(path, df):
    with open(path, "w", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            f.write(f"{row.user_id_mapped} {row.video_id_mapped} {int(row.is_click)}\n")


def save_calibration(path, a, b):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{a} {b}\n")


def build_item_useragg_emb(train_click_df, user_emb, n_items, dim):
    item_users = defaultdict(list)
    for row in train_click_df.itertuples(index=False):
        item_users[row.video_id_mapped].append(row.user_id_mapped)

    item_useragg_emb = np.zeros((n_items, dim), dtype=np.float32)

    for it in range(n_items):
        users = item_users.get(it, [])
        if len(users) == 0:
            continue
        item_useragg_emb[it] = user_emb[np.array(users)].mean(axis=0)

    return item_useragg_emb



def build_item_coclick_graph(train_click_df, top_m=20):
    user_items = defaultdict(list)
    for row in train_click_df.itertuples(index=False):
        user_items[row.user_id_mapped].append(row.video_id_mapped)

    pair_counter = Counter()

    for _, items in user_items.items():
        items = sorted(set(items))
        if len(items) < 2:
            continue
        for a, b in combinations(items, 2):
            pair_counter[(a, b)] += 1
            pair_counter[(b, a)] += 1

    neighbor_map = defaultdict(list)
    for (a, b), w in pair_counter.items():
        neighbor_map[a].append((b, w))

    edges = []
    for a, arr in neighbor_map.items():
        arr.sort(key=lambda x: (-x[1], x[0]))
        for b, _ in arr[:top_m]:
            edges.append((a, b))

    return edges


def l2_normalize(mat):
    norm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norm


def build_mode2_fused_item_emb(item_emb, item_useragg_emb, beta=0.5):
    item_norm = l2_normalize(item_emb)
    agg_norm = l2_normalize(item_useragg_emb)
    fused = beta * item_norm + (1.0 - beta) * agg_norm
    fused = l2_normalize(fused)
    return fused.astype(np.float32)




def compute_mode1_scores(user_emb, item_emb, item_useragg_emb, users, items, alpha):
    s1 = np.sum(user_emb[users] * item_emb[items], axis=1)
    s2 = np.sum(user_emb[users] * item_useragg_emb[items], axis=1)
    return alpha * s1 + (1.0 - alpha) * s2


def compute_mode2_scores(user_emb, fused_item_emb, users, items):
    user_norm = user_emb[users]
    user_norm = user_norm / (np.linalg.norm(user_norm, axis=1, keepdims=True) + 1e-12)
    return np.sum(user_norm * fused_item_emb[items], axis=1)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kuairand_root", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)

    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--item_graph_top_m", type=int, default=20)

    parser.add_argument("--alpha_mode1", type=float, default=0.5)
    parser.add_argument("--beta_mode2", type=float, default=0.5)
    parser.add_argument("--max_calib_samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    data_dir = os.path.join(args.kuairand_root, "data")

    train_log = os.path.join(data_dir, "log_standard_4_08_to_4_21_1k.csv")
    calib_log = os.path.join(data_dir, "log_standard_4_22_to_5_08_1k.csv")
    test_log = os.path.join(data_dir, "log_random_4_22_to_5_08_1k.csv")

    usecols = ["user_id", "video_id", "time_ms", "is_click"]

    print("Loading KuaiRand-1K logs...")
    df_train = pd.read_csv(train_log, usecols=usecols)
    df_calib = pd.read_csv(calib_log, usecols=usecols)
    df_test = pd.read_csv(test_log, usecols=usecols)

    # 仍然限制到 random test 出现过的 candidate pool item
    candidate_items = set(df_test["video_id"].unique().tolist())
    df_train = df_train[df_train["video_id"].isin(candidate_items)].copy()
    df_calib = df_calib[df_calib["video_id"].isin(candidate_items)].copy()

    user_ids = sorted(set(df_train["user_id"].tolist()) |
                      set(df_calib["user_id"].tolist()) |
                      set(df_test["user_id"].tolist()))
    item_ids = sorted(candidate_items)

    user2id = {u: i for i, u in enumerate(user_ids)}
    item2id = {it: i for i, it in enumerate(item_ids)}

    for df in [df_train, df_calib, df_test]:
        df["user_id_mapped"] = df["user_id"].map(user2id).astype(np.int32)
        df["video_id_mapped"] = df["video_id"].map(item2id).astype(np.int32)

    n_users = len(user2id)
    n_items = len(item2id)

    print(f"n_users={n_users}, n_items(candidate_pool)={n_items}")
    print(f"train impressions={len(df_train)}, calib impressions={len(df_calib)}, test(random) impressions={len(df_test)}")

    # train clicks
    train_click_df = df_train[df_train["is_click"] == 1][["user_id_mapped", "video_id_mapped"]].drop_duplicates()
    train_pairs = list(train_click_df.itertuples(index=False, name=None))
    print(f"train clicked pairs={len(train_pairs)}")

    print("Training BPR-MF...")
    model = train_bprmf(
        train_pairs=train_pairs,
        n_users=n_users,
        n_items=n_items,
        dim=args.dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        reg=args.reg,
        seed=args.seed
    )

    user_emb = model.user_emb.weight.detach().cpu().numpy().astype(np.float32)
    item_emb = model.item_emb.weight.detach().cpu().numpy().astype(np.float32)

    print("Building item user-aggregate embedding...")
    item_useragg_emb = build_item_useragg_emb(train_click_df, user_emb, n_items, args.dim)

    print("Building item co-click graph...")
    item_graph_edges = build_item_coclick_graph(train_click_df, top_m=args.item_graph_top_m)

    print("Building mode2 fused item embedding...")
    fused_item_emb_mode2 = build_mode2_fused_item_emb(item_emb, item_useragg_emb, beta=args.beta_mode2)

    print("Fitting calibrators...")
    calib_users = df_calib["user_id_mapped"].to_numpy(dtype=np.int32)
    calib_items = df_calib["video_id_mapped"].to_numpy(dtype=np.int32)
    calib_labels = df_calib["is_click"].to_numpy(dtype=np.float32)

    raw_mode1 = compute_mode1_scores(
        user_emb, item_emb, item_useragg_emb,
        calib_users, calib_items, args.alpha_mode1
    )
    a1, b1 = fit_calibrator_with_sampling(raw_mode1, calib_labels, max_samples=args.max_calib_samples, seed=args.seed)

    raw_mode2 = compute_mode2_scores(
        user_emb, fused_item_emb_mode2,
        calib_users, calib_items
    )
    a2, b2 = fit_calibrator_with_sampling(raw_mode2, calib_labels, max_samples=args.max_calib_samples, seed=args.seed + 1)

    print(f"mode1 calibrator: a={a1:.6f}, b={b1:.6f}")
    print(f"mode2 calibrator: a={a2:.6f}, b={b2:.6f}")

    print("Exporting files...")
    save_embedding_bin(os.path.join(args.outdir, "user_emb.bin"), user_emb)
    save_embedding_bin(os.path.join(args.outdir, "item_emb.bin"), item_emb)
    save_embedding_bin(os.path.join(args.outdir, "item_useragg_emb.bin"), item_useragg_emb)
    save_embedding_bin(os.path.join(args.outdir, "fused_item_emb_mode2.bin"), fused_item_emb_mode2)

    save_edges(os.path.join(args.outdir, "item_graph_edges.txt"), item_graph_edges)
    save_eval_impressions(os.path.join(args.outdir, "eval_impressions_random.txt"),
                          df_test[["user_id_mapped", "video_id_mapped", "is_click"]])

    save_calibration(os.path.join(args.outdir, "calibration_mode1.txt"), a1, b1)
    save_calibration(os.path.join(args.outdir, "calibration_mode2.txt"), a2, b2)

    with open(os.path.join(args.outdir, "config.txt"), "w", encoding="utf-8") as f:
        f.write(f"alpha_mode1 {args.alpha_mode1}\n")
        f.write(f"beta_mode2 {args.beta_mode2}\n")

    with open(os.path.join(args.outdir, "stats.txt"), "w", encoding="utf-8") as f:
        f.write(f"n_users {n_users}\n")
        f.write(f"n_items {n_items}\n")
        f.write(f"train_impressions {len(df_train)}\n")
        f.write(f"calib_impressions {len(df_calib)}\n")
        f.write(f"test_impressions {len(df_test)}\n")
        f.write(f"train_clicked_pairs {len(train_pairs)}\n")
        f.write(f"item_graph_edges {len(item_graph_edges)}\n")

    print("Done.")


if __name__ == "__main__":
    main()