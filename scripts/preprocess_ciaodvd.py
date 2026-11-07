import argparse
import csv
import math
import os
from collections import defaultdict
from datetime import datetime
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Preprocess CiaoDVD for item-to-user retrieval experiments.')
    p.add_argument('--movie_ratings', type=str, required=True, help='Path to movie-ratings.txt')
    p.add_argument('--trusts', type=str, required=True, help='Path to trusts.txt')
    p.add_argument('--outdir', type=str, required=True, help='Output directory')
    p.add_argument('--rating_threshold', type=float, default=4.0, help='Positive rating threshold')
    p.add_argument('--dim', type=int, default=64, help='Embedding dimension')
    p.add_argument('--epochs', type=int, default=20, help='BPR epochs')
    p.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    p.add_argument('--reg', type=float, default=1e-4, help='L2 regularization')
    p.add_argument('--seed', type=int, default=2026, help='Random seed')
    p.add_argument('--min_test_users_per_item', type=int, default=1, help='Minimum held-out positives for an item to become a query')
    return p.parse_args()


def parse_date(s: str):
    s = s.strip()
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y'):
        try:
            return datetime.strptime(s, fmt).toordinal()
        except ValueError:
            pass
    return 0


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_matrix_bin(path: str, mat: np.ndarray):
    mat = np.asarray(mat, dtype=np.float32)
    with open(path, 'wb') as f:
        np.array([mat.shape[0], mat.shape[1]], dtype=np.int32).tofile(f)
        mat.tofile(f)


def normalize_rows(mat: np.ndarray):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return mat / norms


def load_positive_ratings(path: str, threshold: float):
    user_hist = defaultdict(list)
    items_seen = set()
    users_seen = set()
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            user_id = row[0].strip()
            item_id = row[1].strip()
            rating = float(row[4])
            date_ord = parse_date(row[5])
            users_seen.add(user_id)
            items_seen.add(item_id)
            if rating >= threshold:
                user_hist[user_id].append((date_ord, item_id, rating))
    for u in user_hist:
        user_hist[u].sort(key=lambda x: x[0])
    return user_hist, users_seen, items_seen


def split_leave_one_out(user_hist):
    train_pairs = []
    test_pairs = []
    for u, hist in user_hist.items():
        if len(hist) >= 2:
            for entry in hist[:-1]:
                train_pairs.append((u, entry[1], entry[0]))
            last = hist[-1]
            test_pairs.append((u, last[1], last[0]))
        elif len(hist) == 1:
            entry = hist[0]
            train_pairs.append((u, entry[1], entry[0]))
    return train_pairs, test_pairs


def build_id_maps(train_pairs, test_pairs):
    users = sorted({u for u, _, _ in train_pairs} | {u for u, _, _ in test_pairs})
    items = sorted({i for _, i, _ in train_pairs} | {i for _, i, _ in test_pairs})
    u2id = {u: idx for idx, u in enumerate(users)}
    i2id = {i: idx for idx, i in enumerate(items)}
    return u2id, i2id


def remap_pairs(pairs, u2id, i2id):
    out = []
    for u, i, t in pairs:
        if u in u2id and i in i2id:
            out.append((u2id[u], i2id[i], t))
    return out


def load_trusts(path: str, u2id):
    edges = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            u = row[0].strip()
            v = row[1].strip()
            if u in u2id and v in u2id and u != v:
                edges.append((u2id[u], u2id[v]))
    edges = sorted(set(edges))
    return edges


def build_train_sets(train_pairs_remap, num_users):
    user_pos = [set() for _ in range(num_users)]
    for u, i, _ in train_pairs_remap:
        user_pos[u].add(i)
    active_users = [u for u in range(num_users) if user_pos[u]]
    return user_pos, active_users


def train_bpr(num_users, num_items, user_pos, active_users, dim, epochs, lr, reg, seed):
    rng = np.random.default_rng(seed)
    user_emb = 0.01 * rng.standard_normal((num_users, dim), dtype=np.float32)
    item_emb = 0.01 * rng.standard_normal((num_items, dim), dtype=np.float32)

    n_samples = sum(len(s) for s in user_pos)
    if n_samples == 0:
        raise RuntimeError('No positive training samples found.')

    active_users_arr = np.array(active_users, dtype=np.int32)

    for epoch in range(epochs):
        total_loss = 0.0
        for _ in range(n_samples):
            u = int(active_users_arr[rng.integers(0, len(active_users_arr))])
            pos_items = list(user_pos[u])
            i = int(pos_items[rng.integers(0, len(pos_items))])
            j = int(rng.integers(0, num_items))
            while j in user_pos[u]:
                j = int(rng.integers(0, num_items))

            pu = user_emb[u]
            qi = item_emb[i]
            qj = item_emb[j]
            x = float(np.dot(pu, qi - qj))
            # grad of -log(sigmoid(x))
            g = 1.0 / (1.0 + math.exp(x))
            total_loss += -math.log(1.0 / (1.0 + math.exp(-x)) + 1e-12)

            pu_old = pu.copy()
            qi_old = qi.copy()
            qj_old = qj.copy()

            user_emb[u] += lr * (g * (qi_old - qj_old) - reg * pu_old)
            item_emb[i] += lr * (g * pu_old - reg * qi_old)
            item_emb[j] += lr * (-g * pu_old - reg * qj_old)

        print(f'Epoch {epoch+1}/{epochs} loss={total_loss / n_samples:.6f}')

    return normalize_rows(user_emb.astype(np.float32)), normalize_rows(item_emb.astype(np.float32))


def write_edges(path, edges):
    with open(path, 'w', encoding='utf-8') as f:
        for u, v in edges:
            f.write(f'{u} {v}\n')


def write_query_positives(outdir, test_pairs_remap, min_test_users_per_item):
    item_pos = defaultdict(list)
    for u, i, _ in test_pairs_remap:
        item_pos[i].append(u)

    query_items = sorted([i for i, users in item_pos.items() if len(users) >= min_test_users_per_item])

    with open(os.path.join(outdir, 'item_test_positives.txt'), 'w', encoding='utf-8') as f:
        for i in query_items:
            users = sorted(set(item_pos[i]))
            f.write(str(i) + ' ' + ' '.join(map(str, users)) + '\n')

    with open(os.path.join(outdir, 'query_items.txt'), 'w', encoding='utf-8') as f:
        for i in query_items:
            f.write(f'{i}\n')

    return query_items


def write_mapping(path, mapping):
    with open(path, 'w', encoding='utf-8') as f:
        for raw_id, new_id in sorted(mapping.items(), key=lambda x: x[1]):
            f.write(f'{new_id}\t{raw_id}\n')


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    user_hist, _, _ = load_positive_ratings(args.movie_ratings, args.rating_threshold)
    train_pairs, test_pairs = split_leave_one_out(user_hist)
    u2id, i2id = build_id_maps(train_pairs, test_pairs)

    train_pairs_remap = remap_pairs(train_pairs, u2id, i2id)
    test_pairs_remap = remap_pairs(test_pairs, u2id, i2id)
    trust_edges = load_trusts(args.trusts, u2id)

    num_users = len(u2id)
    num_items = len(i2id)
    user_pos, active_users = build_train_sets(train_pairs_remap, num_users)

    print(f'num_users={num_users}, num_items={num_items}, train_pos={len(train_pairs_remap)}, test_pos={len(test_pairs_remap)}, trust_edges={len(trust_edges)}')

    user_emb, item_emb = train_bpr(
        num_users=num_users,
        num_items=num_items,
        user_pos=user_pos,
        active_users=active_users,
        dim=args.dim,
        epochs=args.epochs,
        lr=args.lr,
        reg=args.reg,
        seed=args.seed,
    )

    save_matrix_bin(os.path.join(args.outdir, 'user_emb.bin'), user_emb)
    save_matrix_bin(os.path.join(args.outdir, 'item_emb.bin'), item_emb)
    write_edges(os.path.join(args.outdir, 'trust_edges.txt'), trust_edges)
    query_items = write_query_positives(args.outdir, test_pairs_remap, args.min_test_users_per_item)
    write_mapping(os.path.join(args.outdir, 'user_id_map.txt'), u2id)
    write_mapping(os.path.join(args.outdir, 'item_id_map.txt'), i2id)

    with open(os.path.join(args.outdir, 'stats.txt'), 'w', encoding='utf-8') as f:
        f.write(f'num_users={num_users}\n')
        f.write(f'num_items={num_items}\n')
        f.write(f'train_pos={len(train_pairs_remap)}\n')
        f.write(f'test_pos={len(test_pairs_remap)}\n')
        f.write(f'trust_edges={len(trust_edges)}\n')
        f.write(f'num_query_items={len(query_items)}\n')

    print('Done. Files written to', args.outdir)


if __name__ == '__main__':
    main()
