# Reproducing the reported results

All experiments use `OMP_NUM_THREADS=1` (the graph construction is serial, so
multi-threading would change the trajectories).

## 0. Data

See `data/README.md`. Expected layout:

```
data/
  queries.u8bin                # 100 query vectors, uint8
  sift100m.idx                 # 100M index (see data/README for construction)
  sift100m.idx.ckpt1000000     # 1M prefix checkpoint
  gt_1m.bin, gt_100m.bin       # exact ground truth, K=10
  gt_50_1m.bin, gt_50_100m.bin # exact ground truth, K=50
```

## 1. Table 2 (SIFT, K=10 and K=50)

```bash
# ours
bin/stage3u8 0 100 --query=data/queries.u8bin --K=10 --ef=300 \
  --method=ours --gt=data/gt_1m.bin --load=data/sift100m.idx.ckpt1000000
# greedy baseline
bin/stage3u8 0 100 --query=data/queries.u8bin --K=10 --ef=300 \
  --method=greedy-rep --gt=data/gt_1m.bin --load=data/sift100m.idx.ckpt1000000
# HNSW baseline only
bin/stage3u8 0 100 --query=data/queries.u8bin --K=10 --ef=300 \
  --baseline-only --gt=data/gt_1m.bin --load=data/sift100m.idx.ckpt1000000
```

Replace `gt_1m.bin` with `gt_100m.bin` and the index with `data/sift100m.idx`
for the 100M column; use `--K=50` with `gt_50_*.bin` for the K=50 block.

## 2. Table 3 (CiaoDVD)

```bash
python3 scripts/preprocess_ciaodvd.py \
  --movie_ratings data/raw/ciaodvd/movie-ratings.txt \
  --trusts data/raw/ciaodvd/trusts.txt \
  --outdir data/processed/ciaodvd

bin/ciaodvd --dir=data/processed/ciaodvd --K=10 --method=base
bin/ciaodvd --dir=data/processed/ciaodvd --K=10 --method=greedy --tau=0.7
bin/ciaodvd --dir=data/processed/ciaodvd --K=10 --method=ours   --tau=0.7
bin/ciaodvd --dir=data/processed/ciaodvd --K=10 --method=celf   --celf=near
```

## 3. Table 4 (KuaiRand-1K)

```bash
python3 scripts/prepare_kuairand1k_itemgraph_ctr.py \
  --kuairand_root data/raw/kuairand1k/KuaiRand-1K \
  --outdir data/processed/kuairand1k_itemgraph_ctr --dim 64 --epochs 20

bin/kuairand data/processed/kuairand1k_itemgraph_ctr
```

## 4. Figures

```bash
python3 scripts/make_figs.py   # writes FIG2_scale.pdf and FIG3_ciao.pdf
```
(edit the output directory at the top of the script)
