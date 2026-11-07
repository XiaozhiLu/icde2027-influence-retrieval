# Data

The data itself is not redistributed. Download from the official sources and
place the files as described below.

## SIFT (1M / 10M / 50M / 100M prefixes)

- Source: http://corpus-texmex.irisa.fr/ (SIFT1M) and the SIFT100M release
  used in the paper (uint8 `base.u8bin`, `queries.u8bin`).
- After download, convert the corpus to the uint8 binary format expected by
  `stage3_u8_main.cpp` and build the index once with the serial builder; the
  checkpoints (`*.ckpt<size>`) are prefix snapshots of the same trajectory.

## CiaoDVD

- Source: https://guoguibing.github.io/librec/datasets.html (CiaoDVD.zip)
- Files needed: `movie-ratings.txt`, `trusts.txt`
- Preprocess: `scripts/preprocess_ciaodvd.py` (BPR-MF, dim 64, 20 epochs,
  leave-one-out split).

## KuaiRand-1K

- Source: https://zenodo.org/records/10439422 (`KuaiRand-1K.tar.gz`, 1.1 GB)
- Files needed: `log_standard_4_08_to_4_21_1k.csv`,
  `log_standard_4_22_to_5_08_1k.csv`, `log_random_4_22_to_5_08_1k.csv`
- Preprocess: `scripts/prepare_kuairand1k_itemgraph_ctr.py`
  (`--kuairand_root` points to the directory that contains `data/`).
