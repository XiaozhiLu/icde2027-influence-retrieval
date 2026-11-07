# Recall-Constrained Influence-Aware User Retrieval

Reference implementation and experiment logs for the ICDE 2027 submission
"Recall-Constrained Influence-Aware User Retrieval for Product Recommendation".

## What this repository contains

| Path | Content |
|---|---|
| `src/` | C++ implementation (HNSW-based two-stage retrieval, uint8 and float pipelines) |
| `scripts/` | Python preprocessing for CiaoDVD / KuaiRand-1K and figure generation |
| `results/logs/` | Raw stdout of every run reported in the paper |
| `data/` | Dataset download instructions (data itself is not redistributed) |

## Method at a glance

1. **Anchoring** — the pure-similarity Top-K is kept as the recall reference.
2. **Candidate pool** — hybrid matching-influence scoring with a progress-aware
   alpha, plus a degree-ordered influencer list.
3. **Replacement** — 1v1 substitution, oversampled pool, 2v2 fallback, all
   bounded by a proxy-overlap floor calibrated offline against exact GT.
4. **Eligibility** — only the furthest ceil(K(1-tau)) anchors may be replaced.

## Build

```bash
g++ -O3 -march=native -std=c++17 -fopenmp -static -I src \
    src/stage3_u8_main.cpp -o bin/stage3u8
g++ -O3 -march=native -std=c++17 -fopenmp -static -I src \
    src/ciaodvd_main.cpp -o bin/ciaodvd
g++ -O3 -march=native -std=c++17 -fopenmp -static -I src \
    src/KuaiRand_industrial_validation.cpp -o bin/kuairand
```

See `REPRODUCE.md` for the exact commands behind each table.

## Citation

If you use this code, please cite the paper (bibtex will be added after publication).

## License

Research use only. Contact the authors for other uses.
