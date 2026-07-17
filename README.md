# Bird Autoencoder

Phase 2 concept/residual experiments are implemented in `main_factorized.py`.
They use the official CUB split, certainty-weighted image attributes, whole
attribute-group selection, a fixed `[8,8,15]` residual, matched continuous
controls, hard/soft concept diagnostics, group interventions, optional CUB
part-landmark ROIs, and residual-to-concept probes.

See [ATTRIBUTE_EXPERIMENTS.md](ATTRIBUTE_EXPERIMENTS.md) for the exact staged
experiment protocol.  Phase 1 topology experiments remain available through
`main_experiment.py` and are not changed by the Phase 2 runner.

Controlled autoencoder experiments on CUB-200-2011 for studying how latent
topology, global mixing, compression, and decoder accessibility affect image
reconstruction.

The current evidence does **not** support the claim that vectors are inherently
worse than spatial tensors. A parameter-free
`8x8xC -> Flatten -> [B, K] -> Reshape -> 8x8xC` interface preserves the same
ordered information. The working empirical hypothesis is:

```text
unstructured global vector < structured vector ≈ spatial map
```

The project uses 64x64 full images as a controlled setting, does not use
bounding-box crops, and does not add encoder-decoder skip connections.

## Current experiment families

- `residual_lite`: legacy globally mixed dense-vector baseline.
- `spatial_lite`: spatial latent reference.
- `structured_vector_lite`: rank-2 vector interface with fixed spatial order.
- `bottleneck_ablation`: identity, fixed permutation, fixed/trainable global
  mixing, global compression, and spatial channel compression controls.

The next research phase will add a factorized single-vector representation
`z=[c;m]`, with supervised concepts `c` and a structured reconstruction
residual `m`. That model is intentionally deferred until the CUB attribute and
official-split pipeline is defined.

## Repository structure

```text
bird-autoencoder/
├── main_experiment.py
├── data.py
├── losses.py
├── train_utils.py
├── visualize.py
├── evaluate.py
├── aggregate_results.py
├── model/
│   ├── model_common.py
│   ├── model_residual_lite.py
│   ├── model_spatial_lite.py
│   ├── model_structured_vector_lite.py
│   └── model_bottleneck_ablation.py
├── configs/
│   ├── topology_ablation.json
│   ├── structured_comparison.json
│   └── loss_ablation.json
└── analysis/
    ├── aggregate_seeds.py
    └── make_paper_tables.py
```

Legacy models and entry points remain available while the experiment runner is
migrated; they should not be used to infer the current research conclusions.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy a config and replace its dataset/output paths. The dataset path must point
to the CUB `images/` directory.

## Run controlled experiments

```bash
python main_experiment.py --config configs/topology_ablation.json
python main_experiment.py --config configs/structured_comparison.json
python main_experiment.py --config configs/loss_ablation.json
```

`split_seed` fixes the pilot 80/20 image split. `training_seeds` expands every
listed experiment across multiple initialization/training seeds. Every run
saves its config, exact split manifest, history, provenance, best checkpoint,
model summaries, per-image metrics, curves, and fixed reconstruction grids.

Aggregate completed runs with:

```bash
python aggregate_results.py outputs/topology_ablation
python analysis/make_paper_tables.py outputs/topology_ablation/mean_std.csv
```

Formal concept experiments should use the official CUB train/test split, with a
validation subset drawn only from the official training set. Existing random
80/20 results are pilot results and must not be combined with official-split
statistics.
