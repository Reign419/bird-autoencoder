# Bird Autoencoder

Controlled autoencoder experiments on **CUB-200-2011** for studying how latent
interface design, spatial correspondence, concept supervision, and residual
information affect image reconstruction and interpretability.

The current repository contains two experiment tracks:

1. **Stage 1: latent topology / decoder accessibility**
   - Run through `main_experiment.py`.
   - Studies whether a single-vector interface is harmful by itself.
   - Includes spatial, ordered-vector, fixed-permutation, global-mixing,
     global-compression, and spatial-channel-compression controls.

2. **Stage 2: factorized concept/residual reconstruction**
   - Run through `main_factorized.py`.
   - Uses the official CUB train/test split, CUB attributes, certainty-weighted
     concept supervision, matched continuous controls, semantic bottleneck
     diagnostics, group interventions, optional part-landmark ROIs, and
     residual-to-concept probes.

The current evidence does **not** support the simple claim that vectors are
inherently worse than spatial tensors. A parameter-free
`8x8xC -> Flatten -> [B, K] -> Reshape -> 8x8xC` interface preserves ordered
spatial correspondence. The working Stage 1 conclusion is:

```text
unstructured global vector < structured vector ≈ spatial map
```

The project uses **64x64 full images** as a controlled setting. It does not use
bounding-box crops and does not add encoder-decoder skip connections.

---

## Current implementation status

### Implemented

- Stage 1 config-driven experiment runner: `main_experiment.py`.
- Stage 1 topology primitives and model registry under `model/`.
- Deprecated wrappers: `main.py` and `main_bottleneck_ablation.py` forward to the
  unified Stage 1 runner and should not be used as new entry points.
- CUB attribute preparation and group selection: `prepare_attributes.py`.
- Stage 2 factorized runner: `main_factorized.py`.
- Stage 2 modes currently supported by code:
  - `concept`: reconstruction plus supervised concept head, decoder receives
    `[residual, concepts]`;
  - `control`: matched continuous non-semantic control `u` with residual;
  - `concept_only`: decoder receives only the concept representation.
- Semantic bottleneck diagnostics:
  - hard predicted concepts;
  - soft concept probabilities;
  - ground-truth visible concepts.
- Group-level concept interventions and local change summaries.
- Optional CUB bird bounding boxes and part-landmark ROIs for local intervention
  analysis.
- Residual-to-concept leakage probes with backward-compatible CSV outputs and
  optional real-vs-null linear/MLP diagnostics.
- Factorized result aggregation through `analysis/aggregate_factorized.py`.

### Not yet implemented in code

The latest research protocol discusses a stricter Workshop-oriented capacity
sweep, but the repository code does **not** yet implement all of it. In
particular, the current code does not yet provide:

- residual-only `z=m` mode inside `main_factorized.py`;
- fixed `8x8x15` residual-head masking for residual-capacity sweeps;
- frozen-trunk concept observability probes;
- standalone semantic intervention evaluator;
- donor-swap semantic success metrics;
- frozen probe-hyperparameter selection for confirmatory null inference.

Until those pieces are implemented, the code should be treated as the current
Stage 1 pipeline plus the first Stage 2 factorized diagnostic pipeline, not the
full final Workshop protocol.

---

## Repository structure

```text
bird-autoencoder/
├── main_experiment.py              # Stage 1 official entry point
├── main_factorized.py              # Stage 2 factorized concept/residual entry point
├── prepare_attributes.py           # CUB attribute validation and initial group selection
├── ATTRIBUTE_EXPERIMENTS.md        # Current Stage 2 execution notes
├── data.py                         # Stage 1 image loading and deterministic pilot split
├── attribute_data.py               # CUB official split and attribute cache utilities
├── losses.py                       # Reconstruction and concept losses / metrics
├── train_utils.py                  # Checkpoints, LR scheduling, early stopping
├── visualize.py                    # Reconstruction and difference grids
├── evaluate.py                     # Per-image reconstruction metrics
├── aggregate_results.py            # Stage 1 aggregation
├── factorized_analysis.py          # Stage 2 concept/intervention/local metrics
├── model/
│   ├── model_registry.py
│   ├── model_topology_common.py
│   ├── model_residual_lite.py
│   ├── model_spatial_lite.py
│   ├── model_structured_vector_lite.py
│   ├── model_bottleneck_ablation.py
│   └── model_factorized_lite.py
├── configs/
│   ├── topology_ablation.json
│   ├── ordered_vector_equivalence.json
│   ├── structured_comparison.json
│   ├── loss_ablation.json
│   ├── factorized_smoke.json
│   ├── concept_pilot.json
│   └── factorized_concepts.json
├── analysis/
│   ├── validate_stage1_config.py
│   ├── aggregate_seeds.py
│   ├── make_paper_tables.py
│   ├── refine_attribute_selection.py
│   ├── concept_probe.py
│   └── aggregate_factorized.py
└── tests/
```

---

## Dataset layout

### Stage 1 topology experiments

Stage 1 configs expect `dataset_path` to point to the CUB image directory:

```text
CUB_200_2011/
└── images/
```

Stage 1 uses a deterministic 80/20 pilot split. These results should not be
mixed with official-split Stage 2 results.

### Stage 2 concept experiments

Stage 2 expects the full CUB metadata layout:

```text
CUB_200_2011/
├── images/
├── images.txt
├── train_test_split.txt
├── attributes/
│   ├── attributes.txt
│   └── image_attribute_labels.txt
└── parts/                       # optional local intervention metrics
    ├── parts.txt
    └── part_locs.txt
```

The large image-level attribute file can stay on the training server. The first
preparation run creates an attribute cache; later runs reuse it.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run static checks and tests:

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

The pinned environment currently uses TensorFlow/Keras, NumPy, pandas,
scikit-learn, matplotlib, Pillow, and joblib. See `requirements.txt` for exact
versions.

---

## Stage 1: topology and reconstruction experiments

Validate a Stage 1 config without importing TensorFlow:

```bash
python analysis/validate_stage1_config.py configs/topology_ablation.json
```

Run controlled Stage 1 experiments:

```bash
python main_experiment.py --config configs/topology_ablation.json
python main_experiment.py --config configs/ordered_vector_equivalence.json
python main_experiment.py --config configs/structured_comparison.json
python main_experiment.py --config configs/loss_ablation.json
```

`split_seed` fixes the deterministic pilot image split. `training_seeds` expands
each experiment across multiple initialization/training seeds.

Each Stage 1 run saves:

```text
config.json
result.json
history.csv
provenance.json
split_manifest.csv
model_summary.txt
encoder_summary.txt
decoder_summary.txt
per_image_metrics.csv
curves/
figures/
checkpoints/best.keras
```

Aggregate completed Stage 1 runs with:

```bash
python aggregate_results.py outputs/topology_ablation
python analysis/make_paper_tables.py outputs/topology_ablation/mean_std.csv
```

---

## Stage 2: CUB concept/residual experiments

For the exact staged protocol currently supported by code, see
[`ATTRIBUTE_EXPERIMENTS.md`](ATTRIBUTE_EXPERIMENTS.md).

### 1. Prepare attributes and split manifest

```bash
python prepare_attributes.py \
  --cub-root /data/CUB_200_2011 \
  --output outputs/attribute_preparation
```

Expected outputs include:

```text
outputs/attribute_preparation/
├── attribute_statistics.csv
├── group_statistics.csv
├── split_manifest.csv
├── selected_attributes.json
└── attribute_selection_report.md
```

Selection uses only the training subset inside the official CUB training split.
Do not use official-test metrics for selection.

### 2. Smoke test

Edit paths in `configs/factorized_smoke.json`, then run:

```bash
python main_factorized.py --config configs/factorized_smoke.json
```

This runs a small concept model and a small continuous-control model with at
most 64 images per split.

### 3. Concept-predictor pilot and predictability filter

```bash
python main_factorized.py --config configs/concept_pilot.json
```

Then refine the selected attribute groups from the pilot concept metrics:

```bash
python analysis/refine_attribute_selection.py \
  --initial-selection outputs/attribute_preparation/selected_attributes.json \
  --concept-metrics outputs/concept_pilot/REPLACE_WITH_RUN/concept_metrics.csv \
  --attribute-definitions outputs/concept_pilot/selected_attribute_definitions.csv \
  --min-group-ap-lift 0.05 \
  --output outputs/attribute_preparation/selected_attributes_final.json
```

### 4. Full matched factorized experiment

Edit `cub_root` in `configs/factorized_concepts.json`. The default config runs
seeds 42, 43, and 44 for:

- clean concept/residual factorized model;
- mild residual corruption;
- medium residual corruption;
- rate-proxy-matched continuous `u` control;
- concept-only reconstruction.

Run:

```bash
python main_factorized.py --config configs/factorized_concepts.json
```

### 5. Residual-to-concept leakage probe

Fast backward-compatible linear probe:

```bash
python analysis/concept_probe.py \
  --train-latents outputs/factorized_concepts/RUN/train_probe_latents.npz \
  --validation-latents outputs/factorized_concepts/RUN/validation_latents.npz \
  --attribute-definitions outputs/factorized_concepts/selected_attribute_definitions.csv \
  --output outputs/factorized_concepts/RUN/concept_probe.csv
```

Two-level diagnostic with real-vs-null linear/MLP probes:

```bash
python analysis/concept_probe.py \
  --train-latents outputs/factorized_concepts/RUN/train_probe_latents.npz \
  --validation-latents outputs/factorized_concepts/RUN/validation_latents.npz \
  --test-latents outputs/factorized_concepts/RUN/official_test_probe_latents.npz \
  --evaluation-split test \
  --attribute-definitions outputs/factorized_concepts/selected_attribute_definitions.csv \
  --output outputs/factorized_concepts/RUN/concept_probe.csv \
  --probe-types linear,mlp \
  --null-repeats 20 \
  --jobs 4
```

`20` null repeats is a diagnostic setting. It is too coarse for strong
confirmatory p-values or FDR claims.

Aggregate Stage 2 runs with:

```bash
python analysis/aggregate_factorized.py outputs/factorized_concepts
```

---

## Stage 2 output files

Each concept run may contain:

```text
config.json
result.json
history.csv
model_summary.txt
encoder_summary.txt
decoder_summary.txt
concept_metrics.csv
concept_group_metrics.csv
semantic_bottleneck_analysis.csv
group_interventions.csv
validation_latents.npz
train_probe_latents.npz
official_test_probe_latents.npz
official_test_result.json
figures/group_interventions/
```

Interpret key files together:

- `semantic_bottleneck_analysis.csv`
  - compares hard predicted, soft predicted, and ground-truth visible concepts;
- `group_interventions.csv`
  - measures whether changing concept groups affects reconstructions;
- `concept_metrics.csv` and `concept_group_metrics.csv`
  - report concept prediction quality;
- `concept_probe.csv` and `concept_probe_groups.csv`
  - backward-compatible residual-to-concept leakage summaries;
- `concept_probe_linear.csv`, `concept_probe_mlp.csv`, and null tables
  - detailed probe diagnostics when requested.

Important interpretation boundaries:

- Good reconstruction does not prove concept faithfulness.
- High concept prediction quality does not prove the decoder uses concepts.
- A positive `m -> concept` probe shows recoverable information in the residual,
  but does not by itself prove the decoder uses that information.
- Bird bounding-box metrics are not segmentation-mask metrics.
- Part-landmark ROIs are local approximations, not strict localization proof.
- Global SSIM can hide small localized concept effects.

---

## Research notes and limitations

- Stage 1 is a controlled 64x64 topology study; do not claim universal
  generalization to all resolutions, datasets, or decoders.
- Stage 2 currently implements an initial concept/residual diagnostic pipeline,
  not the final capacity-sweep protocol.
- Official CUB test metrics should be used only after concept selection,
  checkpoint rules, and analysis definitions are frozen.
- Historical random 80/20 Stage 1 results are pilot results and must not be
  combined with official-split Stage 2 statistics.

---

## Deprecated entry points

The following files are kept for backward compatibility only:

```text
main.py
main_bottleneck_ablation.py
```

New experiments should use:

```text
main_experiment.py      # Stage 1
main_factorized.py      # Stage 2
```
