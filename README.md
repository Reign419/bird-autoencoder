# Bird Autoencoder

Controlled experiments on **CUB-200-2011** for studying latent topology,
decoder accessibility, concept supervision, residual side channels, and
concept-conditioned image reconstruction.

The repository contains two experiment tracks:

1. **Stage 1 — latent topology and reconstruction**
   - official entry: `main_experiment.py`;
   - compares spatial, ordered-vector, fixed-permutation, global-mixing,
     global-compression, and spatial-channel-compression bottlenecks;
   - current controlled conclusion:

     ```text
     unstructured global vector < structured vector ≈ spatial map
     ```

2. **Stage 2 — factorized concept/residual reconstruction**
   - safety-checked entry: `run_factorized.py`;
   - internal training implementation: `main_factorized.py`;
   - uses official CUB metadata, certainty-weighted concepts, a fixed-width
     structured residual, matched unsupervised controls, concept interventions,
     optional part ROIs, and residual-to-concept probes.

The project currently uses **64×64 full images**, no bounding-box crop, and no
encoder-decoder skip connections.

---

## Critical safety rule: official test is locked

All pilot and development configs must contain:

```json
"evaluate_official_test": false,
"official_test_release": false
```

Run Stage 2 through:

```bash
python run_factorized.py --config CONFIG.json
```

Official-test evaluation is allowed only when both flags are explicitly true
**and every experiment name contains `confirmatory`**. The same guard also runs
inside `main_factorized.main()`, so calling the internal runner directly does
not bypass the lock.

Never use official-test results for concept selection, capacity selection,
checkpoint-rule selection, or intervention-definition selection.

---

## Current implementation status

### Stage 1 implemented

- config-driven runner: `main_experiment.py`;
- deterministic pilot split with separate split/training seeds;
- per-run config, provenance, split manifest, model summaries, checkpoint,
  per-image metrics, curves, reconstruction grids, and difference maps;
- A/B ordered-vector equivalence, position permutation, orthogonal/global
  mixing, and compression controls;
- seed aggregation and paper-table utilities.

### Stage 2 implemented

- CUB official train/test parsing and train-internal validation split;
- certainty-weighted multi-label attribute supervision;
- whole-group attribute preparation and predictability refinement;
- standalone decoder-free concept observability predictor:
  `standalone_concept_predictor.py`;
- three current factorized modes:
  - `concept`: supervised concepts plus residual;
  - `control`: unsupervised matched binary condition `u` plus residual;
  - `concept_only`: supervised concepts without residual;
- fixed residual-head geometry:
  - encoder always produces `8×8×15` residual channels;
  - a non-trainable prefix `ChannelMask` activates 15/8/4/2 channels;
  - decoder always receives `8×8×15`;
  - encoder and decoder trainable parameter counts are invariant across the
    capacity sweep;
- matched `u` path:

  ```text
  c: Dense(Dc) -> sigmoid -> SemanticBottleneck + concept loss
  u: Dense(Dc) -> sigmoid -> SemanticBottleneck + no concept loss
  ```

- hard/soft/visible-ground-truth concept diagnostics;
- group interventions, bird bounding-box summaries, and landmark-centred ROIs;
- linear/MLP residual-to-concept probes with backward-compatible outputs;
- Stage 2 result aggregation.

### Still pending for the full Workshop protocol

- residual-only `z=m` mode in the unified Stage 2 runner;
- frozen-shared-trunk concept readout probe;
- independent semantic intervention evaluator;
- valid donor-swap semantic-success metrics;
- frozen probe hyperparameters for confirmatory real/null inference;
- final confirmatory capacity config and one-shot official-test release.

---

## Important architectural limitation

The current concept path is image-level:

```text
shared spatial features
-> GlobalAveragePooling2D
-> Dense(Dc)
-> sigmoid / binary bottleneck
-> Dense back to an 8×8 condition map
```

The residual path preserves the full `8×8` feature grid through a `1×1`
convolution. Therefore, the concept path cannot encode instance-specific spatial
location before the bottleneck, while the residual can. This is intentional for
image-level CUB attributes, but it is also a structural confound:

- low concept use may reflect residual bypass;
- low concept use may reflect weak conditioning or training competition;
- low concept use may also reflect the concept path's loss of spatial detail.

The current work must not treat these explanations as interchangeable.

---

## Repository structure

```text
bird-autoencoder/
├── main_experiment.py
├── run_factorized.py
├── main_factorized.py
├── standalone_concept_predictor.py
├── prepare_attributes.py
├── ATTRIBUTE_EXPERIMENTS.md
├── attribute_data.py
├── factorized_analysis.py
├── losses.py
├── train_utils.py
├── model/
│   ├── model_registry.py
│   ├── model_topology_common.py
│   ├── model_bottleneck_ablation.py
│   └── model_factorized_lite.py
├── configs/
│   ├── topology_ablation.json
│   ├── ordered_vector_equivalence.json
│   ├── structured_comparison.json
│   ├── loss_ablation.json
│   ├── factorized_smoke.json
│   ├── standalone_concept_pilot.json
│   ├── factorized_capacity_pilot.json
│   └── factorized_concepts.json
├── analysis/
│   ├── validate_stage1_config.py
│   ├── refine_attribute_selection.py
│   ├── concept_probe.py
│   ├── aggregate_factorized.py
│   └── make_paper_tables.py
└── tests/
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m compileall -q .
python -m unittest discover -s tests -v
```

The pinned environment currently uses TensorFlow 2.21 / Keras 3.15. See
`requirements.txt` for exact versions.

---

## Dataset layout

### Stage 1

Stage 1 configs point `dataset_path` to:

```text
CUB_200_2011/images/
```

Stage 1 uses a deterministic 80/20 pilot split. Do not combine those statistics
with official-split Stage 2 results.

### Stage 2

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

The first attribute preparation run creates a compact cache under the dataset
root unless another cache path is configured.

---

## Stage 1 commands

Validate a config without importing TensorFlow:

```bash
python analysis/validate_stage1_config.py configs/topology_ablation.json
```

Run experiments:

```bash
python main_experiment.py --config configs/topology_ablation.json
python main_experiment.py --config configs/ordered_vector_equivalence.json
python main_experiment.py --config configs/structured_comparison.json
python main_experiment.py --config configs/loss_ablation.json
```

Aggregate:

```bash
python aggregate_results.py outputs/topology_ablation
python analysis/make_paper_tables.py outputs/topology_ablation/mean_std.csv
```

---

## Stage 2 execution order

### 1. Prepare and audit attributes

```bash
python prepare_attributes.py \
  --cub-root /data/CUB_200_2011 \
  --output outputs/attribute_preparation
```

This writes attribute/group statistics, a split manifest, the initial selection,
and an audit report. Selection uses only the train subset inside official train.

### 2. Smoke test

Edit paths in `configs/factorized_smoke.json`, then run:

```bash
python run_factorized.py --config configs/factorized_smoke.json
```

The smoke config is validation-only and may not load official-test images.

### 3. Standalone concept observability pilot

```bash
python standalone_concept_predictor.py \
  --config configs/standalone_concept_pilot.json
```

This model is exactly:

```text
shared convolutional trunk -> GAP -> Dense -> sigmoid
```

It has no residual, no `SemanticBottleneck`, no reconstruction decoder, and no
reconstruction loss. It is the primary Week-1 observability upper bound.

### 4. Freeze predictable groups

Use only selection-validation metrics:

```bash
python analysis/refine_attribute_selection.py \
  --initial-selection outputs/attribute_preparation/selected_attributes.json \
  --concept-metrics outputs/standalone_concept_pilot/concept_metrics.csv \
  --attribute-definitions outputs/standalone_concept_pilot/selected_attribute_definitions.csv \
  --min-group-ap-lift 0.05 \
  --output outputs/attribute_preparation/selected_attributes_predictable_groups.json
```

Held-out or official-test reporting must not reuse the same split used for
selection without being labelled selection-biased.

### 5. Seed-41 validation-only capacity pilot

```bash
python run_factorized.py --config configs/factorized_capacity_pilot.json
```

The pilot compares active residual capacities 960/512/256/128 under fixed model
parameter counts. It does not evaluate official test.

`configs/factorized_capacity_sweep.json` is retained as a compatibility alias
for a validation-only pilot; new work should use
`configs/factorized_capacity_pilot.json`.

### 6. Leakage probes

Fast linear diagnostic:

```bash
python analysis/concept_probe.py \
  --train-latents outputs/factorized_capacity_pilot/RUN/train_probe_latents.npz \
  --validation-latents outputs/factorized_capacity_pilot/RUN/validation_latents.npz \
  --attribute-definitions outputs/factorized_capacity_pilot/selected_attribute_definitions.csv \
  --output outputs/factorized_capacity_pilot/RUN/concept_probe.csv
```

For confirmatory real/null inference, freeze probe hyperparameters on pilot
latents first; then use the same fixed hyperparameters for real and every null.
Do not tune separately inside each null replicate.

### 7. Confirmatory release

The repository intentionally does not ship a ready-to-run official-test config.
After the concept subset, retained capacities, checkpoint rules, and analysis
definitions are frozen:

1. create a dedicated confirmatory config;
2. use seeds 42/43/44, or a pre-recorded hardware-failure replacement;
3. make every experiment name contain `confirmatory`;
4. set both release flags to `true`;
5. run once through `run_factorized.py`.

---

## Interpretation boundaries

- Good reconstruction does not prove concept faithfulness.
- High concept AP/BA does not prove the decoder uses concepts.
- A positive `m -> concept` probe shows recoverable residual information, not
  necessarily information used by the decoder.
- A binary unsupervised `u` is a supervision control, not proof that semantics
  are the only difference learned during optimization.
- Pixel or ROI change is not automatically semantic intervention success.
- Bird bounding boxes are not segmentation masks.
- Part-landmark ROIs are approximations.
- Global SSIM can hide localized effects.
- The concept path's GAP bottleneck is structurally disadvantaged for spatial
  reconstruction relative to the residual path.
- Current conclusions are specific to 64×64 CUB and the present convolutional
  encoder-decoder unless separately confirmed.
