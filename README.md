# Bird Autoencoder

Controlled experiments on **CUB-200-2011** for studying latent topology,
decoder accessibility, concept supervision, residual side channels, and
concept-conditioned image reconstruction.

The repository contains two experiment tracks:

1. **Stage 1 — latent topology and reconstruction**
   - entry point: `main_experiment.py`;
   - compares spatial, ordered-vector, fixed-permutation, global-mixing,
     global-compression, and spatial-channel-compression bottlenecks;
   - current controlled conclusion:

     ```text
     unstructured global vector < structured vector ≈ spatial map
     ```

2. **Stage 2 — factorized concept/residual reconstruction**
   - safety-checked entry point: `run_factorized.py`;
   - internal runner: `main_factorized.py`;
   - uses official CUB metadata, certainty-weighted concepts, fixed-width
     structured residuals, matched unsupervised controls, interventions, local
     ROI summaries, and residual-to-concept probes.

The project currently uses **64×64 full images**, no bounding-box crop, and no
encoder-decoder skip connections.

---

## Official test is locked

All development and pilot configs must use:

```json
"evaluate_official_test": false,
"official_test_release": false
```

Run Stage 2 with:

```bash
python run_factorized.py --config CONFIG.json
```

Official-test evaluation is permitted only when:

1. `evaluate_official_test=true`;
2. `official_test_release=true`;
3. every factorized experiment name contains `confirmatory`, or a standalone
   config has a top-level `run_name` containing `confirmatory`.

The same guard executes inside `main_factorized.main()`, so invoking the
internal runner directly does not bypass the lock.

Official-test results must not be used for concept selection, capacity
selection, checkpoint-rule selection, or intervention-definition selection.

---

## Implemented

### Stage 1

- config-driven experiment runner;
- deterministic pilot split with separate split and training seeds;
- A/B ordered-vector equivalence and topology controls;
- per-run config, provenance, split manifest, checkpoint, summaries,
  per-image metrics, curves, reconstructions, and difference maps;
- seed aggregation and paper-table utilities.

### Stage 2

- official CUB train/test parsing and train-internal validation splitting;
- certainty-weighted multi-label attribute supervision;
- whole-group attribute preparation and predictability refinement;
- standalone decoder-free concept observability predictor;
- `concept`, `control`, and `concept_only` modes;
- fixed residual geometry:

  ```text
  shared features
  -> Conv2D(15)
  -> non-trainable prefix ChannelMask
  -> always-shaped 8×8×15 decoder input
  ```

  Active channels are 15/8/4/2, while encoder and decoder trainable parameter
  counts remain invariant.

- matched binary unsupervised condition:

  ```text
  c: Dense(Dc) -> sigmoid -> SemanticBottleneck + concept loss
  u: Dense(Dc) -> sigmoid -> SemanticBottleneck + no concept loss
  ```

- hard/soft/visible-ground-truth concept diagnostics;
- group interventions, bird bounding boxes, and landmark-centred ROIs;
- linear/MLP `m -> concept` probes with backward-compatible outputs;
- factorized result aggregation.

### Still pending for the full Workshop protocol

- residual-only `z=m` mode in the unified runner;
- frozen-shared-trunk concept readout probe;
- independent semantic intervention evaluator;
- valid donor-swap semantic-success metrics;
- `u` intervention/usage diagnostics;
- frozen probe hyperparameters for confirmatory real/null inference;
- final confirmatory capacity config and one-shot official-test release.

---

## Important structural limitation

The current concept path is global:

```text
shared 8×8 features
-> GlobalAveragePooling2D
-> Dense(Dc)
-> sigmoid / binary bottleneck
-> Dense back to an 8×8 condition map
```

The residual path preserves the full `8×8` grid through a `1×1` convolution.
Therefore, the concept code cannot carry instance-specific spatial location
before the bottleneck, while the residual can.

Low concept use may therefore reflect at least four mechanisms:

1. limited concept observability at 64×64;
2. joint-training or discretization failure;
3. accessible residual bypass;
4. structural spatial disadvantage of the global concept path.

A null intervention result does not distinguish these mechanisms by itself.

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
│   ├── factorized_concepts.json
│   └── archive/factorized_corruption_legacy.json
├── analysis/
│   ├── validate_stage1_config.py
│   ├── refine_attribute_selection.py
│   ├── concept_probe.py
│   ├── aggregate_factorized.py
│   └── make_paper_tables.py
└── tests/
```

---

## Setup and checks

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m compileall -q .
python -m unittest discover -s tests -v
```

The pinned environment currently uses TensorFlow 2.21 and Keras 3.15. See
`requirements.txt` for exact versions.

Do not start GPU experiments unless the official-test guard and residual-mask
invariance tests pass on the training server.

---

## Dataset layout

### Stage 1

Stage 1 configs point `dataset_path` to:

```text
CUB_200_2011/images/
```

Stage 1 uses a deterministic 80/20 pilot split. Do not combine these statistics
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

---

## Stage 1 commands

```bash
python analysis/validate_stage1_config.py configs/topology_ablation.json
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

### 1. Prepare attributes

```bash
python prepare_attributes.py \
  --cub-root /data/CUB_200_2011 \
  --output outputs/attribute_preparation
```

Selection uses only the train subset inside official train.

### 2. Smoke test

```bash
python run_factorized.py --config configs/factorized_smoke.json
```

The smoke config is seed 41, validation-only, and cannot load official test.

### 3. Standalone concept observability pilot

```bash
python standalone_concept_predictor.py \
  --config configs/standalone_concept_pilot.json
```

The actual predictor graph is:

```text
shared convolutional trunk -> GAP -> Dense -> sigmoid
```

It contains no residual, `SemanticBottleneck`, decoder, or reconstruction loss.
The train-internal validation pool is split into two disjoint, class-stratified
subsets:

- `selection_validation`: checkpointing and concept/group selection;
- `reporting_validation`: untouched held-out observability reporting.

Outputs include:

```text
concept_metrics.csv                       # selection only
concept_group_metrics.csv                 # selection only
reporting_concept_metrics.csv             # held-out reporting
reporting_concept_group_metrics.csv       # held-out reporting
split_manifest.csv
result.json
```

### 4. Freeze predictable groups

Use only the selection tables:

```bash
python analysis/refine_attribute_selection.py \
  --initial-selection outputs/attribute_preparation/selected_attributes.json \
  --concept-metrics outputs/standalone_concept_pilot/concept_metrics.csv \
  --attribute-definitions outputs/standalone_concept_pilot/selected_attribute_definitions.csv \
  --min-group-ap-lift 0.05 \
  --output outputs/attribute_preparation/selected_attributes_predictable_groups.json
```

Do not present the selection metrics as an unbiased upper bound. Use the
`reporting_*` files for held-out observability reporting.

### 5. Seed-41 capacity pilot

```bash
python run_factorized.py --config configs/factorized_capacity_pilot.json
```

The pilot compares active residual capacities 960/512/256/128 under fixed model
parameter counts and does not evaluate official test.

`configs/factorized_capacity_sweep.json` remains a validation-only compatibility
alias; new experiments should use `factorized_capacity_pilot.json`.

### 6. Leakage probe

```bash
python analysis/concept_probe.py \
  --train-latents outputs/factorized_capacity_pilot/RUN/train_probe_latents.npz \
  --validation-latents outputs/factorized_capacity_pilot/RUN/validation_latents.npz \
  --attribute-definitions outputs/factorized_capacity_pilot/selected_attribute_definitions.csv \
  --output outputs/factorized_capacity_pilot/RUN/concept_probe.csv
```

For confirmatory real/null inference, choose and freeze all probe
hyperparameters on seed-41 pilot latents. Use the same fixed values for the real
probe and every null; do not run a new grid search inside each null replicate.

### 7. Confirmatory release

The repository intentionally does not ship a ready-to-run official-test config.
After the concept subset, retained capacities, checkpoint rules, and analysis
definitions are frozen:

1. create a dedicated confirmatory config;
2. use pre-registered seeds 42/43/44, or a pre-recorded infrastructure-failure
   replacement;
3. include `confirmatory` in every run name;
4. set both release flags to `true`;
5. run once through the safety-checked entry point.

---

## Archived experiments

Mild and medium residual corruption were removed from the primary path until
leakage and conditioning diagnostics are complete. Their historical config is
preserved at:

```text
configs/archive/factorized_corruption_legacy.json
```

---

## Interpretation boundaries

- Good reconstruction does not prove concept faithfulness.
- High concept AP/BA does not prove decoder concept use.
- A positive `m -> concept` probe shows recoverable residual information, not
  necessarily information used by the decoder.
- A binary unsupervised `u` is a supervision control, not proof that semantics
  are the only learned difference.
- Pixel or ROI change is not automatically semantic intervention success.
- Bird bounding boxes are not segmentation masks.
- Part-landmark ROIs are approximations.
- Global SSIM can hide localized effects.
- The GAP concept path is structurally disadvantaged for spatial reconstruction
  relative to the residual path.
- Current conclusions are specific to 64×64 CUB and the present convolutional
  encoder-decoder unless separately confirmed.
