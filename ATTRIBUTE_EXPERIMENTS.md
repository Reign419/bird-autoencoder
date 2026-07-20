# CUB factorized concept experiments

This file describes the code-supported Stage 2 workflow for an Interpretability
Workshop study of concept observability, reconstruction, decoder concept use,
and residual side channels.

Stage 1 topology experiments remain available through `main_experiment.py`.

---

## 1. Non-negotiable safeguards

### Official-test lock

All development and pilot configs must contain:

```json
"evaluate_official_test": false,
"official_test_release": false
```

Use:

```bash
python run_factorized.py --config CONFIG.json
```

Official-test evaluation is rejected unless both release flags are true and all
configured run names contain `confirmatory`. The guard also runs inside
`main_factorized.main()`, so direct invocation cannot silently bypass it.

### Seeds

- pilot training seed: `41`;
- confirmatory training seeds: `42`, `43`, `44`;
- split seed is separate from training seed;
- replacement seeds are allowed only for logged infrastructure failures, not
  for model collapse or unfavourable outcomes.

---

## 2. Dataset layout

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

## 3. Environment and checks

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m compileall -q .
python -m unittest discover -s tests -v
```

The factorized tests cover:

- official-test release rules;
- fixed residual-mask parameter invariance;
- concept/control parameter-budget matching;
- Keras save/load of the residual mask.

Do not start GPU runs while these checks fail.

---

## 4. Prepare attributes

```bash
python prepare_attributes.py \
  --cub-root /data/CUB_200_2011 \
  --output outputs/attribute_preparation
```

The initial group selection uses only the train subset inside official train.

Expected outputs:

```text
attribute_statistics.csv
group_statistics.csv
split_manifest.csv
selected_attributes.json
attribute_selection_report.md
```

---

## 5. Smoke test

```bash
python run_factorized.py --config configs/factorized_smoke.json
```

The smoke config is seed 41, validation-only, and cannot load official-test
images.

---

## 6. Standalone concept observability upper bound

```bash
python standalone_concept_predictor.py \
  --config configs/standalone_concept_pilot.json
```

The predictor graph is exactly:

```text
shared convolutional trunk
-> GlobalAveragePooling2D
-> Dense(Dc)
-> sigmoid
```

It has no residual branch, `SemanticBottleneck`, decoder, or reconstruction
loss.

The train-internal validation pool is split into two disjoint, class-stratified
subsets:

- `selection_validation`: early stopping and concept/group selection;
- `reporting_validation`: untouched held-out observability reporting.

Outputs:

```text
outputs/standalone_concept_pilot/
├── result.json
├── history.csv
├── model_summary.txt
├── split_manifest.csv
├── selected_attribute_definitions.csv
├── concept_metrics.csv
├── concept_group_metrics.csv
├── reporting_concept_metrics.csv
└── reporting_concept_group_metrics.csv
```

The backward-compatible `concept_metrics.csv` files are selection-only. Do not
quote them as an unbiased upper bound. Use `reporting_*` for held-out reporting.

---

## 7. Freeze predictable groups

```bash
python analysis/refine_attribute_selection.py \
  --initial-selection outputs/attribute_preparation/selected_attributes.json \
  --concept-metrics outputs/standalone_concept_pilot/concept_metrics.csv \
  --attribute-definitions outputs/standalone_concept_pilot/selected_attribute_definitions.csv \
  --min-group-ap-lift 0.05 \
  --output outputs/attribute_preparation/selected_attributes_predictable_groups.json
```

Do not use reporting-validation, official-test, reconstruction, or intervention
results to refine the subset.

---

## 8. Fixed-geometry residual capacity pilot

```bash
python run_factorized.py \
  --config configs/factorized_capacity_pilot.json
```

The config uses seed 41 and validation only.

Residual capacity is implemented as:

```text
Conv2D(15) residual head
-> fixed non-trainable prefix ChannelMask
-> always-shaped 8×8×15 decoder input
```

Active channels 15/8/4/2 correspond to 960/512/256/128 active scalars. Encoder
and decoder trainable parameter counts must remain identical.

### Matched unsupervised control

```text
c: Dense(Dc) -> sigmoid -> SemanticBottleneck + concept loss
u: Dense(Dc) -> sigmoid -> SemanticBottleneck + no concept loss
```

The old `LayerNorm -> tanh -> Gaussian` control is not used for primary
comparisons because its information capacity and decoder accessibility were not
matched to the binary semantic code.

---

## 9. Structural confound: global concept path

The concept path applies global average pooling before concept prediction. The
residual path preserves the `8×8` feature grid. Therefore, the concept code
cannot carry instance-specific spatial location before the bottleneck, while
the residual can.

Low concept effects may reflect:

1. limited concept observability;
2. joint-training/discretization failure;
3. accessible residual bypass;
4. structural spatial disadvantage of the global concept path.

A null concept-intervention result does not isolate one mechanism.

---

## 10. Diagnostics

Concept runs may write:

```text
concept_metrics.csv
concept_group_metrics.csv
semantic_bottleneck_analysis.csv
group_interventions.csv
validation_latents.npz
train_probe_latents.npz
figures/group_interventions/
```

Interpret jointly:

- soft better than hard: discretization may be costly;
- visible ground truth better than hard: concept prediction is limiting;
- low intervention effect: decoder neglect, spatial disadvantage, or residual
  dominance remains possible;
- positive `m -> concept` lift: concept information is recoverable from the
  residual, not necessarily used by the decoder;
- bbox and landmark ROI metrics are approximations, not segmentation proof.

---

## 11. Residual-to-concept probes

```bash
python analysis/concept_probe.py \
  --train-latents outputs/factorized_capacity_pilot/RUN/train_probe_latents.npz \
  --validation-latents outputs/factorized_capacity_pilot/RUN/validation_latents.npz \
  --attribute-definitions outputs/factorized_capacity_pilot/selected_attribute_definitions.csv \
  --output outputs/factorized_capacity_pilot/RUN/concept_probe.csv
```

For confirmatory real/null inference:

1. choose and freeze all probe hyperparameters on seed-41 pilot latents;
2. use the same fixed values for real and every null;
3. preserve complete label/certainty supervision records when permuting;
4. do not run a new grid search inside each null replicate;
5. treat MLP nulls as exploratory unless adequately powered.

---

## 12. Archived corruption experiments

Residual corruption is not part of the primary path before the mechanism gate.
The previous configuration is preserved at:

```text
configs/archive/factorized_corruption_legacy.json
```

---

## 13. Confirmatory release

The repository does not ship a pre-released official-test config. After the
subset, retained capacities, checkpoint rules, and analysis definitions are
frozen:

1. create a dedicated confirmatory config;
2. use pre-registered seeds;
3. include `confirmatory` in every configured run name;
4. set both release flags to true;
5. run once through the guarded entry point;
6. record release date and commit in the experiment log.

Any earlier official-test access must be disclosed rather than relabelled as
untouched evaluation.
