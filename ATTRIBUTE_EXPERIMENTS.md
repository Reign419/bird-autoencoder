# CUB factorized concept experiments

This file describes the code-supported Stage 2 workflow. The current scientific
target is an Interpretability Workshop study of reconstruction quality, concept
observability, decoder concept use, and residual side channels.

Stage 1 topology experiments remain available through `main_experiment.py` and
are not modified by this workflow.

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

Official-test evaluation is rejected unless:

1. `evaluate_official_test` is explicitly `true`;
2. `official_test_release` is explicitly `true`;
3. every experiment name contains `confirmatory`.

The same validation runs inside `main_factorized.main()`, so direct invocation
cannot silently bypass the release guard.

### Pilot and confirmatory seeds

- pilot training seed: `41`;
- confirmatory training seeds: `42`, `43`, `44`;
- split seed is separate from training seed;
- a replacement seed may be used only for a recorded infrastructure failure,
  never to replace model collapse or an unfavourable result.

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

The first metadata pass creates `cache/attribute_labels.npz`; later runs reuse
it.

---

## 3. Environment and checks

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m compileall -q .
python -m unittest discover -s tests -v
```

The factorized tests include:

- official-test release guard checks;
- fixed residual-mask parameter invariance;
- concept/control parameter-budget matching.

Do not start GPU experiments if these tests fail.

---

## 4. Prepare attributes

```bash
python prepare_attributes.py \
  --cub-root /data/CUB_200_2011 \
  --output outputs/attribute_preparation
```

Expected outputs:

```text
outputs/attribute_preparation/
├── attribute_statistics.csv
├── group_statistics.csv
├── split_manifest.csv
├── selected_attributes.json
└── attribute_selection_report.md
```

The initial selection uses only the train subset inside official train.

---

## 5. Smoke test

Edit paths in `configs/factorized_smoke.json`, then run:

```bash
python run_factorized.py --config configs/factorized_smoke.json
```

This uses at most 64 images per split and never loads official-test images.

---

## 6. Standalone concept observability upper bound

Run:

```bash
python standalone_concept_predictor.py \
  --config configs/standalone_concept_pilot.json
```

The architecture is:

```text
shared convolutional trunk
-> GlobalAveragePooling2D
-> Dense(Dc)
-> sigmoid
```

It has no residual path, `SemanticBottleneck`, decoder, or reconstruction loss.
This is the primary evidence for whether the selected CUB concepts are
observable at 64×64 under the current trunk.

Expected outputs:

```text
outputs/standalone_concept_pilot/
├── result.json
├── history.csv
├── model_summary.txt
├── split_manifest.csv
├── selected_attribute_definitions.csv
├── concept_metrics.csv
└── concept_group_metrics.csv
```

Freeze attribute groups using selection-validation metrics only. Reported
standalone upper-bound metrics must later come from a held-out split that did
not participate in selection.

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

Do not use official-test, reconstruction, or intervention results to refine the
subset.

---

## 8. Fixed-geometry residual capacity pilot

Run:

```bash
python run_factorized.py \
  --config configs/factorized_capacity_pilot.json
```

The config uses seed `41` and validation only.

Residual capacity is implemented as:

```text
Conv2D(15) residual head
-> fixed non-trainable prefix ChannelMask
-> always-shaped 8×8×15 decoder input
```

Active channels are 15, 8, 4, and 2, corresponding to 960, 512, 256, and 128
active scalars. The residual head and decoder parameter counts must be identical
at every capacity.

### Matched unsupervised control

The current matched control follows the same condition topology as concepts:

```text
c: Dense(Dc) -> sigmoid -> SemanticBottleneck + concept loss
u: Dense(Dc) -> sigmoid -> SemanticBottleneck + no concept loss
```

The old `LayerNorm -> tanh -> Gaussian` control is no longer used for primary
comparisons. Its information capacity and decoder accessibility were not
matched to a binary semantic code.

---

## 9. Structural confound: global concept path

The current concept path applies global average pooling before concept
prediction. The residual path preserves the `8×8` feature grid. Therefore the
concept path cannot carry instance-specific spatial location, while the
residual can.

This is intentional for image-level CUB attributes, but it creates an important
alternative explanation for low concept effects:

1. accessible residual bypass;
2. weak conditioning or joint-training competition;
3. limited concept observability;
4. structural spatial disadvantage of the concept path.

The current four-way interpretation must be preserved in analysis and
limitations. A null concept-intervention result does not isolate one mechanism
by itself.

---

## 10. Concept and residual diagnostics

Each concept run may write:

```text
concept_metrics.csv
concept_group_metrics.csv
semantic_bottleneck_analysis.csv
group_interventions.csv
validation_latents.npz
train_probe_latents.npz
figures/group_interventions/
```

Interpret them jointly:

- soft much better than hard: discretization may be costly;
- visible ground truth much better than hard: concept prediction is limiting;
- concept intervention has little effect: decoder neglect, spatial disadvantage,
  or residual dominance remains possible;
- large `m -> concept` probe lift: concept information is recoverable from the
  residual;
- recoverable information is not automatically information used by the decoder;
- bbox and landmark ROI metrics are approximations, not segmentation proof.

---

## 11. Residual-to-concept probes

Fast linear diagnostic:

```bash
python analysis/concept_probe.py \
  --train-latents outputs/factorized_capacity_pilot/RUN/train_probe_latents.npz \
  --validation-latents outputs/factorized_capacity_pilot/RUN/validation_latents.npz \
  --attribute-definitions outputs/factorized_capacity_pilot/selected_attribute_definitions.csv \
  --output outputs/factorized_capacity_pilot/RUN/concept_probe.csv
```

For confirmatory real/null inference:

1. choose and freeze linear regularization and all probe hyperparameters on
   seed-41 pilot latents;
2. use the same fixed hyperparameters for the real probe and every null;
3. preserve complete label/certainty rows when permuting supervision;
4. do not perform a new grid search inside each null replicate;
5. treat MLP nulls as exploratory unless adequately powered.

---

## 12. Corruption experiments are archived

Mild and medium residual corruption are not part of the primary path before the
leakage/conditioning mechanism gate. Their previous config is preserved at:

```text
configs/archive/factorized_corruption_legacy.json
```

Do not run it as part of the primary capacity study.

---

## 13. Confirmatory release

The repository intentionally does not provide an already-released official-test
config. After the subset, retained capacity points, checkpoint rules, and
analysis definitions are frozen:

1. create a dedicated confirmatory config;
2. use pre-registered seeds;
3. include `confirmatory` in every experiment name;
4. set both official-test flags to `true`;
5. run once through `run_factorized.py`;
6. record the release date and commit in the experiment log.

Any earlier access to official test must be disclosed rather than silently
relabelled as untouched evaluation.
