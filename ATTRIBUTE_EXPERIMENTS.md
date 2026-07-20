# CUB factorized concept experiments

This protocol keeps the Phase 1 topology experiments unchanged.  The concept
study uses `main_factorized.py`, the official CUB train/test split, whole
attribute groups, certainty-weighted supervision, and matched controls.

## 1. Dataset layout

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

The large image-level attribute file stays on the training server.  The first
preparation run creates `cache/attribute_labels.npz`; later runs use the cache.

## 2. Environment and static checks

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m compileall -q .
python -m unittest discover -s tests -v
```

## 3. Validate metadata and make the initial group selection

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

Inspect the report before training.  Selection uses only the training subset
inside the official training split.  The suggested control-noise value is an
AWGN information-rate proxy, not an exact mutual-information match.

## 4. Two-epoch smoke test

Edit these two paths in `configs/factorized_smoke.json`:

```json
"cub_root": "/data/CUB_200_2011",
"selected_attributes_path": "outputs/attribute_preparation/selected_attributes.json"
```

Then run:

```bash
python main_factorized.py --config configs/factorized_smoke.json
```

This uses at most 64 images per split and runs one concept model plus one
continuous control.  Confirm that both runs create `result.json`,
`history.csv`, `checkpoints/best.keras`, and `validation_latents.npz`.

## 5. Concept-predictor pilot and predictability filter

Edit the dataset path in `configs/concept_pilot.json`, then run:

```bash
python main_factorized.py --config configs/concept_pilot.json
```

Find the pilot run directory containing `concept_metrics.csv`, then freeze the
predictability-filtered selection:

```bash
python analysis/refine_attribute_selection.py \
  --initial-selection outputs/attribute_preparation/selected_attributes.json \
  --concept-metrics outputs/concept_pilot/REPLACE_WITH_RUN/concept_metrics.csv \
  --attribute-definitions outputs/concept_pilot/selected_attribute_definitions.csv \
  --min-group-ap-lift 0.05 \
  --output outputs/attribute_preparation/selected_attributes_final.json
```

Do not use official-test metrics for this refinement.

## 6. Full matched experiment

Edit `cub_root` in `configs/factorized_concepts.json`.  Its selection path
already points to `selected_attributes_final.json`.

```bash
python main_factorized.py --config configs/factorized_concepts.json
```

The configuration runs seeds 42, 43, and 44 for:

- factorized model with clean residual;
- mild residual corruption;
- medium residual corruption;
- rate-proxy-matched continuous `u` control;
- concept-only reconstruction.

Run the existing structured `8x8x16` and residual-only `8x8x15` models as
reference baselines.  Do not compare different selected concept dimensions
without building a new matched `u(Dc)` control for each dimension.

## 7. Residual-to-concept leakage probe

For each factorized concept run:

```bash
python analysis/concept_probe.py \
  --train-latents outputs/factorized_concepts/RUN/train_probe_latents.npz \
  --validation-latents outputs/factorized_concepts/RUN/validation_latents.npz \
  --attribute-definitions outputs/factorized_concepts/selected_attribute_definitions.csv \
  --output outputs/factorized_concepts/RUN/concept_probe.csv
```

`probe_ap_lift` measures how much concept information remains recoverable from
`m` above the prevalence baseline.

The command above remains the backward-compatible fast linear probe.  For the
pre-registered two-level real-vs-null diagnostic, use the same output reference
and add:

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

The example uses 20 permutations as a diagnostic pilot. Empirical p-values and
FDR q-values are coarse at that resolution; increase `--null-repeats` for
confirmatory inference. Each permutation moves an atomic label together with
its certainty weight, preserving the observed label/certainty distribution
while breaking the relationship between `m` and that concept.

The legacy `concept_probe.csv` and `concept_probe_groups.csv` filenames and
columns are unchanged.  Detailed outputs are additive:

```text
concept_probe_linear.csv
concept_probe_linear_null.csv
concept_probe_mlp.csv
concept_probe_mlp_null.csv
concept_probe_comparison.csv
concept_probe_summary.json
```

Interpret linear lift as low-complexity accessibility and MLP lift as nonlinear
recoverability.  An MLP-only signal is evidence that information exists in `m`,
but is not by itself proof that the current decoder can easily use it.

After all probes finish, aggregate every axis across seeds:

```bash
python analysis/aggregate_factorized.py outputs/factorized_concepts
```

## 8. Main diagnostic files

Each concept run contains:

```text
concept_metrics.csv
concept_group_metrics.csv
semantic_bottleneck_analysis.csv
group_interventions.csv
figures/group_interventions/*.png
validation_latents.npz
train_probe_latents.npz
official_test_probe_latents.npz
official_test_result.json
```

Interpret them together:

- soft much better than hard: continuous confidence values carry extra detail;
- ground truth much better than hard: concept prediction is the bottleneck;
- group shuffle has no effect: decoder is ignoring concepts;
- large `m -> concept` probe lift: concept leakage remains in the residual;
- bird bounding-box effects are not segmentation-mask measurements;
- local ROI effects are landmark-centred approximations; interpret target/non-target
  enrichment and Top-1% overlap together rather than treating either as strict
  localization proof;
- global SSIM can underestimate localized changes, so always inspect pixel
  change, effective-change subsets, and difference maps.

## 9. STE fallback

Switch `semantic_method` from `ste` to `gumbel` only if multiple seeds show
loss oscillation, repeated gradient-clipping/NaN events, probabilities stuck
near 0.5, or collapsed concept distributions.  Add:

```json
"semantic_method": "gumbel",
"temperature_start": 1.0,
"temperature_end": 0.2,
"temperature_anneal_epochs": 50
```

The implementation uses Binary Concrete/Gumbel-sigmoid for atomic multi-label
attributes and hard thresholds at test time.  Treat this as a pre-registered
fallback, not another result chosen after looking at official-test scores.

## 10. Predictable complete groups and capacity sweep

Freeze a complete-group primary subset without changing the historical JSON
reference pattern:

```bash
python analysis/refine_attribute_selection.py \
  --initial-selection outputs/attribute_preparation/selected_attributes_final.json \
  --concept-metrics outputs/concept_pilot/RUN/concept_metrics.csv \
  --attribute-definitions outputs/attribute_preparation/attribute_definitions.csv \
  --output outputs/attribute_preparation/selected_attributes_predictable_groups.json \
  --min-group-ap-lift 0.05 \
  --min-attribute-ap-lift 0.05 \
  --min-balanced-accuracy 0.60 \
  --min-positive-count 25 \
  --min-negative-count 25 \
  --min-predictable-fraction 0.60
```

The main JSON contains complete admitted groups.  The additive `.atomic.json`
file is secondary attribute-level analysis and should not replace complete
groups in the primary group-intervention experiment.

Run the seed-42 `m=960/512/256/128/0` pilot and its matched continuous controls:

```bash
python main_factorized.py --config configs/factorized_capacity_sweep.json
```

## 11. Reporting

Report mean and standard deviation across seeds along four axes:

1. reconstruction quality;
2. concept prediction;
3. group intervention/use;
4. bidirectional leakage.

A successful factorization needs competitive reconstruction, predictable hard
concepts, non-zero and spatially plausible group interventions, and limited
leakage in both directions.
