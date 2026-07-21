# Experiment Configurations

Only configs in the **Active** section should be used for new runs. Files under
`archive/` are retained for historical reproducibility and must not be used for
model selection, confirmatory claims, or official-test release.

## Active Stage 1 configs

- `topology_ablation.json` — full Stage 1 topology controls.
- `ordered_vector_equivalence.json` — exact ordered-vector equivalence check.
- `structured_comparison.json` — structured vector versus spatial comparison.
- `loss_ablation.json` — reconstruction-loss ablation.
- `smoke_topology.json` — short Stage 1 smoke test.

## Active Stage 2 configs

- `factorized_smoke.json` — two-epoch, seed-41, validation-only smoke test.
- `standalone_concept_pilot.json` — decoder-free observability pilot; uses seed
  41 and splits the internal validation pool into disjoint selection and
  reporting subsets.
- `factorized_capacity_pilot.json` — seed-41 fixed-mask capacity pilot at active
  residual capacities 960/512/256/128 plus concept-only and matched binary
  control endpoints.

All active Stage 2 configs keep:

```json
"evaluate_official_test": false,
"official_test_release": false
```

The repository intentionally does not include a ready-to-run confirmatory or
official-test config. Create it only after the concept subset, retained
capacities, checkpoint rule, probe hyperparameters, and intervention definitions
are frozen.

## Archived configs

- `archive/factorized_corruption_legacy.json` — residual-corruption experiments
  removed from the primary diagnostic path.
- `archive/concept_pilot_joint_legacy.json` — old joint autoencoder concept pilot;
  replaced by the decoder-free standalone predictor.
- `archive/factorized_concepts_preconfirmatory_legacy.json` — old mixed
  three-seed config that does not implement the final confirmatory matrix.

Archived configs contain `"archived": true` and use validation-only release
flags. Their outputs should go under `outputs/archive/`.
