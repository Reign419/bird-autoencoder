import argparse
import unittest

import numpy as np
import pandas as pd

from analysis.concept_probe import LEGACY_COLUMNS, legacy_tables, run_probe


class ConceptProbeCompatibilityTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        train_x = rng.normal(size=(160, 8)).astype(np.float32)
        evaluation_x = rng.normal(size=(80, 8)).astype(np.float32)
        self.train = {
            "residual": train_x,
            "labels": np.stack(
                [train_x[:, 0] > 0, train_x[:, 1] + train_x[:, 2] > 0], axis=1
            ).astype(np.float32),
            "weights": np.ones((160, 2), dtype=np.float32),
            "attribute_ids": np.asarray([1, 2]),
        }
        self.evaluation = {
            "residual": evaluation_x,
            "labels": np.stack(
                [
                    evaluation_x[:, 0] > 0,
                    evaluation_x[:, 1] + evaluation_x[:, 2] > 0,
                ],
                axis=1,
            ).astype(np.float32),
            "weights": np.ones((80, 2), dtype=np.float32),
            "attribute_ids": np.asarray([1, 2]),
        }
        self.definitions = pd.DataFrame(
            {
                "attribute_id": [1, 2],
                "group": ["g1", "g2"],
                "value": ["a", "b"],
            }
        ).set_index("attribute_id")
        self.args = argparse.Namespace(
            evaluation_split="validation",
            seed=42,
            null_seed=1042,
            null_repeats=3,
            jobs=1,
            linear_alpha=0.0001,
            max_iter=1000,
            mlp_hidden_layers=(16, 8),
            mlp_alpha=0.0001,
            mlp_batch_size=32,
            mlp_learning_rate=0.001,
            mlp_max_iter=20,
            mlp_validation_fraction=0.15,
            mlp_patience=3,
        )

    def test_linear_null_and_legacy_schema(self):
        real, null = run_probe(
            "linear",
            self.train,
            self.evaluation,
            self.definitions,
            self.args,
        )
        self.assertEqual(len(real), 2)
        self.assertEqual(len(null), 6)
        self.assertIn("average_precision_fdr_q", real.columns)
        legacy, groups = legacy_tables(real)
        self.assertEqual(legacy.columns.tolist(), LEGACY_COLUMNS)
        self.assertEqual(groups.columns.tolist(), [
            "group",
            "n_attributes",
            "macro_probe_ap",
            "macro_probe_ap_lift",
            "macro_probe_balanced_accuracy",
        ])

    def test_detailed_schema_is_stable_without_null_repeats(self):
        arguments = argparse.Namespace(**{**vars(self.args), "null_repeats": 0})
        real, _ = run_probe(
            "linear",
            self.train,
            self.evaluation,
            self.definitions,
            arguments,
        )
        for metric in ("average_precision", "balanced_accuracy", "roc_auc"):
            self.assertIn(f"{metric}_fdr_q", real.columns)


if __name__ == "__main__":
    unittest.main()
