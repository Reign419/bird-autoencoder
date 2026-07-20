import json
import unittest
from pathlib import Path

from stage1_experiments import build_run_name, prepare_experiments


ROOT = Path(__file__).resolve().parents[1]


class Stage1ExperimentConfigTest(unittest.TestCase):
    def test_legacy_latent_list_is_translated_to_shared_runner_format(self):
        experiments = prepare_experiments(
            {
                "model_name": "cnn",
                "latent_dims": [128, 256],
                "random_state": 42,
            }
        )
        self.assertEqual(
            [item["name"] for item in experiments],
            ["cnn_latent128", "cnn_latent256"],
        )
        self.assertEqual(
            [item["effective_latent_size"] for item in experiments],
            [128, 256],
        )

    def test_topology_config_expands_to_eleven_variants_by_three_seeds(self):
        config = json.loads(
            (ROOT / "configs" / "topology_ablation.json").read_text(encoding="utf-8")
        )
        experiments = prepare_experiments(config)
        self.assertEqual(len(experiments), 33)
        self.assertEqual(
            {item["training_seed"] for item in experiments},
            {42, 43, 44},
        )
        by_name = {item["name"]: item for item in experiments if item["training_seed"] == 42}
        self.assertEqual(by_name["A_spatial_identity"]["effective_latent_size"], 512)
        self.assertEqual(by_name["B_ordered_vector_interface"]["latent_shape"], "512")
        self.assertEqual(by_name["P_fixed_permutation"]["effective_latent_size"], 512)
        self.assertEqual(by_name["C_global_K256"]["effective_latent_size"], 256)
        self.assertEqual(by_name["D_spatial_K256"]["effective_latent_size"], 256)

    def test_historical_run_directory_reference_is_unchanged(self):
        experiment = prepare_experiments(
            {
                "training_seed": 42,
                "experiments": [
                    {
                        "name": "B ordered vector",
                        "model_name": "bottleneck_ablation",
                        "variant": "B",
                        "spatial_channels": 8,
                    }
                ],
            }
        )[0]
        self.assertEqual(
            build_run_name(experiment, "l1_ssim_edge", 42, "20260720_120000"),
            "B_ordered_vector_latent_B_K512_l1_ssim_edge_seed42_20260720_120000",
        )

    def test_invalid_spatial_compression_fails_before_training(self):
        with self.assertRaisesRegex(ValueError, "divisible by latent area"):
            prepare_experiments(
                {
                    "experiments": [
                        {
                            "model_name": "bottleneck_ablation",
                            "variant": "D",
                            "compressed_dim": 100,
                        }
                    ]
                }
            )


if __name__ == "__main__":
    unittest.main()
