import unittest

from run_factorized import validate_official_test_release


class OfficialTestReleaseGuardTest(unittest.TestCase):
    def test_validation_only_config_is_allowed(self):
        validate_official_test_release(
            {
                "evaluate_official_test": False,
                "official_test_release": False,
                "experiments": [{"name": "pilot_factorized_m960"}],
            }
        )

    def test_test_evaluation_requires_release(self):
        with self.assertRaises(PermissionError):
            validate_official_test_release(
                {
                    "evaluate_official_test": True,
                    "official_test_release": False,
                    "experiments": [{"name": "confirmatory_factorized_m960"}],
                }
            )

    def test_release_requires_confirmatory_names(self):
        with self.assertRaises(PermissionError):
            validate_official_test_release(
                {
                    "evaluate_official_test": True,
                    "official_test_release": True,
                    "experiments": [{"name": "pilot_factorized_m960"}],
                }
            )

    def test_explicit_confirmatory_release_is_allowed(self):
        validate_official_test_release(
            {
                "evaluate_official_test": True,
                "official_test_release": True,
                "experiments": [
                    {"name": "confirmatory_factorized_m960"},
                    {"name": "confirmatory_control_m960"},
                ],
            }
        )


try:
    import tensorflow as tf

    from model.model_factorized_lite import build_factorized_lite_autoencoder
except ImportError:  # pragma: no cover - exercised on CPU-only lightweight CI
    tf = None
    build_factorized_lite_autoencoder = None


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class FactorizedCapacityInvariantTest(unittest.TestCase):
    def _build(self, mode, residual_channels):
        tf.keras.backend.clear_session()
        return build_factorized_lite_autoencoder(
            img_shape=(64, 64, 3),
            concept_dim=8,
            residual_channels=residual_channels,
            max_residual_channels=15,
            condition_channels=2,
            base_channels=8,
            max_channels=32,
            mode=mode,
            residual_dropout=0.0,
            residual_noise_std=0.0,
        )

    def test_parameter_counts_do_not_change_with_active_capacity(self):
        _, encoder_15, decoder_15 = self._build("concept", 15)
        encoder_params = encoder_15.count_params()
        decoder_params = decoder_15.count_params()

        for active in (8, 4, 2):
            _, encoder, decoder = self._build("concept", active)
            self.assertEqual(encoder.count_params(), encoder_params)
            self.assertEqual(decoder.count_params(), decoder_params)
            self.assertEqual(encoder.output["residual"].shape[-1], 15)
            self.assertEqual(decoder.input[0].shape[-1], 15)

    def test_matched_control_uses_same_parameter_budget(self):
        _, concept_encoder, concept_decoder = self._build("concept", 15)
        _, control_encoder, control_decoder = self._build("control", 15)
        self.assertEqual(concept_encoder.count_params(), control_encoder.count_params())
        self.assertEqual(concept_decoder.count_params(), control_decoder.count_params())


if __name__ == "__main__":
    unittest.main()
