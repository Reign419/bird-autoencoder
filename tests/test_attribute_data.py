import unittest

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - optional on metadata-only machines
    tf = None


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class FactorizedShapeTest(unittest.TestCase):
    def test_concept_and_control_interfaces(self):
        from model.model_factorized_lite import build_factorized_lite_autoencoder

        concept, encoder, decoder = build_factorized_lite_autoencoder(
            img_shape=(64, 64, 3), concept_dim=87, residual_channels=15, mode="concept"
        )
        self.assertEqual(encoder.output["residual"].shape, (None, 8, 8, 15))
        self.assertEqual(encoder.output["concepts"].shape, (None, 87))
        self.assertEqual(decoder.input[0].shape, (None, 8, 8, 15))
        self.assertEqual(decoder.input[1].shape, (None, 87))
        self.assertEqual(concept.output["reconstruction"].shape, (None, 64, 64, 3))

        control, control_encoder, control_decoder = build_factorized_lite_autoencoder(
            img_shape=(64, 64, 3), concept_dim=87, residual_channels=15, mode="control"
        )
        self.assertEqual(control_encoder.output["control"].shape, (None, 87))
        self.assertEqual(control_decoder.input[0].shape, (None, 8, 8, 15))
        self.assertEqual(control.output_shape, (None, 64, 64, 3))


if __name__ == "__main__":
    unittest.main()
