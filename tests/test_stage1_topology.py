import unittest

import numpy as np

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - optional on metadata-only machines
    tf = None


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class Stage1TopologyInvariantTest(unittest.TestCase):
    def _build_ablation(self, variant):
        from model.model_bottleneck_ablation import build_bottleneck_ablation_autoencoder

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        return build_bottleneck_ablation_autoencoder(
            img_shape=(64, 64, 3),
            variant=variant,
            spatial_channels=8,
        )

    def test_A_and_B_are_parameter_and_function_equivalent(self):
        images = tf.random.stateless_uniform((2, 64, 64, 3), seed=(1, 2))
        model_a, _, _ = self._build_ablation("A")
        output_a = model_a(images, training=False).numpy()
        model_b, _, _ = self._build_ablation("B")
        output_b = model_b(images, training=False).numpy()
        self.assertEqual(model_a.count_params(), model_b.count_params())
        np.testing.assert_array_equal(output_a, output_b)

    def test_fixed_permutation_preserves_values_and_channel_blocks(self):
        from model.model_bottleneck_ablation import FixedPermutation

        source = tf.reshape(tf.range(24, dtype=tf.float32), (1, 24))
        transformed = FixedPermutation(seed=42, block_size=3)(source).numpy()[0]
        np.testing.assert_array_equal(np.sort(transformed), np.arange(24))
        blocks = transformed.reshape(-1, 3)
        np.testing.assert_array_equal(np.diff(blocks, axis=1), np.ones((8, 2)))

    def test_structured_and_spatial_interfaces_are_function_equivalent(self):
        from model.model_spatial_lite import build_spatial_lite_autoencoder
        from model.model_structured_vector_lite import (
            build_structured_vector_lite_autoencoder,
        )

        images = tf.random.stateless_uniform((2, 64, 64, 3), seed=(3, 4))
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        spatial, _, _ = build_spatial_lite_autoencoder(
            img_shape=(64, 64, 3), latent_channels=16
        )
        spatial_output = spatial(images, training=False).numpy()
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        structured, _, _ = build_structured_vector_lite_autoencoder(
            img_shape=(64, 64, 3), latent_dim=1024
        )
        structured_output = structured(images, training=False).numpy()
        self.assertEqual(spatial.count_params(), structured.count_params())
        np.testing.assert_array_equal(spatial_output, structured_output)


if __name__ == "__main__":
    unittest.main()
