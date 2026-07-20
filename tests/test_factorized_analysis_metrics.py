import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from factorized_analysis import build_bird_bboxes, region_change_metrics


class RegionChangeMetricsTest(unittest.TestCase):
    def test_target_and_top_fraction_enrichment(self):
        pixel_map = np.zeros((2, 4, 4), dtype=np.float32)
        region = np.zeros((2, 4, 4), dtype=bool)
        region[:, :2, :2] = True
        pixel_map[:, :2, :2] = 1.0
        pixel_map[:, 2:, 2:] = 0.1
        result = region_change_metrics(
            pixel_map,
            region,
            np.asarray([True, True]),
            top_fraction=0.25,
        )
        valid = result["valid"]
        self.assertTrue(valid.all())
        self.assertTrue(np.all(result["enrichment"][valid] > 1.0))
        np.testing.assert_allclose(result["top_inside_fraction"][valid], 1.0)
        np.testing.assert_allclose(result["area_fraction"][valid], 0.25)

    def test_cub_bbox_is_resized_without_becoming_a_segmentation_claim(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images" / "class").mkdir(parents=True)
            Image.new("RGB", (20, 10)).save(root / "images" / "class" / "bird.jpg")
            (root / "images.txt").write_text("1 class/bird.jpg\n", encoding="utf-8")
            (root / "bounding_boxes.txt").write_text(
                "1 1.0 1.0 10.0 5.0\n", encoding="utf-8"
            )
            mask = build_bird_bboxes(root, [1], output_shape=(10, 20))
            self.assertEqual(mask.shape, (1, 10, 20))
            self.assertEqual(int(mask.sum()), 50)


if __name__ == "__main__":
    unittest.main()
