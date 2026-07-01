import unittest

import numpy as np

from scripts.evaluation.evaluate_medical_metrics import (
    classification_metrics,
    normalized_confusion,
    refine_predictions,
    segmentation_metrics,
    segmentation_per_sample,
)


class MedicalMetricsTest(unittest.TestCase):
    def test_classification_invariants_and_specificity(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        predictions = np.array([0, 1, 1, 1, 2, 0])
        probabilities = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.3, 0.6, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.7, 0.2],
                [0.1, 0.2, 0.7],
                [0.6, 0.1, 0.3],
            ]
        )
        summary, per_class, matrix = classification_metrics(
            labels, predictions, probabilities
        )
        self.assertEqual(int(matrix.sum()), len(labels))
        self.assertEqual(sum(row["support"] for row in per_class), len(labels))
        self.assertAlmostEqual(summary["accuracy"], 4 / 6)
        malignant = per_class[1]
        self.assertAlmostEqual(malignant["sensitivity"], malignant["recall"])
        self.assertAlmostEqual(malignant["sensitivity"], 1.0)
        self.assertAlmostEqual(malignant["specificity"], 0.75)
        self.assertTrue(np.allclose(normalized_confusion(matrix).sum(axis=1), 1.0))

    def test_segmentation_all_and_lesion_only(self):
        targets = np.array([[[[1, 0], [0, 0]]], [[[0, 0], [0, 0]]]])
        predictions = np.array([[[[1, 0], [0, 0]]], [[[0, 0], [0, 0]]]])
        labels = np.array([0, 2])
        dice, iou = segmentation_per_sample(predictions, targets)
        summary, per_class = segmentation_metrics(labels, dice, iou)
        self.assertAlmostEqual(summary["dice_lesion"], 1.0)
        self.assertAlmostEqual(summary["iou_lesion"], 1.0)
        self.assertAlmostEqual(per_class["normal"]["dice"], 1.0)

    def test_prediction_refining(self):
        classes = np.array([0, 1])
        masks = np.zeros((2, 1, 10, 10), dtype=bool)
        masks[1, :, :2, :] = True
        refined_classes, refined_masks, areas = refine_predictions(
            classes, masks, area_threshold=0.005
        )
        self.assertEqual(refined_classes[0], 2)
        self.assertEqual(refined_classes[1], 1)
        self.assertFalse(refined_masks[0].any())
        self.assertAlmostEqual(areas[1], 0.2)


if __name__ == "__main__":
    unittest.main()
