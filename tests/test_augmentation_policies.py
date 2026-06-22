import unittest

import albumentations as A
import numpy as np

from augmentation_policies import (
    ALL_POLICIES,
    PAPER_SINGLE_POLICIES,
    build_augmentation,
)


class AugmentationPoliciesTest(unittest.TestCase):
    def test_paper_pool_contains_18_operations(self):
        self.assertEqual(len(PAPER_SINGLE_POLICIES), 18)
        self.assertEqual(len(set(PAPER_SINGLE_POLICIES)), 18)

    def test_every_policy_preserves_image_mask_shape(self):
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        image[64:160, 72:152] = 160
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[72:152, 80:144] = 1
        for policy in ALL_POLICIES:
            with self.subTest(policy=policy):
                transform = A.Compose(
                    [A.Resize(224, 224), *build_augmentation(policy, 224)]
                )
                result = transform(image=image, mask=mask)
                self.assertEqual(result["image"].shape, (224, 224, 3))
                self.assertEqual(result["mask"].shape, (224, 224))
                self.assertTrue(set(np.unique(result["mask"])).issubset({0, 1}))

    def test_speckle_noise_does_not_change_mask(self):
        image = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 1
        transform = A.Compose(build_augmentation("speckle_noise", 64))
        for _ in range(5):
            result = transform(image=image, mask=mask)
            self.assertTrue(np.array_equal(result["mask"], mask))


if __name__ == "__main__":
    unittest.main()
