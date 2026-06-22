"""Albumentations policies for the controlled BUSI augmentation comparison."""

import albumentations as A
import numpy as np


PAPER_SINGLE_POLICIES = (
    "brightness",
    "contrast",
    "elastic",
    "equalize",
    "horizontal_flip",
    "gaussian_blur",
    "gaussian_noise",
    "grid_distortion",
    "median_blur",
    "random_crop",
    "rotate",
    "saturation",
    "scale",
    "shear_x",
    "shear_y",
    "translate_x",
    "translate_y",
    "vertical_flip",
)

COMPARISON_POLICIES = (
    "none",
    "rotate",
    "translate_y",
    "scale",
    "horizontal_flip",
    "vertical_flip",
    "brightness",
    "contrast",
    "gaussian_noise",
    "elastic",
    "legacy",
    "trivial_all_1",
    "trivial_all_3",
    "speckle_noise",
)

ALL_POLICIES = (
    "none",
    *PAPER_SINGLE_POLICIES,
    "legacy",
    "trivial_all_1",
    "trivial_all_3",
    "speckle_noise",
)


class SpeckleNoise(A.ImageOnlyTransform):
    """Multiplicative Gaussian noise, included as an ultrasound-specific extension."""

    def __init__(self, sigma=0.1, p=0.5):
        super().__init__(p=p)
        self.sigma = sigma

    def apply(self, image, **params):
        noise = self.random_generator.normal(0.0, self.sigma, image.shape)
        if np.issubdtype(image.dtype, np.integer):
            maximum = np.iinfo(image.dtype).max
            result = image.astype(np.float32) * (1.0 + noise)
            return np.clip(result, 0, maximum).astype(image.dtype)
        return np.clip(image * (1.0 + noise), 0.0, 1.0).astype(image.dtype)

    def get_transform_init_args_names(self):
        return ("sigma",)


def paper_operations(image_size, force_apply=False):
    probability = 1.0 if force_apply else 0.5
    always = 1.0
    identity = (1.0, 1.0)
    return {
        "brightness": A.ColorJitter(
            brightness=(0.5, 1.5), contrast=identity, saturation=identity, hue=0, p=always
        ),
        "contrast": A.ColorJitter(
            brightness=identity, contrast=(0.5, 1.5), saturation=identity, hue=0, p=always
        ),
        "elastic": A.ElasticTransform(alpha=50, sigma=5, p=always),
        "equalize": A.Equalize(p=probability),
        "horizontal_flip": A.HorizontalFlip(p=probability),
        "gaussian_blur": A.GaussianBlur(
            blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=always
        ),
        "gaussian_noise": A.GaussNoise(std_range=(0.1, 0.1), mean_range=(0.0, 0.0), p=probability),
        "grid_distortion": A.GridDistortion(num_steps=5, distort_limit=0.03, p=always),
        "median_blur": A.MedianBlur(blur_limit=3, p=probability),
        "random_crop": A.RandomResizedCrop(
            size=(image_size, image_size), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=always
        ),
        "rotate": A.Rotate(limit=(-30, 30), p=always),
        "saturation": A.ColorJitter(
            brightness=identity, contrast=identity, saturation=(0.5, 1.5), hue=0, p=always
        ),
        "scale": A.Affine(scale=(0.8, 1.2), keep_ratio=True, p=always),
        "shear_x": A.Affine(shear={"x": (-30, 30), "y": (0, 0)}, p=always),
        "shear_y": A.Affine(shear={"x": (0, 0), "y": (-30, 30)}, p=always),
        "translate_x": A.Affine(
            translate_percent={"x": (-0.2, 0.2), "y": (0, 0)}, p=always
        ),
        "translate_y": A.Affine(
            translate_percent={"x": (0, 0), "y": (-0.2, 0.2)}, p=always
        ),
        "vertical_flip": A.VerticalFlip(p=probability),
    }


def build_augmentation(policy, image_size):
    if policy not in ALL_POLICIES:
        raise ValueError(f"Unknown augmentation policy: {policy}")
    if policy == "none":
        return []
    if policy == "legacy":
        return [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=20,
                p=0.5,
            ),
        ]
    if policy == "speckle_noise":
        return [SpeckleNoise(sigma=0.1, p=0.5)]
    if policy.startswith("trivial_all_"):
        operation_count = int(policy.rsplit("_", 1)[1])
        pool = list(paper_operations(image_size, force_apply=True).values())
        return [A.SomeOf(pool, n=operation_count, replace=False, p=1.0)]
    return [paper_operations(image_size)[policy]]


def build_train_transform(policy, image_size, normalize, to_tensor):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            *build_augmentation(policy, image_size),
            normalize,
            to_tensor,
        ]
    )
