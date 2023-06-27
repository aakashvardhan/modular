from albumentations import (
    Compose,
    Normalize,
    RandomCrop,
    PadIfNeeded,
    CoarseDropout,
    Sequential,
    OneOf,
    HorizontalFlip,
)
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

class CustomResnetTransforms:
    def __init__(self, means, stds, train=True):
        if train:
            self.transformations = Compose(
                [
                    Normalize(mean=means, std=stds, always_apply=True),
                    Sequential(
                        [
                            PadIfNeeded(
                                min_height=40, min_width=40, always_apply=True
                            ),  # padding of 4 on each side of 32x32 image
                            RandomCrop(height=32, width=32, always_apply=True),
                        ]
                    ),
                    HorizontalFlip(),
                    CoarseDropout(
                        max_height=8,
                        max_width=8,
                        min_height=8,
                        min_width=8,
                        min_holes=1,
                        max_holes=1,
                        fill_value=means,
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            self.transformations = Compose(
                [
                    Normalize(mean=means, std=stds, always_apply=True),
                    ToTensorV2(),
                ]
            )

    def __call__(self, img):
        return self.transformations(image=np.array(img))["image"]
