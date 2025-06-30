import monai
import torch
import numpy as np
from monai.transforms import (
    CropForeground,
    RandRotate,
    RandFlip,
    RandRotate90,
    RandAdjustContrast,
    RandHistogramShift,
    NormalizeIntensity,
    RandStdShiftIntensity,
    RandScaleIntensity,
    RandZoom,
    SpatialPad,
    Compose,
    Orientation,
    RandShiftIntensity,
    Spacing,
    CenterSpatialCrop,
)

from monai.transforms import (
    CropForegroundd,
    RandRotated,
    RandFlipd,
    RandRotate90d,
    RandAdjustContrastd,
    RandHistogramShiftd,
    NormalizeIntensityd,
    RandStdShiftIntensityd,
    RandScaleIntensityd,
    RandZoomd,
    SpatialPadd,
    Orientationd,
    RandShiftIntensityd,
    Spacingd,
    CenterSpatialCropd,
    ScaleIntensityRangePercentiles,
    ScaleIntensityRangePercentilesd,
    Affine,
    CastToType,
    )



size = (160, 192, 160)


train_transforms = Compose(
    [
    RandRotate(prob=0.3, range_x=0.2),
    RandRotate(prob=0.3, range_y=0.2),
    RandRotate(prob=0.3, range_z=0.2),
    ScaleIntensityRangePercentiles(lower=5, upper=95, b_min=0, b_max=255, clip=False, relative=False, channel_wise=True),
    CastToType(dtype=torch.float32),
    SpatialPad(spatial_size=size, mode='constant', constant_values=0),
    ]
)

valid_transforms = Compose(
    [

    ScaleIntensityRangePercentiles(lower=5, upper=95, b_min=0, b_max=255, clip=False, relative=False, channel_wise=True),
    CastToType(dtype=torch.float32),
    SpatialPad(spatial_size=size, mode='constant', constant_values=0),
    ]
)




