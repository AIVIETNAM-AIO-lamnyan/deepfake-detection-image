import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_preprocessing_transform():
    return transforms.Compose([
        transforms.Resize(
            (299, 299),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


