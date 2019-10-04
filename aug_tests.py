import random

import matplotlib.pyplot as plt
from albumentations import (
    Blur, MotionBlur, MedianBlur, OneOf, Compose, RandomGamma
)
from albumentations.core.composition import KeypointParams
from torchvision import transforms

from triplet_dataset import TripletNetTexture, set_seed

mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

transform_train = transforms.Compose([
    # transforms.ColorJitter(brightness=(0.1, 4), contrast=0, saturation=0, hue=0),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


def strong_aug(p=0.5, blur=0.6, ):
    return Compose([
        OneOf([
            MotionBlur(p=0.6),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.9),
        RandomGamma(p=0.9, gamma_limit=(20, 150)),
    ], p=p, keypoint_params=KeypointParams(format='xy'))


def aug_triple(img1, img2, img3, alpha=1.7):
    aug = strong_aug(0.9)
    _seed = set_seed()
    alpha = 1.0
    points = None

    data = {"image": img1, "alpha": alpha, "keypoints": points}
    random.seed(_seed)
    _img1 = aug(**data)["image"]

    data = {"image": img2, "alpha": alpha, "keypoints": points}
    random.seed(_seed)
    _img2 = aug(**data)["image"]

    data = {"image": img3, "alpha": alpha, "keypoints": points}
    random.seed(_seed)
    _img3 = aug(**data)["image"]
    return _img1, _img2, _img3


aug = strong_aug(0.9)
experimental_train_dataset = TripletNetTexture(True, transform=None)
experimental_train_dataset.set_aug(aug)
alpha = 0.3  # @param {run: "auto", type:"slider", min:0, max: 5.0, step: 0.1}
ims = experimental_train_dataset[0][0]
# ims = aug_triple(*ims)

for im in ims:
    plt.figure()

    # vis_points(img["image"], img["keypoints"])
    # vis_points(img1["image"], img1["keypoints"])
    # vis_points(cv2.addWeighted(img["image"], 0.5, img1["image"], 0.5, 1.0), img1["keypoints"])

    plt.imshow(im)
plt.show()
