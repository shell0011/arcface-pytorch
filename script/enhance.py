import numpy as np
import cv2
import os
from albumentations import (
    RandomRotate90, Transpose, ShiftScaleRotate, Blur, 
    OpticalDistortion, CLAHE, GaussNoise, MotionBlur, 
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, 
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, 
    RandomContrast, RandomBrightness, HorizontalFlip, OneOf, Compose, ToGray,
    ElasticTransform
)

def strong_aug(p=1):
    return Compose([
        # RandomRotate90(),
        # Flip(),
        # Transpose(),
        HorizontalFlip(p=0.65),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.6),
        # ElasticTransform(p=1),
        OneOf([
            MotionBlur(p=.4),
            MedianBlur(blur_limit=3, p=.5),
            Blur(blur_limit=3, p=.5), 
        ], p=0.5),
        ShiftScaleRotate(shift_limit=0.01, scale_limit=0.01, rotate_limit=30, p=.8),
        OneOf([
            OpticalDistortion(p=0.4),
            # GridDistortion(p=.1),
            # IAAPiecewiseAffine(p=0.3),
        ], p=0.9),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            # IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.6),
        HueSaturationValue(p=0.3),
        ToGray(p=0.75),
    ], p=p)


root_path = "/home/lihebeizi/data/FaceRegDataset/train_enhanced"
dataset_dict = {}
for root, dirs, files in os.walk(root_path):
    parent_dirname = os.path.basename(root)
    if parent_dirname not in dataset_dict:
        dataset_dict[parent_dirname] = []
    for file in files:
        file_path = os.path.join(root, file)
        dataset_dict[parent_dirname].append(file_path)

target_lenth = 20
aug  =  strong_aug(p=1)

for name in dataset_dict:
    alist = dataset_dict[name]
    alist_len = len(alist)
    if alist_len == 0:
        print(name)
        continue
    num_to_add = target_lenth - alist_len
    for i in range(num_to_add):
        image = cv2.imread(alist[i % alist_len], 1)
        if image is None:
            print(alist[i % alist_len])
            continue
        img_strong_aug = aug(image=image)['image']
        cv2.imwrite(f'{alist[i % alist_len]}.enhance.{i}.jpg', img_strong_aug)
    print(f'{name} added {str(num_to_add)}')
