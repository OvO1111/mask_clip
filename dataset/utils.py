import os, os.path as path, yaml, pathlib as pb
import json, torchio as tio, torchvision as tv, shutil, nibabel as nib
import re, SimpleITK as sitk, scipy.ndimage as ndimage, numpy as np, multiprocessing as mp

import torch

from tqdm import tqdm
from einops import rearrange
from datetime import datetime
from omegaconf import OmegaConf
from functools import reduce, partial
from collections import OrderedDict, defaultdict


def identity(x, *a, **b): return x


def conserve_only_certain_labels(label, designated_labels=[1, 2, 3, 5, 6, 10, 55, 56, 57, 104]):
    if isinstance(label, np.ndarray):
        if designated_labels is None:
            return label.astype(np.uint8)
        label_ = np.zeros_like(label)
    elif isinstance(label, torch.Tensor):
        if designated_labels is None:
            return label.long()
        label_ = torch.zeros_like(label)
    for il, l in enumerate(designated_labels):
        label_[label == l] = il + 1
    return label_


def load_or_write_split(basefolder, force=False, **splits):
    splits_file = os.path.join(basefolder, "splits.json")
    if os.path.exists(splits_file) and not force:
        with open(splits_file, "r") as f:
            splits = json.load(f)
    else:
        with open(splits_file, "w") as f:
            json.dump(splits, f, indent=4)
    splits = list(splits.get(_) for _ in ["train", "val", "test"])
    return splits


def maybe_mkdir(p, destory_on_exist=False):
    if path.exists(p) and destory_on_exist:
        shutil.rmtree(p)
    os.makedirs(p, exist_ok=True)
    return pb.Path(p)


class TorchioForegroundCropper(tio.transforms.Transform):
    def __init__(self, crop_level="all", crop_kwargs=None,
                 *args, **kwargs):
        self.crop_level = crop_level
        self.crop_kwargs = crop_kwargs
        super().__init__(*args, **kwargs)

    def apply_transform(self, data: tio.Subject):
        # data: c h w d
        subject_ = {k: v.data for k, v in data.items()}
        type_ = {k: v.type for k, v in data.items()}

        if self.crop_level == "all":
            return data

        if self.crop_level == "patch":
            assert "image" in subject_
            image_ = subject_["image"]
            output_size = self.crop_kwargs["output_size"]
            
            pw = max((output_size[0] - image_.shape[1]) // 2 + 3, 0)
            ph = max((output_size[1] - image_.shape[2]) // 2 + 3, 0)
            pd = max((output_size[2] - image_.shape[3]) // 2 + 3, 0)
            image_ = torch.nn.functional.pad(image_, (pw, pw, ph, ph, pd, pd), mode='constant', value=0)

            (c, w, h, d) = image_.shape
            w1 = np.random.randint(0, w - output_size[0])
            h1 = np.random.randint(0, h - output_size[1])
            d1 = np.random.randint(0, d - output_size[2])
            
            padder = identity if pw + ph + pd == 0 else lambda x: torch.nn.functional.pad(x, (pw, pw, ph, ph, pd, pd), mode='constant', value=0)
            cropper = [slice(w1, w1 + output_size[0]), slice(h1, h1 + output_size[1]), slice(d1, d1 + output_size[2])]
            subject_ = {k: tio.Image(tensor=padder(v)[:, cropper[0], cropper[1], cropper[2]], type=type_[k]) for k, v in subject_.items()}
            
        outline = self.crop_kwargs.get("outline", [0] * 6)
        if isinstance(outline, int): outline = [outline] * 6
        if len(outline) == 3: outline = reduce(lambda x, y: x + y, zip(outline, outline))
        if self.crop_level == "image_foreground":
            assert "image" in subject_
            image_ = subject_["image"]
            s1, e1 = torch.where((image_ >= self.crop_kwargs.get('foreground_hu_lb', 0)).any(-1).any(-1).any(0))[0][[0, -1]]
            s2, e2 = torch.where((image_ >= self.crop_kwargs.get('foreground_hu_lb', 0)).any(1).any(-1).any(0))[0][[0, -1]]
            s3, e3 = torch.where((image_ >= self.crop_kwargs.get('foreground_hu_lb', 0)).any(1).any(1).any(0))[0][[0, -1]]
            cropper = [slice(max(0, s1 - outline[0]), min(e1 + 1 + outline[1], image_.shape[1])),
                       slice(max(0, s2 - outline[2]), min(e2 + 1 + outline[3], image_.shape[2])),
                       slice(max(0, s3 - outline[4]), min(e3 + 1 + outline[5], image_.shape[3]))]
            subject_ = {k: tio.Image(tensor=v[:, cropper[0], cropper[1], cropper[2]], type=type_[k]) for k, v in subject_.items()}
        
        if self.crop_level == "mask_foreground":
            assert "mask" in subject_
            mask_ = conserve_only_certain_labels(subject_["mask"], self.crop_kwargs.get("foreground_mask_label", None))
            s1, e1 = torch.where(mask_.any(-1).any(-1).any(0))[0][[0, -1]]
            s2, e2 = torch.where(mask_.any(1).any(-1).any(0))[0][[0, -1]]
            s3, e3 = torch.where(mask_.any(1).any(1).any(0))[0][[0, -1]]
            cropper = [slice(max(0, s1 - outline[0]), min(e1 + 1 + outline[1], mask_.shape[1])),
                       slice(max(0, s2 - outline[2]), min(e2 + 1 + outline[3], mask_.shape[2])),
                       slice(max(0, s3 - outline[4]), min(e3 + 1 + outline[5], mask_.shape[3]))]
            subject_ = {k: tio.Image(tensor=v[:, cropper[0], cropper[1], cropper[2]], type=type_[k]) for k, v in subject_.items()}
            
        return tio.Subject(subject_)