
import re
import json

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torchio as tio
from einops import rearrange
import nibabel
import SimpleITK as sitk

import numpy as np
import multiprocessing as mp
from tqdm import tqdm

import torch
import torch.nn.functional as functional
from torch.utils.data import _utils, Dataset
from dataset.utils import identity, TorchioForegroundCropper, load_or_write_split, conserve_only_certain_labels


class Ruijin_3D(Dataset):
    def __init__(self, split="train", 
                 max_size=None,
                force_rewrite_split=False, 
                resize_to=(64, 128, 128)):
        super().__init__()
        with open('/mnt/data/oss_beijing/dailinrui/data/ruijin/records/dataset_crc_v2.json', 'rt') as f:
            self.data = json.load(f)
            self.data_keys = list(self.data.keys())

        self.base_folder = "/mnt/data/oss_beijing/dailinrui/data/ruijin"
        self.load_fn = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
        self.transforms = dict(
            resize=tio.Resize(resize_to) if resize_to is not None else tio.Lambda(identity),
            crop=TorchioForegroundCropper(crop_level="mask_foreground", crop_kwargs=dict(foreground_mask_label=None,
                                                                                         outline=(0, 0, 0))),
            normalize_mask=tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 11), include=["mask"])
        )

        self.split = split
        np.random.shuffle(self.data_keys)
        self.train_keys = self.data_keys[:round(len(self.data_keys) * 0.7)]
        self.val_keys = self.data_keys[round(len(self.data_keys) * 0.7):round(len(self.data_keys) * 0.8)]
        self.test_keys = self.data_keys[round(len(self.data_keys) * 0.8):]

        self.train_keys, self.val_keys, self.test_keys = load_or_write_split("/mnt/data/smart_health_02/dailinrui/data/pretrained/ldm/contrastive_exp_split",
                                                                             force_rewrite_split,
                                                                             train=self.train_keys, 
                                                                             val=self.val_keys, test=self.test_keys)
        self.split_keys = getattr(self, f"{split}_keys")[:max_size]
        self.context = np.load("/mnt/workspace/dailinrui/data/pretrained/ccdm/CT_report_abstract_BLS_PULSE-20bv5_short.npz")

    def __len__(self):
        return len(self.split_keys)

    def __getitem__(self, idx):
        item, context = self.data[self.split_keys[idx]], self.context[self.split_keys[idx]]
        totalseg, crcseg, text = map(lambda x: item[x], ["totalseg", "crcseg", "summary"])
        mask, crcmask = map(self.load_fn, [totalseg, crcseg])
        
        mask = conserve_only_certain_labels(mask)
        mask[crcmask > 0] = 11
        
        subject = tio.Subject(mask=tio.ScalarImage(tensor=mask[None]),)
        # crop
        subject = self.transforms["crop"](subject)
        # normalize
        subject = self.transforms["normalize_mask"](subject)
        # resize
        subject = self.transforms["resize"](subject)
        # random aug
        subject = self.transforms.get("augmentation", tio.Lambda(identity))(subject)
        subject = {k: v.data for k, v in subject.items()} | {"text": text, "context": torch.tensor(context)}

        return subject
        
        
class CLIPDataset(Ruijin_3D):
    def __init__(self,
                 split="train",
                 max_size=None,
                 spatial_size=(64, 128, 128),
                 force_collate_len=16):
        self.spatial_size = spatial_size
        self.collate_maxlen = force_collate_len
        super().__init__(split, max_size, resize_to=spatial_size)

    def __getitem__(self, idx):
        return super().__getitem__(idx)
    
    def collate_fn(self, batch):
        context = [b["context"] for b in batch]
        for b in batch: del b["context"]
        collated = _utils.collate.default_collate(batch)
        longest_context = max([b.shape[0] for b in context]) if self.collate_maxlen is None else self.collate_maxlen
        collated_context = torch.cat([torch.nn.functional.pad(c, (0, 0, 0, longest_context - c.shape[1])) if c.shape[1] <= longest_context else c[:, :longest_context] for c in context], dim=0)
        collated = {"mask": collated["mask"].float(), "text": collated["text"], "context": collated_context}
        return collated


if __name__ == "__main__":
    ...