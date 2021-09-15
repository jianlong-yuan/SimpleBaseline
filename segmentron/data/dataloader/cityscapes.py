"""Prepare Cityscapes dataset"""
import os
import torch
import numpy as np
import logging

from PIL import Image
from .seg_data_base import SegmentationDataset
from segmentron.config import cfg
from ..transform import *
import cv2
from segmentron.data.randaug import Rand_Augment

class CitySegmentation(SegmentationDataset):

    BASE_DIR = 'cityscapes'
    NUM_CLASS = 19

    def __init__(self, root='', split='train', mode=None, transform=None, **kwargs):
        super(CitySegmentation, self).__init__(root, split, mode, transform, **kwargs)
        root = 'data'

        if split == 'train_fine':
            _split_f = ["data/train_fine.txt"]

        elif split == 'train_extra':
            _split_f = ["data/train_extra.txt"]

        elif split == 'val_fine':
            _split_f = ["data/val_fine.txt"]

        elif split == 'trainval_fine':
            _split_f = ["data/train_fine.txt",
                        "data/val_fine.txt"]

        elif split == 'train_18label':
            _split_f = ["data/train_fine_18label.txt"]
        elif split == 'train_18unlabel':
            _split_f = ["data/train_fine_18unlabel.txt"]
        elif split == 'train_18unlabel_deeplabv3_1t':
            _split_f = [
                        "data/train_fine_18unlabel_pseudo_TTA_1t_deeplabv3.txt",
                        "data/train_fine_18label.txt"]

        elif split == 'train_fine_12label':
            _split_f = ["data/train_fine_12label.txt"]
        elif split == 'train_fine_12unlabel':
            _split_f = ["data/train_fine_12unlabel.txt"]
        elif split == 'train_fine_12_1t':
            _split_f = ["data/train_fine_12label.txt",
                        "data/train_fine_12unlabel_noisy_1t.txt"]

        elif split == 'train_fine_14label':
            _split_f = ["data/train_fine_14label.txt"]
        elif split == 'train_fine_14unlabel':
            _split_f = ["data/train_fine_14unlabel.txt"]
        elif split == 'train_fine_14_1t':
            _split_f = ["data/train_fine_14label.txt",
                        "data/train_fine_14unlabel_noisy_1t.txt"]

        elif split == 'train_fine_deeplabv3_1t':
            _split_f = ["data/train_extra_noisy_deeplabv3_1t.txt",
                        "data/train_fine.txt"]

        else:
            raise RuntimeError('Unknown dataset split.')
        self.mode = mode
        self.images = []
        self.masks = []
        self.nirs = []
        self.split = split

        lines= []
        for path in _split_f:
            lines += open(path, "r").readlines()

        for line in lines:
            if split != 'test':
                image_path, label_path = line.strip('\n').split(' ')
                _image = os.path.join(root, image_path)
                assert os.path.isfile(_image), _image
                self.images.append(_image)
                _mask = os.path.join(root, label_path)
                assert os.path.isfile(_mask), _mask
                self.masks.append(_mask)
            elif split == 'test':
                image_path = line.strip('\n')
                _image = os.path.join(root, image_path)
                assert os.path.isfile(_image), _image
                self.images.append(_image)

        if cfg.TRAIN.SEMI.USING_AUG_NOISY == 'randaug':
            logging.info('using random aug')
            self.aug = Rand_Augment(Numbers=cfg.TRAIN.SEMI.Numbers, Magnitude=cfg.TRAIN.SEMI.Magnitude, max_Magnitude=cfg.TRAIN.SEMI.max_Magnitude, p=cfg.TRAIN.SEMI.Prob)


        logging.info('{} data num {}'.format(self.split,len(self.images)))

    def __getitem__(self, index):
        img = cv2.imread(self.images[index], -1)
        if self.mode == 'test':
            mask = np.zeros([img.shape[0], img.shape[1]])
        else:
            mask = np.array(Image.open(self.masks[index]))

        if 'noisy' in self.masks[index]:
            if cfg.TRAIN.SEMI.USING_AUG_NOISY == 'randaug':
                img, mask = self.aug(img, mask)

        img, mask = self.transform(img, mask)
        mask = self._set_ignore_lable(mask)

        if self.mode == 'test':
            return {'image': img, 'label': 'None', 'name': os.path.basename(self.images[index])}

        mask = self._mask_transform(mask)
        return {'image': img, 'label': mask, 'name': os.path.basename(self.images[index]), 'is_noisy': float('noisy' in self.masks[index])}

    def _mask_transform(self, mask):
        target = mask
        return torch.LongTensor(np.array(target).astype('int32'))

    def _set_ignore_lable(self, mask):
        mask[mask == 255] = -1
        return mask

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        return ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',
                'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle')




if __name__ == '__main__':
    dataset = CitySegmentation()
