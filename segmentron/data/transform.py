import random
import math
import numpy as np
import numbers
import collections
import cv2
import torch
from segmentron.utils.registry import Registry
from segmentron.config import cfg
import logging
from segmentron.data.randaug import Rand_Augment
PIPELINES_REGISTRY = Registry("PIPELINES")
PIPELINES_REGISTRY.__doc__ = """
Registry for segment model, i.e. the whole model.

The registered object will be called with `obj()`
and expected to return a `nn.Module` object.
"""

class Compose(object):
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, image, label):
        for t in self.segtransform:
            image, label = t(image, label)
        return image, label

@PIPELINES_REGISTRY.register()
class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, image, label):
        if not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray"
                                "[eg: data readed by cv2.imread()].\n"))
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        # if not len(label.shape) == 2:
        #     raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray labellabel with 2 dims.\n"))
        #
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        return image, label

@PIPELINES_REGISTRY.register()
class Normalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, scale255=True):
        mean = cfg.DATASET.MEAN

        std = cfg.DATASET.STD
        # if scale255:
        #     mean = [x * 255 for x in mean]
        #     std = [x * 255 for x in std]
        self.scale255 = scale255
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        if self.scale255:
            image = image / 255.

        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        return image, label



@PIPELINES_REGISTRY.register()
class Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = tuple(size)

    def __call__(self, image, label):
        image = cv2.resize(image, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.size[::-1], interpolation=cv2.INTER_NEAREST)
        return image, label

@PIPELINES_REGISTRY.register()
class ResizeLongerScale(object):
    def __init__(self, Longer_side):
        assert isinstance(Longer_side, int)
        self.longer_side = Longer_side

    def __call__(self, image, mask):
        max_side = max(image.shape[:2])
        scale = self.longer_side * 1.0 / max_side
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return image, mask



@PIPELINES_REGISTRY.register()
class ResizeLongerScaleForVal(object):
    def __init__(self, Longer_side):
        assert isinstance(Longer_side, int)
        self.longer_side = Longer_side

    def __call__(self, image, mask):
        max_side = max(image.shape[:2])
        scale = self.longer_side * 1.0 / max_side
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        return image, mask



@PIPELINES_REGISTRY.register()
class RandScale(object):
    # Randomly resize image & label with scale factor in [scale_min, scale_max]
    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("segtransform.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransform.RandScale() aspect_ratio param error.\n"))

    def __call__(self, image, label):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        return image, label

@PIPELINES_REGISTRY.register()
class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    # def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
    def __init__(self, crop_type='center', padding=None, ignore_label=255, crop_size=None):
        size = crop_size if crop_size is not None else cfg.TRAIN.CROP_SIZE
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = 0
        elif isinstance(padding, list):
            logging.info("{} uisng padding and * 255 !!!!!!!!!!!!!!".format(padding))
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            # if len(padding) != 3:
            #     raise (RuntimeError("padding channel is not equal with 3\n"))

            self.padding = [x * 255 for x in self.padding]

        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))



        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, image, label):
        h, w = label.shape[:2]
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransform.Crop() need padding while padding argument is None\n"))
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
            label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)
        h, w = label.shape[:2]
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        image = image[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        label = label[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        return image, label

@PIPELINES_REGISTRY.register()
class RandRotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransform.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
        return image, label



@PIPELINES_REGISTRY.register()
class Rotate(object):
    def __init__(self, ignore_label=255):
        self.ignore_label = ignore_label
    def __call__(self, image, label):
        rf = random.randint(0, 3)
        for i in range(rf):
            image = np.rot90(image)
            label = np.rot90(label)
        image = image.copy()
        label = label.copy()
        return image, label


@PIPELINES_REGISTRY.register()
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label

@PIPELINES_REGISTRY.register()
class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        return image, label

@PIPELINES_REGISTRY.register()
class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image, label):
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label

@PIPELINES_REGISTRY.register()
class RGB2BGR(object):
    # Converts image from RGB order to BGR order, for model initialized from Caffe
    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, label
@PIPELINES_REGISTRY.register()
class BGR2RGB(object):
    # Converts image from BGR order to RGB order, for model initialized from Pytorch
    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label


@PIPELINES_REGISTRY.register()
class Pad(object):
    def __init__(self, padding=None, ignore_label=255, crop_size=None):
        size = crop_size if crop_size is not None else cfg.TRAIN.CROP_SIZE
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))

        if padding is None:
            self.padding = 0
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))

        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, image, label):
        h, w = label.shape[:2]
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransform.Crop() need padding while padding argument is None\n"))
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.padding)
            label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_label)
        image = image[0:self.crop_h, 0:self.crop_w]
        label = label[0:self.crop_h, 0:self.crop_w]
        return image, label





@PIPELINES_REGISTRY.register()
class Randaug(Rand_Augment):
    def __init__(self, image_size = (1024, 1024), target_size = (1024,1024), Numbers=3, Magnitude=7, max_Magnitude=10, transforms=None, p=1.0):
        super(Randaug, self).__init__(image_size=image_size, target_size=target_size, Numbers=Numbers,
                                      Magnitude=Magnitude, max_Magnitude=max_Magnitude, transforms=transforms, p=p)


