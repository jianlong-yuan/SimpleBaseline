import os, collections
import logging
import numpy as np
import torch

from PIL import Image
#from torchsummary import summary
from thop import profile
import copy
import pickle
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from segmentron.config import cfg
from .distributed import get_rank
import cv2
import torch.nn.functional as F


__all__ = ['get_color_pallete', 'print_iou', 'set_img_color',
           'show_prediction', 'show_colorful_images', 'save_colorful_images']


def print_iou(iu, mean_pixel_acc, class_names=None, show_no_back=False):
    n = iu.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i + 1)
        else:
            cls = '%d %s' % (i + 1, class_names[i])
        # lines.append('%-8s: %.3f%%' % (cls, iu[i] * 100))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    if show_no_back:
        lines.append('mean_IU: %.3f%% || mean_IU_no_back: %.3f%% || mean_pixel_acc: %.3f%%' % (
            mean_IU * 100, mean_IU_no_back * 100, mean_pixel_acc * 100))
    else:
        lines.append('mean_IU: %.3f%% || mean_pixel_acc: %.3f%%' % (mean_IU * 100, mean_pixel_acc * 100))
    lines.append('=================================================')
    line = "\n".join(lines)

    print(line)

@torch.no_grad()
def show_flops_params(model, device, input_shape=[1, 3, 1024, 1024]):
    # summary(model, tuple(input_shape[1:]), device=device)
    input = torch.randn(*input_shape).to(torch.device(device))
    model_clone = pickle.loads(pickle.dumps(model))

    flops, params = profile(model_clone, inputs=(input,), verbose=False)

    logging.info('{} flops: {:.3f}G input shape is {}, params: {:.3f}M'.format(
        model.__class__.__name__, flops / 1000000000, input_shape[1:], params / 1000000))

def count_parameters_in_MB(model):
    params = np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "aux" not in name) / 1e6
    logging.info('{} flops: params: {:.3f}M'.format(
        model.__class__.__name__, params))


def set_img_color(img, label, colors, background=0, show255=False):
    for i in range(len(colors)):
        if i != background:
            img[np.where(label == i)] = colors[i]
    if show255:
        img[np.where(label == 255)] = 255

    return img


def show_prediction(img, pred, colors, background=0):
    im = np.array(img, np.uint8)
    set_img_color(im, pred, colors, background)
    out = np.array(im)

    return out


def show_colorful_images(prediction, palettes):
    im = Image.fromarray(palettes[prediction.astype('uint8').squeeze()])
    im.show()


def save_colorful_images(prediction, filename, output_dir, palettes):
    '''
    :param prediction: [B, H, W, C]
    '''
    im = Image.fromarray(palettes[prediction.astype('uint8').squeeze()])
    fn = os.path.join(output_dir, filename)
    out_dir = os.path.split(fn)[0]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    im.save(fn)


def get_color_pallete(npimg, dataset='cityscape'):
    """Visualize image.

    Parameters
    ----------
    npimg : numpy.ndarray
        Single channel image with shape `H, W, 1`.
    dataset : str, default: 'pascal_voc'
        The dataset that model pretrained on. ('pascal_voc', 'ade20k')
    Returns
    -------
    out_img : PIL.Image
        Image with color pallete
    """
    # recovery boundary
    if dataset in ('pascal_voc', 'pascal_aug'):
        npimg[npimg == -1] = 255
    # put colormap
    if dataset == 'ade20k':
        npimg = npimg + 1
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(adepallete)
        return out_img
    elif dataset == 'cityscape':
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(cityscapepallete)
        return out_img
    out_img = Image.fromarray(npimg.astype('uint8'))
    out_img.putpalette(vocpallete)
    return out_img


def _getvocpallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete


vocpallete = _getvocpallete(256)

adepallete = [
    0, 0, 0, 120, 120, 120, 180, 120, 120, 6, 230, 230, 80, 50, 50, 4, 200, 3, 120, 120, 80, 140, 140, 140, 204,
    5, 255, 230, 230, 230, 4, 250, 7, 224, 5, 255, 235, 255, 7, 150, 5, 61, 120, 120, 70, 8, 255, 51, 255, 6, 82,
    143, 255, 140, 204, 255, 4, 255, 51, 7, 204, 70, 3, 0, 102, 200, 61, 230, 250, 255, 6, 51, 11, 102, 255, 255,
    7, 71, 255, 9, 224, 9, 7, 230, 220, 220, 220, 255, 9, 92, 112, 9, 255, 8, 255, 214, 7, 255, 224, 255, 184, 6,
    10, 255, 71, 255, 41, 10, 7, 255, 255, 224, 255, 8, 102, 8, 255, 255, 61, 6, 255, 194, 7, 255, 122, 8, 0, 255,
    20, 255, 8, 41, 255, 5, 153, 6, 51, 255, 235, 12, 255, 160, 150, 20, 0, 163, 255, 140, 140, 140, 250, 10, 15,
    20, 255, 0, 31, 255, 0, 255, 31, 0, 255, 224, 0, 153, 255, 0, 0, 0, 255, 255, 71, 0, 0, 235, 255, 0, 173, 255,
    31, 0, 255, 11, 200, 200, 255, 82, 0, 0, 255, 245, 0, 61, 255, 0, 255, 112, 0, 255, 133, 255, 0, 0, 255, 163,
    0, 255, 102, 0, 194, 255, 0, 0, 143, 255, 51, 255, 0, 0, 82, 255, 0, 255, 41, 0, 255, 173, 10, 0, 255, 173, 255,
    0, 0, 255, 153, 255, 92, 0, 255, 0, 255, 255, 0, 245, 255, 0, 102, 255, 173, 0, 255, 0, 20, 255, 184, 184, 0,
    31, 255, 0, 255, 61, 0, 71, 255, 255, 0, 204, 0, 255, 194, 0, 255, 82, 0, 10, 255, 0, 112, 255, 51, 0, 255, 0,
    194, 255, 0, 122, 255, 0, 255, 163, 255, 153, 0, 0, 255, 10, 255, 112, 0, 143, 255, 0, 82, 0, 255, 163, 255,
    0, 255, 235, 0, 8, 184, 170, 133, 0, 255, 0, 255, 92, 184, 0, 255, 255, 0, 31, 0, 184, 255, 0, 214, 255, 255,
    0, 112, 92, 255, 0, 0, 224, 255, 112, 224, 255, 70, 184, 160, 163, 0, 255, 153, 0, 255, 71, 255, 0, 255, 0,
    163, 255, 204, 0, 255, 0, 143, 0, 255, 235, 133, 255, 0, 255, 0, 235, 245, 0, 255, 255, 0, 122, 255, 245, 0,
    10, 190, 212, 214, 255, 0, 0, 204, 255, 20, 0, 255, 255, 255, 0, 0, 153, 255, 0, 41, 255, 0, 255, 204, 41, 0,
    255, 41, 255, 0, 173, 0, 255, 0, 245, 255, 71, 0, 255, 122, 0, 255, 0, 255, 184, 0, 92, 255, 184, 255, 0, 0,
    133, 255, 255, 214, 0, 25, 194, 194, 102, 255, 0, 92, 0, 255]

cityscapepallete = [
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def get_color_palletes(npimgs, dataset='cityscape'):
    batch_size = npimgs.shape[0]
    out_imgs = []
    for i in range(batch_size):
        npimg = npimgs[i, :, :]
        if dataset in ('pascal_voc', 'pascal_aug'):
            npimg[npimg == -1] = 255
        # put colormap
        if dataset == 'ade20k':
            npimg = npimg + 1
            out_img = Image.fromarray(npimg.astype('uint8'))
            out_img.putpalette(adepallete)
        elif dataset == 'cityscape':
            out_img = Image.fromarray(npimg.astype('uint8'))
            out_img.putpalette(cityscapepallete)
        else:
            out_img = Image.fromarray(npimg.astype('uint8'))
            out_img.putpalette(vocpallete)
        out_img = np.array(out_img.convert('RGB'))
        out_imgs.append(out_img)
    out = np.stack(out_imgs, axis=0)
    out = out.transpose(0, 3, 1, 2)
    return out



class tensorboard():
    def __init__(self, log_dir, distributed, data_name='pascal_voc'):
        self.distributed = distributed
        if self.distributed and get_rank()==0:
            os.makedirs(os.path.join(log_dir, "tb"), exist_ok=True)
            self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))
        elif not self.distributed:
            os.makedirs(os.path.join(log_dir, "tb"), exist_ok=True)
            self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))
        self.data_name = data_name
    def add_images(self, tag, image, label, logits, iter, using_nir=False, issigmoid=False, threshold=0.5):
        if self.distributed and get_rank() == 0:
            image_clone = copy.deepcopy(image).cpu()
            if isinstance(label, list):
                label_clone=[copy.deepcopy(l).cpu() for l in label]
            else:
                label_clone = copy.deepcopy(label).cpu()
            if issigmoid:
                pred_clone = F.sigmoid(logits).clone().detach().cpu()
                pred_clone[pred_clone>=threshold]=1
                pred_clone[pred_clone<threshold]=0
                pred_clone = torch.squeeze(pred_clone, dim=1)
            else:
                pred_clone  = copy.deepcopy(torch.argmax(logits, dim=1).cpu().detach())

            if isinstance(label_clone, list):
                label_clone = [torch.from_numpy(get_color_palletes(l.cpu().numpy(), dataset=self.data_name)) for l in label_clone]
            else:
                label_clone = torch.from_numpy(get_color_palletes(label_clone.cpu().numpy(), dataset=self.data_name))
            pred_clone  = torch.from_numpy(get_color_palletes(pred_clone.cpu().numpy(), dataset=self.data_name))

            if using_nir:
                nir_clone = image_clone[:, 3:, :, :]
                image_clone = image_clone[:, :3, :, :]
                image_clone = make_grid(image_clone, normalize=True, nrow=8, padding=2)
                pred_clone = make_grid(pred_clone, nrow=8, padding=2)
                nir_clone = make_grid(nir_clone, normalize=True, nrow=8, padding=2)

                if isinstance(label_clone, list):
                    label_clone = [make_grid(l, nrow=8, padding=2).float() for l in label_clone]
                    visual = torch.cat(
                        [image_clone.float(), nir_clone.float()] + label_clone + [pred_clone.float()], dim=1)
                else:
                    label_clone = make_grid(label_clone, nrow=8, padding=2)
                    visual = torch.cat([image_clone.float(), nir_clone.float(), label_clone.float(), pred_clone.float()], dim=1)
            else:
                image_clone = make_grid(image_clone, normalize=True, nrow=8, padding=2)
                pred_clone = make_grid(pred_clone, nrow=8, padding=2)
                if isinstance(label_clone, list):
                    label_clone = [make_grid(l, nrow=8, padding=2) for l in label_clone]
                    visual = torch.cat(
                        [image_clone.float()] + label_clone + [pred_clone.float()], dim=1)
                else:
                    label_clone = make_grid(label_clone, nrow=8, padding=2)
                    visual = torch.cat([image_clone.float(), label_clone.float(), pred_clone.float()], dim=1)

            self.writer.add_image(tag, visual, iter)
        elif not self.distributed:
            image_clone = copy.deepcopy(image).cpu()
            label_clone = copy.deepcopy(label).cpu()
            pred_clone = copy.deepcopy(torch.argmax(logits, dim=1).detach()).cpu()
            label_clone = torch.from_numpy(get_color_palletes(label_clone.cpu().numpy(), dataset=self.data_name))
            pred_clone = torch.from_numpy(get_color_palletes(pred_clone.cpu().numpy(), dataset=self.data_name))
            if using_nir:
                nir_clone = image_clone[:, 3:, :, :]
                image_clone = image_clone[:, :3, :, :]
                image_clone = make_grid(image_clone, normalize=True, nrow=8, padding=2)
                label_clone = make_grid(label_clone, nrow=8, padding=2)
                pred_clone = make_grid(pred_clone, nrow=8, padding=2)
                nir_clone = make_grid(nir_clone, normalize=True, nrow=8, padding=2)
                visual = torch.cat([image_clone.float(), nir_clone.float(), label_clone.float(), pred_clone.float()], dim=1)
            else:
                image_clone = make_grid(image_clone, normalize=True, nrow=8, padding=2)
                label_clone = make_grid(label_clone, nrow=8, padding=2)
                pred_clone = make_grid(pred_clone, nrow=8, padding=2)
                visual = torch.cat([image_clone.float(), label_clone.float(), pred_clone.float()], dim=1)
            self.writer.add_image(tag, visual, iter)
    def add_scalar(self, tag, scalar, iter):
        if self.distributed and get_rank() == 0:
            self.writer.add_scalar(tag, scalar, iter)
        elif not self.distributed:
            self.writer.add_scalar(tag, scalar, iter)
    def add_scalars(self, main_tag, tag_scalar_dict, iter):
        if self.distributed and get_rank() == 0:
            for k, v in tag_scalar_dict.items():
                self.add_scalar(os.path.join(main_tag, k), v, iter)
        elif not self.distributed:
            for k, v in tag_scalar_dict.items():
                self.add_scalar(os.path.join(main_tag, k), v, iter)
    def add_graph(self,  model, verbose=False):
        size = cfg.TRAIN.CROP_SIZE
        if isinstance(size, int):
            size1 = size
            size2 = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                 and isinstance(size[0], int) and isinstance(size[1], int) \
                 and size[0] > 0 and size[1] > 0:
            size1 = size[0]
            size2 = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        tensor = torch.rand(1, 4, size1, size2) if cfg.DATASET.USING_4C else \
            torch.rand(1, 3, size1, size2)
        if self.distributed and get_rank()==0:
            self.writer.add_graph(model, input_to_model=[tensor.cuda(), ], verbose=verbose)
        elif not self.distributed:
            self.writer.add_graph(model, input_to_model=[tensor.cuda(), ], verbose=verbose)
        del tensor
    def add_single_image(self, tag, image, iter, normalize):
        if self.distributed and get_rank() == 0:
            if len(image.shape)==3:
                image = torch.unsqueeze(image, dim=1)
            image = make_grid(image, nrow=8, padding=2, normalize=normalize)
            self.writer.add_image(tag, image, iter)
        elif not self.distributed:
            if len(image.shape)==3:
                image = torch.unsqueeze(image, dim=1)
            image = make_grid(image, nrow=8, padding=2, normalize=normalize)
            self.writer.add_image(tag, image, iter)

def save_predictions(images, names, root, issigmoid=False, threshold=0.5):
    org_images = images.clone().detach().cpu().numpy()
    if issigmoid:
        pred_clone = images.clone().detach().cpu()
        pred_clone[pred_clone >= threshold] = 1
        pred_clone[pred_clone < threshold] = 0
        tmp = torch.squeeze(pred_clone, dim=1).numpy()
    else:
        tmp = torch.argmax(images, 1).cpu().numpy()
    batch = tmp.shape[0]
    os.makedirs(root, exist_ok=True)
    assert len(names) == batch
    for ibatch in range(batch):
        tmp_image = tmp[ibatch, :, :]
        tmp_image = get_color_pallete(tmp_image)
        tmp_image.save(os.path.join(root, os.path.splitext(names[ibatch])[0]+'.png'))

    if cfg.TEST.SAVE_PREDICTION_NPY_PATH != '':
        os.makedirs(cfg.TEST.SAVE_PREDICTION_NPY_PATH, exist_ok=True)
        for ibatch in range(batch):
            image = org_images[ibatch, :, :, :]
            np.save(os.path.join(cfg.TEST.SAVE_PREDICTION_NPY_PATH, os.path.splitext(names[ibatch])[0] + '.npy'), image)


