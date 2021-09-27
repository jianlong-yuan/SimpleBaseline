import time
import datetime
import os
import sys
import cv2
import numpy as np
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
from tqdm import tqdm
import logging, copy, math
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from tabulate import tabulate
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.solver.loss import get_segmentation_loss
from segmentron.solver.optimizer import get_optimizer
from segmentron.solver.lr_scheduler import get_scheduler
from segmentron.utils.distributed import *
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.filesystem import save_checkpoint
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.config import cfg
from segmentron.utils.visualize import tensorboard
from segmentron.data.dataset_zoo import get_dataset_pipelines
from segmentron.solver.loss import L2SP
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        # image transform
        train_transform_l = get_dataset_pipelines(cfg.PIPELINES.SEMI.TRAIN_PIPELINES_L)
        train_transform_u = get_dataset_pipelines(cfg.PIPELINES.SEMI.TRAIN_PIPELINES_U)
        val_transform = get_dataset_pipelines(cfg.PIPELINES.SEMI.VAL_PIPELINES)
        # dataset and dataloader
        train_dataset_l = get_segmentation_dataset(cfg.DATASET.NAME, split=cfg.DATASET.SEMI.TRAIN_SPLIT_L, mode='train',
                                                   transform=train_transform_l)
        train_dataset_u = get_segmentation_dataset(cfg.DATASET.NAME, split=cfg.DATASET.SEMI.TRAIN_SPLIT_U, mode='train',
                                                   transform=train_transform_u)
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split=cfg.DATASET.SEMI.VAL_SPLIT,
                                               mode=cfg.DATASET.MODE, transform=val_transform)
        self.iters_per_epoch = len(train_dataset_l) // (args.num_gpus * cfg.TRAIN.BATCH_SIZE)
        self.epochs = cfg.TRAIN.EPOCHS if cfg.TRAIN.MAX_ITERS is None else math.ceil(cfg.TRAIN.MAX_ITERS / self.iters_per_epoch)
        self.max_iters = self.epochs * self.iters_per_epoch if cfg.TRAIN.MAX_ITERS is None else cfg.TRAIN.MAX_ITERS
        self.classes = val_dataset.classes

        train_sampler_l = make_data_sampler(train_dataset_l, shuffle=True, distributed=args.distributed)
        train_batch_sampler_l = make_batch_data_sampler(train_sampler_l, cfg.TRAIN.SEMI.BATCH_SIZE_L, self.max_iters, drop_last=True)

        train_sampler_u = make_data_sampler(train_dataset_u, shuffle=True, distributed=args.distributed)
        train_batch_sampler_u = make_batch_data_sampler(train_sampler_u, cfg.TRAIN.SEMI.BATCH_SIZE_U, self.max_iters, drop_last=True)

        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, cfg.TEST.BATCH_SIZE, drop_last=False)

        self.train_loader_l = data.DataLoader(dataset=train_dataset_l,
                                              batch_sampler=train_batch_sampler_l,
                                              num_workers=cfg.DATASET.WORKERS,
                                              pin_memory=True)
        self.train_loader_u = data.DataLoader(dataset=train_dataset_u,
                                              batch_sampler=train_batch_sampler_u,
                                              num_workers=cfg.DATASET.WORKERS,
                                              pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)

        self.train_loader = zip(self.train_loader_l, self.train_loader_u)

        # create network
        self.model = get_segmentation_model().to(self.device)
        self.tb = tensorboard(cfg.VISUAL.OUTPUT_DIR, args.distributed)

        if cfg.MODEL.BN_TYPE not in ['BN']:
            logging.info('Batch norm type is {}, convert_sync_batchnorm is not effective'.format(cfg.MODEL.BN_TYPE))
        elif args.distributed and cfg.TRAIN.SYNC_BATCH_NORM:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            logging.info('SyncBatchNorm is effective!')
        else:
            logging.info('Not use SyncBatchNorm!')

        # create criterion
        self.criterion = get_segmentation_loss(cfg.MODEL.MODEL_NAME, use_ohem=cfg.SOLVER.OHEM,
                                               aux=cfg.SOLVER.AUX, aux_weight=cfg.SOLVER.AUX_WEIGHT,
                                               ignore_index=cfg.DATASET.IGNORE_INDEX).to(self.device)

        # optimizer, for model just includes encoder, decoder(head and auxlayer).
        self.optimizer = get_optimizer(self.model)

        # lr scheduling
        self.lr_scheduler = get_scheduler(self.optimizer, max_iters=self.max_iters,
                                          iters_per_epoch=self.iters_per_epoch)

        # resume checkpoint if needed
        self.start_epoch = 0
        if args.resume and os.path.isfile(args.resume):
            name, ext = os.path.splitext(args.resume)
            assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
            logging.info('Resuming training, loading {}...'.format(args.resume))
            resume_sate = torch.load(args.resume)
            self.model.load_state_dict(resume_sate['state_dict'])

            self.start_epoch = resume_sate['epoch']
            logging.info('resume train from epoch: {}'.format(self.start_epoch))
            if resume_sate['optimizer'] is not None and resume_sate['lr_scheduler'] is not None:
                logging.info('resume optimizer and lr scheduler from resume state..')
                self.optimizer.load_state_dict(resume_sate['optimizer'])
                self.lr_scheduler.load_state_dict(resume_sate['lr_scheduler'])

        if args.distributed:
            self.model = self.model.to(device=self.device)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=True)

        self.using_regular = cfg.SOLVER.REGULAR_MINE.USING
        if self.using_regular:
            self.regular = L2SP(self.model)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset_l.num_class, args.distributed)
        self.best_pred = 0.0

        # if get_rank()==0:
        #     save_checkpoint(self.model, 0, self.optimizer, self.lr_scheduler, is_best=False)
        logging.info('{}'.format(__file__))
        # logging.info(self.model)

    def train(self):
        torch.cuda.empty_cache()
        self.save_to_disk = get_rank() == 0
        epochs, max_iters, iters_per_epoch = self.epochs, self.max_iters, self.iters_per_epoch
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.iters_per_epoch
        val_per_iters = val_per_iters if cfg.TRAIN.VAL_PRE_ITERS is None else cfg.TRAIN.VAL_PRE_ITERS
        start_time = time.time()
        start_time_step = time.time()
        logging.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()
        iteration = self.start_epoch * iters_per_epoch if self.start_epoch > 0 else 0
        for data_l, data_u in self.train_loader:
            epoch = iteration // iters_per_epoch + 1
            iteration += 1
            images_l, targets_l = data_l['image'], data_l['label']
            images_u, targets_u = data_u['image'], data_u['label']
            images = torch.cat([images_l, images_u], dim=0)
            targets = torch.cat([targets_l, targets_u], dim=0)
            images = images.to(self.device)
            if isinstance(targets, list):
                targets_o = []
                for target in targets:
                    targets_o.append(target.to(self.device))
                targets = targets_o
            else:
                targets = targets.to(self.device)
            outputs = self.model(images)

            if cfg.SOLVER.LOSS_NAME == 'DynamicCEAndSCELoss':
                is_noisy = torch.cat([data_l['is_noisy'], data_u['is_noisy']], dim=0)
                loss_dict = self.criterion(outputs, targets, is_noisy)
            else:
                loss_dict = self.criterion(outputs, targets)

            losses = sum(loss for loss in loss_dict.values())

            if self.using_regular:
                losses = losses + self.regular(self.model)
            assert not torch.isnan(losses), losses

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and self.save_to_disk:
                logging.info(
                    "Epoch: {:d}/{:d} || Iters: {:d}/{:d}/{:d} || Lr: {:.6f} || "
                    "Loss: {:.4f} || Iter Time: {:.3f} || Cost Time: {} || Estimated Time: {}".format(
                        epoch, epochs, iteration % iters_per_epoch, iters_per_epoch, iteration,
                        self.optimizer.param_groups[0]['lr'], losses_reduced.item(),
                        time.time() - start_time_step,
                        str(datetime.timedelta(seconds=int(time.time() - start_time))),
                        eta_string))


            if iteration % cfg.TRAIN.SUMMARY_LOG_ITERS == 0 or iteration==1:
                self.tb.add_scalars('train', {'loss': losses_reduced, 'lr': self.optimizer.param_groups[0]['lr'], 'Iter_Time': time.time() - start_time_step}, iteration)

            del losses_reduced, images, targets, outputs, losses, loss_dict, loss_dict_reduced

            if iteration % val_per_iters == 0 and self.save_to_disk:
                save_checkpoint(self.model, epoch, self.optimizer, self.lr_scheduler, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation(iteration, self.model, scope='val')
                self.model.train()
            start_time_step = time.time()

        if not self.args.skip_val:
            self.validation(iteration, self.model, scope='val')
            self.model.train()
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logging.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation(self, epoch, model, scope='val'):
        torch.cuda.empty_cache()
        self.metric.reset()
        model.eval()
        for data in tqdm(self.val_loader):
            image, target, filename = data['image'], data['label'], data['name']
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                if cfg.DATASET.MODE == 'val' or cfg.TEST.CROP_SIZE is None:
                    output = model(image)[0]
                else:
                    input, size = self.input_process(image)
                    output = model(input)[0]
                    output = output[..., :size[0], :size[1]]

            self.metric.update(output, target)
        pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
        logging.info("[EVAL END] Epoch: {:d}, pixAcc: {:.3f}, mIoU: {:.3f}".format(epoch, pixAcc * 100, mIoU * 100))
        self.tb.add_scalars(scope, {'mIoU': mIoU, 'pixAcc': pixAcc}, epoch)
        log_dict = {}
        table = []
        for i, cls_name in enumerate(self.classes):
            log_dict[cls_name]= category_iou[i] * 100
            table.append([cls_name, category_iou[i] * 100])
        self.tb.add_scalars(scope, log_dict, epoch)

        headers = ['class id', 'class name', 'iou']
        logging.info('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid', showindex="always",
                                                           numalign='center', stralign='center')))
        synchronize()
        if self.best_pred < np.mean(category_iou) and self.save_to_disk and scope=='val':
            self.best_pred = np.mean(category_iou)
            logging.info('Epoch {} is the best model, best pixAcc: {:.3f}, mIoU: {:.3f}, mIoU_classes: {:.3f}, '
                         'save the model..'.format(epoch, pixAcc * 100, mIoU * 100, np.mean(category_iou)*100))
            save_checkpoint(model, epoch, is_best=True)

        del category_iou, mIoU, pixAcc, output, log_dict, table
        torch.cuda.empty_cache()

    def input_process(self, image):
        size = image.size()[2:]
        pad_height = cfg.TEST.CROP_SIZE[0] - size[0]
        pad_width = cfg.TEST.CROP_SIZE[1] - size[1]
        image = F.pad(image, [0, pad_height, 0, pad_width])
        return image, size


if __name__ == '__main__':
    args = parse_args()
    # get config
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'train'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    # setup python train environment, logger, seed..
    default_setup(args)

    # create a trainer and start train
    trainer = Trainer(args)
    trainer.train()
