import logging
import torch
from segmentron.modules.batch_norm import DSBN
from collections import OrderedDict
from segmentron.utils.registry import Registry
from ..config import cfg
import os

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for segment model, i.e. the whole model.

The registered object will be called with `obj()`
and expected to return a `nn.Module` object.
"""


def get_segmentation_model():
    """
    Built the whole model, defined by `cfg.MODEL.META_ARCHITECTURE`.
    """
    model_name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(model_name)()
    if cfg.TRAIN.USING_DSBN:
        model = DSBN.convert_dsbn(model)
        logging.info('DSSyncBatchNorm is effective!')
    load_model_pretrain(model)
    return model


def load_model_pretrain(model):
    if cfg.PHASE == 'train':
        if cfg.TRAIN.PRETRAINED_MODEL_PATH:
            assert os.path.exists(cfg.TRAIN.PRETRAINED_MODEL_PATH)
            logging.info('load pretrained model from {}'.format(cfg.TRAIN.PRETRAINED_MODEL_PATH))
            state_dict_to_load = torch.load(cfg.TRAIN.PRETRAINED_MODEL_PATH, map_location=lambda storage, loc: storage)
            if 'state_dict' in state_dict_to_load.keys():
                state_dict_to_load = state_dict_to_load['state_dict']
            state_dict_suitable = state_dict_to_load
            msg = model.load_state_dict(state_dict_suitable, strict=True)
            logging.info(msg)
    else:
        if cfg.TEST.TEST_MODEL_PATH:
            logging.info('load test model from {}'.format(cfg.TEST.TEST_MODEL_PATH))
            state_dict_to_load = torch.load(cfg.TEST.TEST_MODEL_PATH, map_location=lambda storage, loc: storage)
            if 'state_dict' in state_dict_to_load.keys():
                state_dict_to_load = state_dict_to_load['state_dict']
            msg = model.load_state_dict(state_dict_to_load, strict=True)
            logging.info(msg)