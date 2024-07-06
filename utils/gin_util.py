# ADOBE CONFIDENTIAL
# Copyright 2023 Adobe
# All Rights Reserved.
# NOTICE: All information contained herein is, and remains the property of Adobe
# and its suppliers, if any. The intellectual and technical concepts contained
# herein are proprietary to Adobe and its suppliers and are protected by all
# applicable intellectual property laws, including trade secret and copyright
# laws. Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from Adobe.

import os.path as osp
import string
from typing import Any

import gin

gin.add_config_file_search_path(osp.expanduser('~/code/burst-matting/configs'))

import pytorch_lightning as pl
import datetime

from absl import logging
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from utils.logger.gin_logger import GinLogCallback

TQDMProgressBar = gin.external_configurable(TQDMProgressBar)

GinTrainer = gin.external_configurable(pl.Trainer)
CheckpointCallback = gin.external_configurable(pl.callbacks.ModelCheckpoint)


def gin_trainer_kwargs():
    checkpointing = CheckpointCallback()
    gin_logger_callback = GinLogCallback()
    wandb_logger = WandbLogger(project='burst-matting',
                               name=gin_query_param_or_constant('exp_name'),
                               log_model=True)

    hyperparams = {
        k: gin_query_param_or_constant(k, return_none_if_fail=True)
        for k in (
            'torch.optim.Adam.lr',
            'torch.optim.Adam.betas',
            'torch.optim.Adam.eps',
            'Trainer.max_epochs',
            'Trainer.accumulate_grad_batches',
            'torch.optim.lr_scheduler.MultiStepLR.milestones',
            'torch.optim.lr_scheduler.MultiStepLR.gamma',
            'SyntheticMattingDataset.bg_movement',
            'MattingDataset.train_test_split',
            'unprocessing_settings.random_rgb_gain_range',
            'unprocessing_settings.add_noise',
            'SyntheticMattingDataset.num_fgs',
            'SingleFrameModel.network_type',
            'SingleFrameNetwork.num_channels',
            'SingleFrameNetwork.use_attention',
            'SingleFrameNetwork.upsample_type',
            'SingleFrameNetwork.activation_type',
            'SingleFrameNetwork.normalization_type',
            'SingleFrameNetwork.basic_unit',
            'MattingModel.use_srgb',
            # 'EDA.upsample_type',
        )
    }
    wandb_logger.log_hyperparams({
        k: v
        for k, v in hyperparams.items() if v is not None
    })

    return dict(
        logger=wandb_logger,
        callbacks=[checkpointing, gin_logger_callback,
                   TQDMProgressBar()],
        accelerator='gpu',
    )


def args_boilerplate(argv, parser, config_dict):
    args, unparsed_args = parser.parse_known_args(argv)
    # warn about unparsed arguments.
    if len(unparsed_args) > 0:
        logging.warning(
            f"Some arguments supplied are not parsed: {unparsed_args}.")

    gin.add_config_file_search_path(osp.dirname(args.config_file))

    # parse the gin file.
    config_file = args.config_file
    if config_dict is not None:
        # override the cmd line config file path if supplied using config dict.
        if 'config_file' in config_dict:
            config_file = config_dict['config_file']
    if config_file is None:
        raise ValueError(
            'Need to specify gin config file in commandline or in config_dict.'
        )
    if not config_file.endswith('gin'):
        config_file += '.gin'
    if not '/' in config_file:
        config_file = 'matting/' + config_file
    gin.parse_config_file(config_file)

    # store the logging path variables.
    exp_name = gin_query_param_or_constant('exp_name')
    logging_prefix = gin_query_param_or_constant('logging_prefix')
    logging_root = gin_query_param_or_constant('logging_root')

    # readin checkpoint config file.
    checkpoint_config_path = gin_query_param_or_constant(
        'checkpoint_config', return_none_if_fail=True)
    if hasattr(args, 'ckpt_config'):
        checkpoint_config_path = args.ckpt_config
    if 'checkpoint_config_path' in config_dict:
        checkpoint_config_path = config_dict['checkpoint_config_path']
    if checkpoint_config_path is not None:
        print(f'loading ckpt config {checkpoint_config_path}')
        gin.parse_config_file(checkpoint_config_path)

    # need to reset logging dir.
    gin_set_param_or_constant('exp_name', exp_name)
    gin_set_param_or_constant('logging_prefix', logging_prefix)
    gin_set_param_or_constant('logging_root', logging_root)

    if hasattr(args, 'ckpt'):
        gin_set_param_or_constant('checkpoint_path', args.ckpt)

    # apply the config dictionary
    # entries should be in gin syntax :
    # to set a.b = x, use config['a.b'] = x
    if config_dict is not None:
        for k, v in config_dict.items():
            gin_set_param_or_constant(k, v)

    # update the f-strings
    gin_format_param_string('datestring')
    gin_format_param_string('exp_name')
    gin_format_param_string('logging_prefix')

    # gin.finalize()
    # return parsed args for further customization.
    return args


@gin.configurable
def gin_join_path(input_list):
    return osp.join(*input_list)


def gin_query_param_or_constant(param_name: str,
                                return_none_if_fail=False) -> any:
    # if not gin.config_is_locked():
    #     raise RuntimeError('gin file has to be locked to perform query.')
    try:
        if '.' in param_name:
            param_value = gin.query_parameter(param_name)
        else:
            param_value = gin.query_parameter(param_name + '/macro.value')
        return param_value
    except ValueError:
        try:
            return eval(param_name)
        except Exception as e:
            if return_none_if_fail:
                return None
            else:
                raise ValueError(f'param {param_name} not found. ({e})')


def bind_param_with_reference(param_name, value):
    if type(value) == str:
        if value.startswith(('%', '@')):
            gin.parse_config(param_name + '=' + value)
        else:
            gin.bind_parameter(param_name, value)
    else:
        gin.bind_parameter(param_name, value)


def gin_set_param_or_constant(param_name: str, param_value: Any):
    if param_value is None:
        return
    if gin.config_is_locked():
        raise RuntimeError('gin file has to be unlocked to set values')
    if '.' in param_name:
        bind_param_with_reference(param_name, param_value)
    else:
        bind_param_with_reference(param_name + '/macro.value', param_value)


def gin_format_param_string(param_name: str) -> None:

    param_string = gin_query_param_or_constant(param_name)

    if type(param_string) is not str:
        raise RuntimeError(
            f' Value of "{param_name}" needs to be of string type. Got {type(param_string)} instead.'
        )

    param_value_keys = [t[1] for t in string.Formatter().parse(param_string)]

    format_dict = {}
    for k in param_value_keys:
        if k is not None:
            v = gin_query_param_or_constant(k)
            # need to avoid using '.' in format keys.
            # change them to _ instead.
            if '.' in k:
                new_k = k.replace('.', '_')
            else:
                new_k = k
            format_dict[new_k] = v

    if len(format_dict) > 0:
        # assumuing there's not '.' in param_strings other than keys for formating.
        param_string = param_string.replace('.', '_')
        param_string = param_string.format(**format_dict)

    # with gin.unlock_config():
    gin_set_param_or_constant(param_name, param_string)
