# Copyright (c) CAIRI AI Lab. All rights reserved

import cv2
import os
import os.path as osp
import logging
import subprocess
import sys
from collections import defaultdict, OrderedDict
from typing import Tuple

import torch
import torchvision
from torch import distributed as dist

import openstl
from .config_utils import Config

DATASET_ALIASES = {
    'seivr_vil': 'sevir_vil',
}


def canonicalize_dataname(dataname: str) -> str:
    """Normalize dataset aliases to canonical names."""
    if dataname is None:
        return dataname
    name = str(dataname).lower()
    return DATASET_ALIASES.get(name, name)


def build_default_config_path(dataname: str, method: str, config_root: str = './configs') -> str:
    """Build default config path from canonical dataset and method names."""
    return osp.join(config_root, canonicalize_dataname(dataname), f'{method}.py')


def get_cli_override_keys(parser, argv=None):
    """Return argument dest names explicitly passed by command line."""
    argv = sys.argv[1:] if argv is None else argv
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    overrides = set()
    for token in argv:
        if not token.startswith('-'):
            continue
        if token.startswith('--'):
            option = token.split('=', 1)[0]
        else:
            option = token if token in option_to_dest else token[:2]
        dest = option_to_dest.get(option)
        if dest:
            overrides.add(dest)
    return overrides


def collect_env():
    """Collect the information of the running environments."""
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available

    if cuda_available:
        from torch.utils.cpp_extension import CUDA_HOME
        env_info['CUDA_HOME'] = CUDA_HOME

        if CUDA_HOME is not None and os.path.isdir(CUDA_HOME):
            try:
                nvcc = os.path.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(
                    '"{}" -V | tail -n1'.format(nvcc), shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            env_info['GPU ' + ','.join(devids)] = name

    # gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
    try:
        import platform
        if platform.system() == 'Windows':
            gcc = b'GCC is not standard on Windows'
        else:
            gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
    except Exception:
        gcc = b'Unknown GCC'

    gcc = gcc.decode('utf-8').strip()
    env_info['GCC'] = gcc

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = torch.__config__.show()
    env_info['TorchVision'] = torchvision.__version__
    env_info['OpenCV'] = cv2.__version__

    env_info['openstl'] = openstl.__version__

    return env_info


def print_log(message):
    print(message)
    logging.info(message)


def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    return path


def get_dataset(dataname, config):
    from openstl.datasets import dataset_parameters
    from openstl.datasets import load_data

    dataname = canonicalize_dataname(dataname)
    if dataname not in dataset_parameters:
        raise KeyError(f'Unknown dataset "{dataname}" in dataset_parameters')

    config['dataname'] = dataname
    for k, v in dataset_parameters[dataname].items():
        if config.get(k) is None:
            config[k] = v

    return load_data(**config)


def measure_throughput(model, input_dummy):

    def get_batch_size(H, W):
        max_side = max(H, W)
        if max_side >= 128:
            bs = 10
            repetitions = 1000
        else:
            bs = 100
            repetitions = 100
        return bs, repetitions

    if isinstance(input_dummy, tuple):
        input_dummy = list(input_dummy)
        _, T, C, H, W = input_dummy[0].shape
        bs, repetitions = get_batch_size(H, W)
        _input = torch.rand(bs, T, C, H, W).to(input_dummy[0].device)
        input_dummy[0] = _input
        input_dummy = tuple(input_dummy)
    else:
        _, T, C, H, W = input_dummy.shape
        bs, repetitions = get_batch_size(H, W)
        input_dummy = torch.rand(bs, T, C, H, W).to(input_dummy.device)
    total_time = 0
    with torch.no_grad():
        for _ in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            if isinstance(input_dummy, tuple):
                _ = model(*input_dummy)
            else:
                _ = model(input_dummy)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (repetitions * bs) / total_time
    return Throughput


def load_config(filename: str = None):
    """load and print config"""
    print('loading config from ' + filename + ' ...')
    try:
        configfile = Config(filename=filename)
        config = configfile._cfg_dict
    except (FileNotFoundError, IOError):
        config = dict()
        print('warning: fail to load the config!')
    return config


def update_config(args, config, exclude_keys=None, cli_override_keys=None):
    """update the args dict with config values.

    Priority: explicit CLI > config > existing defaults.
    """
    exclude_keys = set(exclude_keys or [])
    cli_override_keys = set(cli_override_keys or [])

    assert isinstance(args, dict) and isinstance(config, dict)
    for k, v in config.items():
        if k in exclude_keys:
            continue

        if k in cli_override_keys and k in args and args[k] is not None:
            if args[k] != v:
                print(f'overwrite config key -- {k}: {v} -> {args[k]}')
            continue

        args[k] = v
    return args


def weights_to_cpu(state_dict: OrderedDict) -> OrderedDict:
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    # Keep metadata in state_dict
    state_dict_cpu._metadata = getattr(  # type: ignore
        state_dict, '_metadata', OrderedDict())
    return state_dict_cpu


def get_dist_info() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size
