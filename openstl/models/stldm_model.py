import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def _get_stldm_template(dataname: str):
    """Return dataset-specific STLDM template config without modifying core STLDM code."""
    from stldm.config import STLDM_SEVIR

    name = str(dataname).lower()
    if 'sevir' in name:
        return copy.deepcopy(STLDM_SEVIR), {'t_in': 13, 't_out': 12, 'c': 1, 'h': 128, 'w': 128}

    # CIKM template: keep official STLDM core params, only dataset shape defaults differ.
    cikm_cfg = copy.deepcopy(STLDM_SEVIR)
    return cikm_cfg, {'t_in': 5, 't_out': 10, 'c': 4, 'h': 112, 'w': 112}


class STLDM_Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        from stldm import n2n_setup

        dataname = str(kwargs.get('dataname', '')).lower()
        base_config, ds_default = _get_stldm_template(dataname)

        in_shape = kwargs.get('in_shape', None)
        if in_shape is not None and len(in_shape) >= 4:
            in_t, in_c, in_h, in_w = [int(v) for v in in_shape[:4]]
        else:
            in_t = ds_default['t_in']
            in_c = ds_default['c']
            in_h = ds_default['h']
            in_w = ds_default['w']

        self.pre_seq_length = int(kwargs.get('pre_seq_length', in_t))
        self.aft_seq_length = int(kwargs.get('aft_seq_length', ds_default['t_out']))
        if self.aft_seq_length <= 0:
            self.aft_seq_length = ds_default['t_out']

        self.in_channels = int(in_c)
        self.model_hw = (int(in_h), int(in_w))
        self._resize_warned = False

        base_config['vp_param']['shape_in'] = (
            self.pre_seq_length,
            self.in_channels,
            self.model_hw[0],
            self.model_hw[1],
        )
        base_config['vp_param']['shape_out'] = (
            self.aft_seq_length,
            self.in_channels,
            self.model_hw[0],
            self.model_hw[1],
        )

        self.stldm_net = n2n_setup['3D'](base_config)

    def _resize_to(self, tensor, target_hw):
        b, t, c, h, w = tensor.shape
        if (h, w) == target_hw:
            return tensor
        tensor = tensor.reshape(b * t, c, h, w)
        tensor = F.interpolate(tensor, size=target_hw, mode='bilinear', align_corners=False)
        return tensor.view(b, t, c, target_hw[0], target_hw[1])

    def _resize_for_model(self, tensor):
        if tensor.shape[-2:] != self.model_hw and not self._resize_warned:
            print(f'info: STLDM input resized from {tuple(tensor.shape[-2:])} to {self.model_hw} for model forward')
            self._resize_warned = True
        return self._resize_to(tensor, self.model_hw)

    def compute_loss(self, x, y):
        x_model = self._resize_for_model(x)
        y_model = self._resize_for_model(y)
        return self.stldm_net.compute_loss(x_model, y_model)

    def forward(self, x):
        orig_hw = tuple(x.shape[-2:])
        x_model = self._resize_for_model(x)
        y_pred, _ = self.stldm_net(x_model, include_mu=True)

        if tuple(y_pred.shape[-2:]) != orig_hw:
            y_pred = self._resize_to(y_pred, orig_hw)
        return y_pred
