# Copyright (c) CAIRI AI Lab. All rights reserved

import sys
import time
import os
import os.path as osp
from fvcore.nn import FlopCountAnalysis, flop_count_table

import torch

from openstl.methods import method_maps
from openstl.datasets import BaseDataModule
from openstl.utils import (get_dataset, measure_throughput, SetupCallback, EpochEndCallback, BestCheckpointCallback)

from lightning import seed_everything, Trainer
import lightning.pytorch.callbacks as lc


class BaseExperiment(object):
    """The basic class of PyTorch training and evaluation."""

    def __init__(self, args, dataloaders=None, strategy='auto'):
        """Initialize experiments (non-dist as an example)."""
        self.args = args
        self.config = self.args.__dict__
        self.method = None
        self.args.method = self.args.method.lower()
        self._dist = self.args.dist

        base_dir = args.res_dir if args.res_dir is not None else 'work_dirs'
        save_dir = osp.join(base_dir, args.ex_name if not args.ex_name.startswith(args.res_dir)
                            else args.ex_name.split(args.res_dir + '/')[-1])
        ckpt_dir = osp.join(save_dir, 'checkpoints')

        seed_everything(args.seed)
        self.data = self._get_data(dataloaders)
        self.method = method_maps[self.args.method](
            steps_per_epoch=len(self.data.train_loader),
            test_mean=self.data.test_mean,
            test_std=self.data.test_std,
            save_dir=save_dir,
            **self.config,
        )
        callbacks, self.save_dir = self._load_callbacks(args, save_dir, ckpt_dir)
        self.trainer = self._init_trainer(self.args, callbacks, strategy)

    def _is_global_zero(self):
        # Lightning spawns multiple processes in DDP; only rank 0 should print heavy model info.
        return int(os.environ.get('RANK', '0')) == 0

    def _init_trainer(self, args, callbacks, strategy):
        resolved_strategy = strategy
        if strategy == 'auto':
            world_size = int(os.environ.get('WORLD_SIZE', '1'))
            if isinstance(args.gpus, (list, tuple)):
                num_devices = len(args.gpus)
            else:
                num_devices = int(args.gpus)

            # STLDM uses stochastic branches and can have unused params in DDP steps.
            if args.method == 'stldm' and (num_devices > 1 or world_size > 1 or args.dist):
                resolved_strategy = 'ddp_find_unused_parameters_true'

        precision = '16-mixed' if getattr(args, 'fp16', False) else '32-true'

        return Trainer(
            devices=args.gpus,
            max_epochs=args.epoch,
            strategy=resolved_strategy,
            accelerator='gpu',
            precision=precision,
            callbacks=callbacks,
        )

    def _load_callbacks(self, args, save_dir, ckpt_dir):
        method_info = None
        if self._is_global_zero() and (not self.args.no_display_method_info):
            method_info = self.display_method_info(args)

        setup_callback = SetupCallback(
            prefix='train' if (not args.test) else 'test',
            setup_time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            save_dir=save_dir,
            ckpt_dir=ckpt_dir,
            args=args,
            method_info=method_info,
            argv_content=sys.argv + ["gpus: {}".format(torch.cuda.device_count())],
        )

        ckpt_callback = BestCheckpointCallback(
            monitor=args.metric_for_bestckpt,
            filename='best-{epoch:02d}-{val_loss:.3f}',
            mode='min',
            save_last=True,
            dirpath=ckpt_dir,
            verbose=True,
            every_n_epochs=args.log_step,
        )

        epochend_callback = EpochEndCallback()

        callbacks = [setup_callback, ckpt_callback, epochend_callback]
        if args.sched:
            callbacks.append(lc.LearningRateMonitor(logging_interval=None))
        return callbacks, save_dir

    def _get_data(self, dataloaders=None):
        """Prepare datasets and dataloaders."""
        if dataloaders is None:
            train_loader, vali_loader, test_loader = get_dataset(self.args.dataname, self.config)
        else:
            train_loader, vali_loader, test_loader = dataloaders

        vali_loader = test_loader if vali_loader is None else vali_loader
        return BaseDataModule(train_loader, vali_loader, test_loader)

    def train(self):
        self.trainer.fit(self.method, self.data, ckpt_path=self.args.ckpt_path if self.args.ckpt_path else None)

    def test(self):
        if self.args.test is True:
            load_path = self.args.ckpt_path if self.args.ckpt_path else osp.join(self.save_dir, 'checkpoints', 'best.ckpt')
            print(f'[test] loading checkpoint: {load_path}')
            ckpt = torch.load(load_path, map_location='cpu')

            if 'state_dict' in ckpt:
                self.method.load_state_dict(ckpt['state_dict'])
            else:
                try:
                    self.method.load_state_dict(ckpt)
                except RuntimeError:
                    self.method.model.load_state_dict(ckpt)

            # Keep CIKM evaluation metrics enabled for test mode.
            self.method.metric_list = ['mse', 'mae', 'rmse', 'pod', 'csi', 'hss']
            self.method.spatial_norm = True
            self.method.threshold = [0.5 / 255.0, 2.0 / 255.0, 5.0 / 255.0, 10.0 / 255.0, 30.0 / 255.0]
            self.method.dataname = 'weather_cikm'

            if hasattr(self.trainer, 'loggers'):
                self.trainer.loggers = []
            self.trainer.logger = None

        results = self.trainer.test(self.method, self.data)

        if self.args.test is True:
            print('\n' + '=' * 30)
            print('Final test metrics')
            print('=' * 30)
            save_path = 'test_metrics_result.txt'
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write('=== CIKM Test Metrics ===\n')
                if results and len(results) > 0:
                    for k, v in results[0].items():
                        print(f'  {k}: {v}')
                        f.write(f'{k}: {v}\n')
                else:
                    print('No metrics captured from trainer.test output.')
            print(f'Metrics saved to: {save_path}')

    def display_method_info(self, args):
        """Collect model info safely for logging."""
        device = torch.device(args.device)
        if args.device == 'cuda':
            if isinstance(args.gpus, (list, tuple)):
                assign_gpu = 'cuda:' + (str(args.gpus[0]) if len(args.gpus) == 1 else '0')
            else:
                assign_gpu = 'cuda:0'
            device = torch.device(assign_gpu)

        T, C, H, W = args.in_shape
        if args.method in ['simvp', 'tau', 'mmvp', 'wast', 'exprecast', 'stldm']:
            input_dummy = torch.ones(1, args.pre_seq_length, C, H, W).to(device)
        elif args.method == 'alphapre':
            _tmp_input = torch.ones(1, args.pre_seq_length, C, H, W).to(device)
            _tmp_gt = torch.ones(1, args.aft_seq_length, C, H, W).to(device)
            input_dummy = (_tmp_input, _tmp_gt)
        elif args.method == 'phydnet':
            _tmp_input1 = torch.ones(1, args.pre_seq_length, C, H, W).to(device)
            _tmp_input2 = torch.ones(1, args.aft_seq_length, C, H, W).to(device)
            _tmp_constraints = torch.zeros((49, 7, 7)).to(device)
            input_dummy = (_tmp_input1, _tmp_input2, _tmp_constraints)
        elif args.method in ['convlstm', 'predrnnpp', 'predrnn', 'mim', 'e3dlstm', 'mau']:
            Hp, Wp = H // args.patch_size, W // args.patch_size
            Cp = args.patch_size ** 2 * C
            _tmp_input = torch.ones(1, args.total_length, Hp, Wp, Cp).to(device)
            _tmp_flag = torch.ones(1, args.aft_seq_length - 1, Hp, Wp, Cp).to(device)
            input_dummy = (_tmp_input, _tmp_flag)
        elif args.method in ['swinlstm_d', 'swinlstm_b']:
            input_dummy = torch.ones(1, self.args.total_length, H, W, C).to(device)
        elif args.method == 'predrnnv2':
            Hp, Wp = H // args.patch_size, W // args.patch_size
            Cp = args.patch_size ** 2 * C
            _tmp_input = torch.ones(1, args.total_length, Hp, Wp, Cp).to(device)
            _tmp_flag = torch.ones(1, args.total_length - 2, Hp, Wp, Cp).to(device)
            input_dummy = (_tmp_input, _tmp_flag)
        elif args.method == 'prednet':
            input_dummy = torch.ones(1, 1, C, H, W, requires_grad=True).to(device)
        else:
            raise ValueError(f'Invalid method name {args.method}')

        dash_line = '-' * 80 + '\n'
        info = self.method.model.__repr__()

        # STLDM JIT-trace FLOPs is very slow in DDP and can look like a hang.
        if args.method == 'stldm':
            flops = 'FLOPs: skipped for stldm to avoid heavy JIT tracing in DDP.\n'
        else:
            try:
                flops_stats = FlopCountAnalysis(self.method.model.to(device), input_dummy)
                flops = flop_count_table(flops_stats)
            except Exception as exc:
                flops = f'FLOPs: unavailable ({exc}).\n'

        if args.fps:
            try:
                fps_val = measure_throughput(self.method.model.to(device), input_dummy)
                fps = 'Throughputs of {}: {:.3f}\n'.format(args.method, fps_val)
            except Exception as exc:
                fps = f'Throughputs: unavailable ({exc}).\n'
        else:
            fps = ''

        return info, flops, fps, dash_line
