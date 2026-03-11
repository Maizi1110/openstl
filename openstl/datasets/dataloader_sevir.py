"""Input generator for SEVIR (.npz files) without CATALOG dependency."""

import glob
import os
import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


CURRENT_DIR = osp.dirname(osp.abspath(__file__))
PROJECT_ROOT = osp.dirname(osp.dirname(CURRENT_DIR))
DEFAULT_SEVIR_ROOT = osp.abspath(osp.join(PROJECT_ROOT, '../../datasets/SEVIR'))

DTYPES = {
    'vil': np.uint8,
    'vis': np.int16,
    'ir069': np.int16,
    'ir107': np.int16,
    'lght': np.int16,
}


class SEVIRDataset(Dataset):
    """Sequence class for generating batches from SEVIR npz files."""

    def __init__(
        self,
        data_root=DEFAULT_SEVIR_ROOT,
        split='train',
        data_name='vil',
        use_augment=False,
        n_batch_per_epoch=None,
        unwrap_time=False,
        shuffle=False,
        shuffle_seed=1,
        output_type=np.float32,
        normalize_x=None,
        normalize_y=None,
    ):
        # Keep backward-compatible path auto-correction.
        if data_root == './data' or not osp.exists(osp.join(data_root, 'sevir_npz')):
            self.data_root = DEFAULT_SEVIR_ROOT
        else:
            self.data_root = data_root

        self.split = split
        self.data_name = data_name
        self.use_augment = use_augment
        self.n_batch_per_epoch = n_batch_per_epoch
        self.output_type = output_type
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y

        self.mean = 33.44
        self.std = 47.54

        self.npz_dir = osp.join(self.data_root, 'sevir_npz', self.split)

        if not osp.exists(self.npz_dir):
            self.file_list = []
            print(f'error: data folder not found -> {self.npz_dir}')
        else:
            self.file_list = sorted(glob.glob(osp.join(self.npz_dir, '*.npz')))
            print(f'info: SEVIR [{self.split}] loaded from {self.npz_dir} ({len(self.file_list)} samples).')

        if len(self.file_list) == 0:
            print(f'warning: no .npz samples found under {self.npz_dir}.')

    def __len__(self):
        max_n = len(self.file_list)
        if self.n_batch_per_epoch is not None:
            return min(self.n_batch_per_epoch, max_n)
        return max_n

    def _augment_seq(self, x, y):
        return x, y

    def _to_tensor(self, x, y):
        return torch.from_numpy(x), torch.from_numpy(y)

    def __getitem__(self, idx):
        with np.load(self.file_list[idx]) as data:
            x = data['IN'].astype(self.output_type)
            y = data['OUT'].astype(self.output_type)

        if self.use_augment:
            x, y = self._augment_seq(x, y)
        else:
            x, y = self._to_tensor(x, y)

        if self.normalize_x:
            x = SEVIRDataset.normalize(x, self.normalize_x)
        else:
            x = (x - self.mean) / self.std

        if self.normalize_y:
            y = SEVIRDataset.normalize(y, self.normalize_y)
        else:
            y = (y - self.mean) / self.std

        return x, y

    @staticmethod
    def normalize(x, s):
        """Normalize X by tuple s=(scale, offset): Z=(X-offset)*scale."""
        return (x - s[1]) * s[0]

    @staticmethod
    def unnormalize(z, s):
        """Inverse transform of normalize()."""
        return z / s[0] + s[1]


def load_data(
    batch_size,
    val_batch_size,
    data_root=DEFAULT_SEVIR_ROOT,
    num_workers=4,
    data_name='vil',
    in_shape=[13, 1, 384, 384],
    distributed=False,
    use_augment=False,
    use_prefetcher=False,
    drop_last=False,
    **kwargs,
):
    train_set = SEVIRDataset(data_root=data_root, split='train', data_name=data_name, use_augment=use_augment)
    test_set = SEVIRDataset(data_root=data_root, split='test', data_name=data_name, use_augment=False)

    dataloader_train = create_loader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        is_training=True,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
    )

    dataloader_vali = create_loader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        is_training=False,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
    )

    dataloader_test = create_loader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        is_training=False,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
    )

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    print('=' * 60)
    print(f'SEVIR root: {DEFAULT_SEVIR_ROOT}')
    print('=' * 60)

    test_folder = osp.join(DEFAULT_SEVIR_ROOT, 'sevir_npz', 'test')
    if osp.exists(test_folder):
        print(f'found test folder: {test_folder}')
        dataloader_train, _, _ = load_data(batch_size=2, val_batch_size=2, num_workers=0)
        for step, (x, y) in enumerate(dataloader_train):
            print(f'batch {step}: X={x.shape}, Y={y.shape}')
            break
    else:
        print(f'warning: test folder not found: {test_folder}')