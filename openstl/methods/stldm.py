import torch
import torch.nn.functional as F
from .base_method import Base_method
from openstl.models import STLDM_Model


class STLDM(Base_method):
    r"""STLDM 专属 Method
    由于扩散模型的训练逻辑（加噪与去噪）有别于传统模型，
    在此重写 training_step 以调用官方的四大联合 Loss。
    """

    def __init__(self, **args):
        super().__init__(**args)

    def _build_model(self, **kwargs):
        return STLDM_Model(**kwargs)

    def training_step(self, batch, batch_idx):
        """完全按照官方 train.py 的逻辑重写训练步骤"""
        batch_x, batch_y = batch

        # 调用包装在 STLDM_Model 中的官方计算 Loss 的函数
        recon_loss, mu_loss, diff_loss, prior_loss = self.model.compute_loss(batch_x, batch_y)

        # 官方的 Total Loss 计算方式 (train.py 第 153 行)
        loss = recon_loss + mu_loss + diff_loss + prior_loss

        # 记录每种 Loss 以方便在 TensorBoard 中监控扩散是否正常
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True)
        self.log('train_mu_loss', mu_loss, on_step=True, on_epoch=True)
        self.log('train_diff_loss', diff_loss, on_step=True, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def forward(self, batch_x, batch_y=None, **kwargs):
        """重写前向传播：调用扩散模型的采样生成过程"""
        pred_y = self.model(batch_x)

        # 同样保留终极防御墙（防止帧数越界导致 Validation 崩溃）
        target_len = self.hparams.aft_seq_length
        if pred_y.shape[1] > target_len:
            pred_y = pred_y[:, :target_len, ...].contiguous()
        elif pred_y.shape[1] < target_len:
            pad_len = target_len - pred_y.shape[1]
            pred_y = F.pad(pred_y, (0, 0, 0, 0, 0, 0, 0, pad_len))

        return pred_y