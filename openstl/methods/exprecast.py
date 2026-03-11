import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from openstl.models import exPreCast_Model
from .base_method import Base_method


# ====================================================================
# ✨ 原作者专属：频域感知损失函数 (Frequency Aware Charbonnier Loss)
# ====================================================================
class FACL(nn.Module):
    def __init__(self, total_step, const_ratio=0.4, prob_init=1.0, prob_end=0.0, include_sigmoid=False):
        super(FACL, self).__init__()
        const_step = int(total_step * const_ratio)
        self.prob_init = prob_init
        self.prob_end = prob_end

        # 🚨 极度重要的修复：在 PyTorch Lightning 中，内部张量必须注册为 buffer
        # 否则模型在 GPU 上训练时，这个张量会卡在 CPU 上导致崩溃！
        prob_thres = torch.linspace(prob_init, prob_end, int(total_step - const_step))
        self.register_buffer('prob_thres', prob_thres)

        self.step_cnt = 0  # 内部计步器
        self.include_sigmoid = include_sigmoid

    def get_thres(self):
        # 动态获取当前步数的概率阈值
        if self.step_cnt < len(self.prob_thres):
            prob = self.prob_thres[self.step_cnt]
        else:
            prob = self.prob_thres[-1]

        self.step_cnt += 1
        return 1.0 - prob

    def fal(self, fft_pred, fft_gt):
        return nn.MSELoss()(fft_pred.abs(), fft_gt.abs())

    def fcl(self, fft_pred, fft_gt):
        conj_pred = torch.conj(fft_pred)
        numerator = (conj_pred * fft_gt).sum().real
        denominator = torch.sqrt(((fft_gt).abs() ** 2).sum() * ((fft_pred).abs() ** 2).sum())
        return 1.0 - numerator / denominator

    def forward(self, pred, gt):
        if self.include_sigmoid:
            pred = torch.sigmoid(pred)  # 修复: F.sigmoid 已被 PyTorch 弃用
            gt = torch.sigmoid(gt)

        # 进行 2D 傅里叶变换转换到频域
        fft_pred = torch.fft.fftn(pred, dim=[-1, -2], norm='ortho')
        fft_gt = torch.fft.fftn(gt, dim=[-1, -2], norm='ortho')
        prob = self.get_thres()

        H, W = pred.shape[-2:]
        weight = np.sqrt(H * W)

        # 组合频域振幅损失和相位损失
        loss = prob * self.fal(fft_pred, fft_gt) + (1.0 - prob) * self.fcl(fft_pred, fft_gt)
        loss = loss * weight
        return loss


# ====================================================================
# ✨ 核心 Method 训练流
# ====================================================================
class exPreCast(Base_method):
    r"""exPreCast Method

    直接继承 Base_method，保持最纯净的训练流。
    移除了对 SimVP 的依赖，防止隐式的自回归循环干扰模型本身的 time_extractor。
    """

    def __init__(self, **args):
        super().__init__(**args)

        # 估算总训练步数 (原作者默认是 100000 步)
        # 取 Epoch * 每轮的 Batch 数量，如果没有，则安全降级为 100000
        total_steps = self.hparams.get('epochs', 100) * self.hparams.get('steps_per_epoch', 1000)
        total_steps = max(total_steps, 100000)

        # ✨ 替换默认的 MSELoss，注入原作者强大的频域损失函数！
        self.criterion = FACL(total_step=total_steps)

    def _build_model(self, **args):
        """
        构建并返回我们的原生模型。
        """
        return exPreCast_Model(**args)

    def forward(self, batch_x, batch_y=None, **kwargs):
        """
        前向传播逻辑
        """
        # 1. 我们的模型已经通过 time_extractor 完美支持了 T_in 到 T_out 的映射
        pred_y = self.model(batch_x)

        # 2. 获取预期的输出帧数
        aft_seq_length = self.hparams.aft_seq_length

        # 3. 安全防线：确保输出帧数严格等于配置文件要求的帧数
        if pred_y.shape[1] > aft_seq_length:
            pred_y = pred_y[:, :aft_seq_length, ...]

        return pred_y

    def training_step(self, batch, batch_idx):
        """
        定义单步训练的逻辑
        """
        batch_x, batch_y = batch

        # 获取预测值
        pred_y = self(batch_x)

        # 计算 FACL 频域 Loss
        loss = self.criterion(pred_y, batch_y)

        # 记录日志 (prog_bar=True 会让它显示在你的进度条上)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss