import torch
from .base_method import Base_method


class alphapre(Base_method):
    r"""AlphaPre Method

    Implementation of AlphaPre: Amplitude-Phase Disentanglement Model
    """

    def __init__(self, **args):
        super().__init__(**args)

    def _build_model(self, **args):
        # 1. 放弃复杂的 Wrapper，直接调用 AlphaPre 源码原生的 get_model 函数！
        # 假设你已经将原作者的 alphapre.py 放到了 openstl/models/ 目录下
        try:
            from openstl.models.alphapre_model import get_model
        except ImportError:
            raise ImportError("请确保 AlphaPre 源码 alphapre.py 已被放置在 openstl/models/ 目录下")

        _, img_channel, img_height, img_width = self.hparams.in_shape

        # ✨ 核心改动 1：从 hparams 中读取我们在 parser.py 中新加的 img_size
        # 如果没传 img_size，默认退化使用 in_shape 的宽度
        img_size = self.hparams.get('img_size', img_width)

        # 2. 将 OpenSTL 的超参数完美映射给 AlphaPre
        model = get_model(
            img_channels=img_channel,
            dim=self.hparams.get('hidden_dim', 64),
            T_in=self.hparams.pre_seq_length,
            T_out=self.hparams.aft_seq_length,

            # input_shape=(img_height, img_width),
            input_shape=(img_size, img_size),

            n_layers=self.hparams.get('n_layers', 3),
            spec_num=self.hparams.get('spec_num', 20),
            pha_weight=self.hparams.get('pha_weight', 0.01),
            anet_weight=self.hparams.get('anet_weight', 0.1),
            amp_weight=self.hparams.get('amp_weight', 0.01),
            aweight_stop_steps=self.hparams.get('aw_stop_step', 10000)
        )
        return model

    def forward(self, batch_x, batch_y=None, **kwargs):
        """
        验证(Val)和测试(Test)阶段会调用这里。
        我们只需调用预测，不需要计算频域 Loss。
        """
        pred_y, _ = self.model.predict(frames_in=batch_x, frames_gt=batch_y, compute_loss=False)
        return pred_y

    def training_step(self, batch, batch_idx):
        """
        训练阶段的核心逻辑。参考 PredRNN 的写法。
        """
        batch_x, batch_y = batch

        # 核心精髓：直接让模型计算原汁原味的频域解耦 Loss！
        pred_y, loss_dict = self.model.predict(frames_in=batch_x, frames_gt=batch_y, compute_loss=True)

        total_loss = loss_dict['total_loss']

        # 将所有的独立 Loss 记录到 PyTorch Lightning，方便你在 TensorBoard 查看
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('phase_loss', loss_dict['phase_loss'], on_step=True, on_epoch=True, prog_bar=False)
        self.log('ampli_loss', loss_dict['ampli_loss'], on_step=True, on_epoch=True, prog_bar=False)
        self.log('anet_loss', loss_dict['anet_loss'], on_step=True, on_epoch=True, prog_bar=False)

        return total_loss