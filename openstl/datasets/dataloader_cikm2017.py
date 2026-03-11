import os
import glob
import csv
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# # 引入 OpenSTL 官方提供的高级数据加载器 (支持多卡与预取)
# # 增加智能导入机制：解决直接运行测试时 relative import 报错的问题
# try:
#     from .utils import create_loader
# except ImportError:
#     # 如果是直接运行当前脚本进行本地测试，临时将项目根目录加入系统路径
#     import sys
#
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#     from openstl.datasets.utils import create_loader
#
#
# class CIKMDataset(Dataset):
#     def __init__(self, data_root, mode='train', pre_seq_length=5, aft_seq_length=10,data_name = 'cikm'):
#         """
#         OpenSTL 标准的 Dataset 写法
#         :param data_root: 数据集的根目录 (应包含 cikm_data 文件夹和 csv 文件)
#         :param mode: 'train', 'val' 或 'test'
#         :param pre_seq_length: 输入的历史帧数 (默认5)
#         :param aft_seq_length: 预测的未来帧数 (默认10)
#         """
#         super().__init__()
#         self.mode = mode
#         self.pre_seq_length = pre_seq_length
#         self.aft_seq_length = aft_seq_length
#         self.data_name = data_name
#
#         # --- 🚀 新增这2行：提供给 OpenSTL 逆归一化机制使用 ---
#         self.mean = 0.0
#         self.std = 1.0
#
#         # 定义存放 NPY 的具体路径
#         npy_dir = os.path.join(data_root, 'cikm_data')
#
#         # 直接在 data_root 根目录下寻找对应的 CSV 文件
#         csv_path = os.path.join(data_root, f'{mode}.csv')
#
#         if not os.path.exists(csv_path):
#             print(f"⚠️ 警告: 未找到 {mode}.csv！请检查 {data_root} 路径是否正确。")
#             self.file_list = []
#             return
#
#         self.file_list = []
#
#         # 解析 CSV 文件获取 npy 文件名
#         # 使用 utf-8-sig 防止 Windows Excel 保存的 CSV 带有隐藏的 BOM 字符报错
#         with open(csv_path, 'r', encoding='utf-8-sig') as f:
#             reader = csv.reader(f)
#             for row in reader:
#                 if not row:
#                     continue
#                 # 取出 CSV 里的第一列（例如 'train_1'）
#                 fname = row[0].strip()
#
#                 # 自动跳过表格可能存在的表头
#                 if fname.lower() in ['filename', 'file_path', 'name', 'id', '文件名']:
#                     continue
#
#                 # 核心逻辑：拿到 train_1 后，自动拼接成 train_1.npy
#                 if not fname.endswith('.npy'):
#                     fname += '.npy'
#
#                 # 拼接完整路径：D:/.../cikm_data/train_1.npy
#                 full_path = os.path.join(npy_dir, fname)
#                 self.file_list.append(full_path)
#
#         if len(self.file_list) == 0:
#             print(f"⚠️ 警告: 在 {csv_path} 中未读到任何文件记录！")
#
#     def __len__(self):
#         return len(self.file_list)
#
#     def __getitem__(self, idx):
#         # 1. 读取 npy 文件 -> 原始形状 expected: (4, 15, 101, 101) 即 (Channel, Time, Height, Width)
#         data = np.load(self.file_list[idx])
#
#         # 2. 转换维度顺序 (C, T, H, W) -> (T, C, H, W)
#         # OpenSTL 和大多数时空预测模型要求时间维度在前面，因此变成了 (15, 4, 101, 101)
#         if data.ndim == 4:
#             data = np.transpose(data, (1, 0, 2, 3))
#
#         # 3. 归一化 (雷达反射率范围通常约 0~255) -> 转换到 0~1 之间
#         data = data.astype(np.float32) / 255.0
#
#         # 4. 帧数切分
#         seq_x = data[:self.pre_seq_length, ...]  # 前5帧 -> (5, 4, 101, 101)
#         seq_y = data[self.pre_seq_length:self.pre_seq_length + self.aft_seq_length, ...]  # 后10帧 -> (10, 4, 101, 101)
#
#         # 5. 转换成 PyTorch Tensor
#         # 注意：这里已经是 4 通道了，不需要再用 unsqueeze(1) 增加单通道维度了
#         seq_x = torch.tensor(seq_x)  # -> (5, 4, 101, 101)
#         seq_y = torch.tensor(seq_y)  # -> (10, 4, 101, 101)
#
#         # 6. 【防崩溃核心操作】Padding 补边到 112x112 (可以被16整除)
#         # 112 - 101 = 11，左补5，右补6，上补5，下补6
#         pad_left, pad_right = 5, 6
#         pad_top, pad_bottom = 5, 6
#
#         # 使用 F.pad 进行边缘补 0 (constant)
#         # F.pad 从最后一个维度往前补（即补 W 和 H 维度），对前面的 T 和 C 不影响
#         seq_x = F.pad(seq_x, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
#         seq_y = F.pad(seq_y, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
#
#         # 最终输出的形状将是 (5, 4, 112, 112) 和 (10, 4, 112, 112)
#         return seq_x, seq_y
#
#
# # ==========================================
# # OpenSTL 框架规定的标准对外接口
# # ==========================================
# def load_data_cikm(batch_size, val_batch_size, data_root, num_workers=4,
#                    pre_seq_length=5, aft_seq_length=10,
#                    distributed=False, use_prefetcher=False, **kwargs):
#     """
#     暴露给 OpenSTL API 的数据加载函数
#     """
#     train_set = CIKMDataset(data_root, mode='train', pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)
#     val_set = CIKMDataset(data_root, mode='val', pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)
#     test_set = CIKMDataset(data_root, mode='test', pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)
#
#     # 如果没有提供 test 文件记录，就用 val 顶替 test 以防报错
#     if len(test_set) == 0 and len(val_set) > 0:
#         print("💡 提示: 未找到 test 数据，系统将自动使用 val 数据集作为测试评估。")
#         test_set = val_set
#
#     # 智能分流：
#     # 当 num_workers 为 0 时（通常为 Windows 本地调试），强行使用原生 DataLoader 避免 persistent_workers 报错
#     if num_workers == 0:
#         dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True,
#                                       pin_memory=True, drop_last=True, num_workers=0)
#         dataloader_val = DataLoader(val_set, batch_size=val_batch_size, shuffle=False,
#                                     pin_memory=True, drop_last=False, num_workers=0)
#         dataloader_test = DataLoader(test_set, batch_size=val_batch_size, shuffle=False,
#                                      pin_memory=True, drop_last=False, num_workers=0)
#     else:
#         # 构建 OpenSTL 专用的大型 DataLoader (支持多卡与预取加速)
#         dataloader_train = create_loader(train_set,
#                                          batch_size=batch_size,
#                                          shuffle=True, is_training=True,
#                                          pin_memory=True, drop_last=True,
#                                          num_workers=num_workers,
#                                          distributed=distributed,
#                                          use_prefetcher=use_prefetcher)
#
#         dataloader_val = create_loader(val_set,
#                                        batch_size=val_batch_size,
#                                        shuffle=False, is_training=False,
#                                        pin_memory=True, drop_last=False,
#                                        num_workers=num_workers,
#                                        distributed=distributed,
#                                        use_prefetcher=use_prefetcher)
#
#         dataloader_test = create_loader(test_set,
#                                         batch_size=val_batch_size,
#                                         shuffle=False, is_training=False,
#                                         pin_memory=True, drop_last=False,
#                                         num_workers=num_workers,
#                                         distributed=distributed,
#                                         use_prefetcher=use_prefetcher)
#
#     return dataloader_train, dataloader_val, dataloader_test
#
import os
import glob
import csv
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from .utils import create_loader
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from openstl.datasets.utils import create_loader


class CIKMDataset(Dataset):
    def __init__(self, data_root, mode='train', pre_seq_length=5, aft_seq_length=10, data_name='cikm'):
        super().__init__()
        self.mode = mode
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.data_name = data_name
        self.mean = 0.0
        self.std = 1.0

        # ✨ 修复点 1：智能目录映射
        # 因为你只有 train 和 test 文件夹，当框架请求 val 时，我们把它指向 test 文件夹
        target_folder = mode
        if mode == 'val' and not os.path.exists(os.path.join(data_root, 'cikm_data', 'val')):
            target_folder = 'test'

        # 拼接出真实的 npy 存放路径：cikm_data/train 或 cikm_data/test
        npy_dir = os.path.join(data_root, 'cikm_data', target_folder)
        csv_path = os.path.join(data_root, f'{mode}.csv')

        # 如果连 mode 对应的 csv 都没有，就用 test.csv 顶替 val.csv
        if not os.path.exists(csv_path) and mode == 'val':
            csv_path = os.path.join(data_root, 'test.csv')

        if not os.path.exists(csv_path):
            print(f"⚠️ 警告: 未找到 {csv_path}！")
            self.file_list = []
            return

        self.file_list = []

        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                fname = row[0].strip()

                if fname.lower() in ['filename', 'file_path', 'name', 'id', '文件名']:
                    continue

                if not fname.endswith('.npy'):
                    fname += '.npy'

                # 完美拼出: .../cikm_data/train/train_4260.npy
                full_path = os.path.join(npy_dir, fname)
                self.file_list.append(full_path)

        if len(self.file_list) == 0:
            print(f"⚠️ 警告: 在 {csv_path} 中未读到任何文件记录！")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])

        if data.ndim == 4:
            data = np.transpose(data, (1, 0, 2, 3))

        data = data.astype(np.float32) / 255.0

        seq_x = data[:self.pre_seq_length, ...]
        seq_y = data[self.pre_seq_length:self.pre_seq_length + self.aft_seq_length, ...]

        seq_x = torch.tensor(seq_x)
        seq_y = torch.tensor(seq_y)

        pad_left, pad_right = 5, 6
        pad_top, pad_bottom = 5, 6

        seq_x = F.pad(seq_x, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
        seq_y = F.pad(seq_y, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)

        return seq_x, seq_y


def load_data_cikm(batch_size, val_batch_size, data_root, num_workers=4,
                   pre_seq_length=5, aft_seq_length=10,
                   distributed=False, use_prefetcher=False, **kwargs):
    train_set = CIKMDataset(data_root, mode='train', pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)
    val_set = CIKMDataset(data_root, mode='val', pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)
    test_set = CIKMDataset(data_root, mode='test', pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)

    if len(test_set) == 0 and len(val_set) > 0:
        test_set = val_set

    if num_workers == 0:
        dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True,
                                      num_workers=0)
        dataloader_val = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, drop_last=False,
                                    num_workers=0)
        dataloader_test = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                     drop_last=False, num_workers=0)
    else:
        dataloader_train = create_loader(train_set, batch_size=batch_size, shuffle=True, is_training=True,
                                         pin_memory=True, drop_last=True, num_workers=num_workers,
                                         distributed=distributed, use_prefetcher=use_prefetcher)
        dataloader_val = create_loader(val_set, batch_size=val_batch_size, shuffle=False, is_training=False,
                                       pin_memory=True, drop_last=False, num_workers=num_workers,
                                       distributed=distributed, use_prefetcher=use_prefetcher)
        dataloader_test = create_loader(test_set, batch_size=val_batch_size, shuffle=False, is_training=False,
                                        pin_memory=True, drop_last=False, num_workers=num_workers,
                                        distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_val, dataloader_test
if __name__ == '__main__':
    # ==========================================
    # 本地测试模块
    # ==========================================
    # 请根据你电脑的实际情况确认此路径
    test_data_root = r"D:\dp_projects\datasets\cikm_dataset"

    print("=" * 50)
    print("🚀 开始测试 CIKM Dataloader (支持 4 通道与智能回退)")
    print("=" * 50)

    try:
        # 1. 实例化 Dataset 来单独查看前10个数据
        train_dataset = CIKMDataset(data_root=test_data_root, mode='train')
        print(f"✅ 成功加载训练集，共在 CSV 中找到 {len(train_dataset)} 个样本。\n")

        print("--- 🔍 打印前10个样本的详细信息 ---")
        for i in range(min(10, len(train_dataset))):
            seq_x, seq_y = train_dataset[i]
            print(f"样本 [{i + 1}/10]:")
            print(f"  ➡️ 输入 seq_x 形状: {seq_x.shape} | 最大值: {seq_x.max():.4f} | 最小值: {seq_x.min():.4f}")
            print(f"  ➡️ 输出 seq_y 形状: {seq_y.shape} | 最大值: {seq_y.max():.4f} | 最小值: {seq_y.min():.4f}")

            # 打印其中一个小切片 (例如第1帧, 第1通道的中心 3x3 区域)，112/2 = 56，取 55~58
            print(f"  👀 seq_x 第1帧第1通道 中心 3x3 像素值矩阵:\n{seq_x[0, 0, 55:58, 55:58].numpy()}")
            print("-" * 40)

        # 2. 测试 DataLoader 封装
        print("\n--- 📦 测试 DataLoader 批量加载 ---")
        train_loader, val_loader, test_loader = load_data_cikm(
            batch_size=4,
            val_batch_size=4,
            data_root=test_data_root,
            # 0 代表启动安全回退模式，完美解决 persistent_workers 报错
            num_workers=0,
            distributed=False,
            use_prefetcher=False
        )

        for batch_x, batch_y in train_loader:
            print(f"✅ 成功取出一个 Batch!")
            # 预期形状应该是 [BatchSize, Time, Channel, Height, Width] -> [4, 5, 4, 112, 112]
            print(f"📦 Batch X 形状 (模型输入): {batch_x.shape}")
            print(f"📦 Batch Y 形状 (真实标签): {batch_y.shape}")
            break  # 测完第一个 batch 就退出

    except Exception as e:
        print(f"❌ 测试失败，报错信息:\n{e}")
        print("💡 提示: 请检查 test_data_root 路径是否正确，以及路径下是否有 train.csv 和 cikm_data 文件夹。")