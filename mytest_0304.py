import os
import glob
import numpy as np
from tqdm import tqdm


def check_folder(folder_path):
    """
    检查指定文件夹下所有 .npz 文件的键名格式
    """
    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return

    files = glob.glob(os.path.join(folder_path, "*.npz"))
    if len(files) == 0:
        print(f"⚠️ 文件夹中没有找到 .npz 文件: {folder_path}")
        return

    print(f"\n🔍 开始检查文件夹: {folder_path} (共 {len(files)} 个文件)")

    data_count = 0
    in_out_count = 0
    other_count = 0
    error_count = 0

    # 使用 tqdm 显示进度条
    for fpath in tqdm(files, desc="Checking files"):
        try:
            # 使用 np.load 读取文件，默认只加载索引信息，速度极快
            with np.load(fpath) as data:
                keys = data.files

                # 判断键名逻辑
                if 'IN' in keys and 'OUT' in keys:
                    in_out_count += 1
                elif 'data' in keys:
                    data_count += 1
                else:
                    other_count += 1
        except Exception as e:
            error_count += 1

    # 打印最终统计结果
    print(f"\n📊 【{os.path.basename(folder_path)}】 检查结果汇总:")
    print(f"   ✅ 标准格式 (包含 IN 和 OUT): {in_out_count} 个")
    print(f"   ⚠️ 旧版格式 (仅包含 data)   : {data_count} 个   <-- (如果是这个格式，count++)")

    if other_count > 0:
        print(f"   ❓ 其他未知格式数量        : {other_count} 个")
    if error_count > 0:
        print(f"   ❌ 损坏无法读取的文件数    : {error_count} 个")


if __name__ == '__main__':
    # 您指定的两个目标路径
    train_dir = r"D:\dp_projects\datasets\SEVIR\sevir_npz\train"
    test_dir = r"D:\dp_projects\datasets\SEVIR\sevir_npz\test"

    print("==================================================")
    check_folder(train_dir)
    print("==================================================")
    check_folder(test_dir)
    print("==================================================")