import random
import pandas as pd
if __name__ == '__main__':

    # ==========================================
    # 1. 生成 train.csv (1 到 8000，且内部打乱)
    # ==========================================
    # 生成 1 到 8000 的数字列表
    train_indices = list(range(1, 8001))

    # 打乱这 8000 个数字的顺序（对训练集非常重要）
    random.shuffle(train_indices)

    # 拼凑成你需要的格式，例如 "train_1"
    # 如果你的代码读取时需要带后缀，可以改成 f"train_{i}.npy"
    train_filenames = [f"train_{i}" for i in train_indices]

    # 保存为 train.csv
    df_train = pd.DataFrame({'filename': train_filenames})
    df_train.to_csv('train.csv', index=False)
    print("✅ train.csv 生成成功！共", len(df_train), "条。")


    # ==========================================
    # 2. 生成 test.csv (8001 到 10000，不打乱)
    # ==========================================
    # 生成 8001 到 10000 的数字列表
    test_indices = list(range(8001, 10001))

    # 测试集严格保持时间顺序，拼凑格式
    test_filenames = [f"train_{i}" for i in test_indices]

    # 保存为 test.csv
    df_test = pd.DataFrame({'filename': test_filenames})
    df_test.to_csv('test.csv', index=False)
    print("✅ test.csv 生成成功！共", len(df_test), "条。")