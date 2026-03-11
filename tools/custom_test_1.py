import os
import os.path as osp
import torch
import numpy as np
import gc
from tqdm import tqdm

from openstl.utils import create_parser, default_parser, load_config, update_config
from openstl.datasets import load_data


def pixel_to_rainfall_tensor(pixel_data):
    """【纯 GPU 加速的物理转换函数 - CIKM 专属】"""
    p = pixel_data * 255.0
    dbz = p * (95.0 / 255.0) - 10.0
    a, b = 58.53, 1.56
    z = 10.0 ** (dbz / 10.0)
    r = (z / a) ** (1.0 / b)
    return torch.clamp(r, min=0.0)


def calculate_pcc(preds, targets):
    """在 GPU 上计算皮尔逊相关系数"""
    preds_mean = preds - preds.mean(dim=[-1, -2], keepdim=True)
    targets_mean = targets - targets.mean(dim=[-1, -2], keepdim=True)

    covariance = (preds_mean * targets_mean).sum(dim=[-1, -2])
    preds_std = torch.sqrt((preds_mean ** 2).sum(dim=[-1, -2]))
    targets_std = torch.sqrt((targets_mean ** 2).sum(dim=[-1, -2]))

    pcc = covariance / (preds_std * targets_std + 1e-8)
    return pcc.mean().item()


def main():
    # 极限释放显存
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # ==========================================
    # 1. Config 解析逻辑 (与 train.py 保持一致)
    # ==========================================
    args = create_parser().parse_args()
    config = args.__dict__

    cfg_path = osp.join('./configs', args.dataname, f'{args.method}.py') \
        if args.config_file is None else args.config_file
    print("加载配置文件：", cfg_path)

    if args.overwrite:
        config = update_config(config, load_config(cfg_path), exclude_keys=['method'])
    else:
        try:
            loaded_cfg = load_config(cfg_path)
            config = update_config(config, loaded_cfg,
                                   exclude_keys=['method', 'val_batch_size', 'drop_path', 'warmup_epoch'])
        except Exception as e:
            print(f"Warning: fail to load the config! ({e})")

        default_values = default_parser()
        for attribute in default_values.keys():
            if attribute in config and config[attribute] is None:
                config[attribute] = default_values[attribute]

    for k, v in config.items():
        setattr(args, k, v)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 开始多指标稳健测试流程 | 数据集: {args.dataname} | 模型: {args.method}")

    # ==========================================
    # 2. Dataset 加载与双数据集智能适配
    # ==========================================
    print("⏳ 正在构建 DataLoader...")
    train_loader, val_loader, test_loader = load_data(**config)

    if 'sevir' in args.dataname.lower():
        eval_mode = 'SEVIR'
        thresholds = [16, 74, 133, 160, 211, 277]
        unit = 'VIL Pixel'
        print("🎯 检测到 SEVIR 数据集，使用像素级 VIL 评估标准 (无需Z-R转换)。")
    elif 'cikm' in args.dataname.lower():
        eval_mode = 'CIKM'
        thresholds = [0.5, 2, 5, 10, 30]
        unit = 'mm/h'
        print("🎯 检测到 CIKM 数据集，使用降水率 (mm/h) 物理转换评估标准。")
    else:
        eval_mode = 'GENERAL'
        thresholds = [50, 100, 150, 200]
        unit = 'Pixel'
        print("🎯 未知数据集，使用通用像素评估标准。")

    # ==========================================
    # 3. 直接且暴力的模型实例化 (最稳妥，防 OpenSTL 坑)
    # ==========================================
    print("⏳ 正在构建模型...")
    if args.method == 'alphapre':
        from openstl.models.alphapre_model import get_model
        model = get_model(
            img_channels=config.get('in_shape', [13, 1, 128, 128])[1],
            dim=config.get('hidden_dim', 64),
            T_in=config.get('pre_seq_length', 13),
            T_out=config.get('aft_seq_length', 12),
            input_shape=(config.get('img_size', 128), config.get('img_size', 128)),
            n_layers=config.get('n_layers', 3),
            spec_num=config.get('spec_num', 20),
        ).to(device)
    elif args.method == 'exprecast':
        from openstl.models.exprecast_model import exPreCast_Model
        model = exPreCast_Model(**config).to(device)
    else:
        # 其他模型如果需要测，退回到 OpenSTL 的 method_maps
        from openstl.methods import method_maps
        if 'metrics' not in config or config['metrics'] is None:
            config['metrics'] = ['mse', 'mae']
        method = method_maps[args.method](**config)
        model = method.model.to(device)

    # 加载权重
    if hasattr(args, 'ckpt_path') and args.ckpt_path:
        print(f"⏳ 正在加载权重: {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint.get('network', checkpoint.get('model', checkpoint)))

        model_state_dict = model.state_dict()
        clean_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('model.', '').replace('network.', '').replace('module.', '')
            if k in model_state_dict:
                if v.shape != model_state_dict[k].shape:
                    print(f"⚠️ 警告: 忽略 '{k}' | 预训练: {v.shape} <--> 代码: {model_state_dict[k].shape}")
                    continue
            clean_state_dict[k] = v

        model.load_state_dict(clean_state_dict, strict=False)
        print("✅ 权重加载成功！")

    model.eval()

    # ==========================================
    # 4. 纯手工流式验证 (绝对防爆显存)
    # ==========================================
    print(f"🧠 开始前向推理并实时计算多维度指标 (模式: {eval_mode})...")

    TPs = {th: 0.0 for th in thresholds}
    FNs = {th: 0.0 for th in thresholds}
    FPs = {th: 0.0 for th in thresholds}
    TNs = {th: 0.0 for th in thresholds}

    total_mse = 0.0
    total_mae = 0.0
    total_pcc = 0.0
    total_samples = 0
    total_pixels = 0

    with torch.inference_mode():
        for batch_idx, (batch_x, batch_y) in enumerate(tqdm(test_loader, desc="Testing")):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # 手工切分输入输出帧
            total_frames = torch.cat([batch_x, batch_y], dim=1)
            t_in = args.pre_seq_length
            batch_x = total_frames[:, :t_in]
            batch_y = total_frames[:, t_in:]
            del total_frames

            # 纯粹的模型推理
            if args.method == 'alphapre':
                preds, _ = model.predict(frames_in=batch_x, compute_loss=False)
            else:
                preds = model(batch_x)

            preds = preds.float()

            # ✨✨✨ 终极核心补丁 1：防广播变异 BUG ✨✨✨
            # AlphaPre 输出可能是 [Batch, Time, Height, Width] 4D张量
            # 真实标签是 [Batch, Time, Channel, Height, Width] 5D张量
            # 补上 Channel 维度，防止 PyTorch 相减时把时间维强行扩散变成 20x20！
            if preds.ndim == 4 and batch_y.ndim == 5:
                preds = preds.unsqueeze(2)

            if batch_idx == 0:
                print(f"\n[Debug] 实际参与评测的 预测形状: {preds.shape}, 真实标签形状: {batch_y.shape}")

            # ✨✨✨ 终极核心补丁 2：防尺度爆炸 BUG ✨✨✨
            if preds.max() > 2.0:
                # 如果模型没有用 Sigmoid，直接吐出了 0~255 的数据
                preds_255 = torch.clamp(preds, 0.0, 255.0)
            else:
                # 正常情况：模型吐出 0~1 的数据
                preds_255 = torch.clamp(preds, 0.0, 1.0) * 255.0

            targets_255 = batch_y * 255.0

            # 1. 计算通用图像指标 (统一在 0-255 计算)
            total_mse += torch.sum((preds_255 - targets_255) ** 2).item()
            total_mae += torch.sum(torch.abs(preds_255 - targets_255)).item()
            total_pcc += calculate_pcc(preds_255, targets_255) * preds.shape[0]

            total_pixels += preds_255.numel()
            total_samples += preds.shape[0]

            # 2. 特化数据集预处理
            if eval_mode == 'CIKM':
                # CIKM 黑边处理
                if preds.shape[-1] == 112 and preds.shape[-2] == 112:
                    preds = preds[..., 5:-6, 5:-6]
                    batch_y = batch_y[..., 5:-6, 5:-6]
                # 转降雨率 (mm/h)
                preds_eval = pixel_to_rainfall_tensor(preds_255 / 255.0)
                targets_eval = pixel_to_rainfall_tensor(batch_y)
            else:
                # SEVIR：直接使用 0~255 的 VIL 像素值
                preds_eval = preds_255
                targets_eval = targets_255

            # 3. 统计混淆矩阵
            for th in thresholds:
                b_preds = preds_eval >= th
                b_targets = targets_eval >= th

                TPs[th] += torch.sum(b_preds & b_targets).item()
                FNs[th] += torch.sum(~b_preds & b_targets).item()
                FPs[th] += torch.sum(b_preds & ~b_targets).item()
                TNs[th] += torch.sum(~b_preds & ~b_targets).item()

                del b_preds, b_targets

            del batch_x, batch_y, preds, preds_eval, targets_eval, preds_255, targets_255
            gc.collect()
            torch.cuda.empty_cache()

    # ==========================================
    # 5. 指标汇算与报表生成
    # ==========================================
    final_mse = total_mse / total_pixels
    final_mae = total_mae / total_pixels
    final_pcc = total_pcc / total_samples
    final_psnr = 10 * np.log10((255.0 ** 2) / (final_mse + 1e-8))

    global_metrics = {
        'MSE': final_mse, 'MAE': final_mae, 'PSNR': final_psnr, 'PCC': final_pcc
    }

    weather_metrics = {}
    for th in thresholds:
        TP = TPs[th]
        FN = FNs[th]
        FP = FPs[th]
        TN = TNs[th]

        pod = TP / (TP + FN + 1e-6)
        far = FP / (TP + FP + 1e-6)
        csi = TP / (TP + FN + FP + 1e-6)
        f1 = 2 * TP / (2 * TP + FP + FN + 1e-6)

        total = TP + FN + FP + TN
        exp_hits = (TP + FN) * (TP + FP) / (total + 1e-6)
        exp_cns = (TN + FN) * (TN + FP) / (total + 1e-6)
        hss = (TP + TN - exp_hits - exp_cns) / (total - exp_hits - exp_cns + 1e-6)

        weather_metrics[f'th_{th}'] = {
            'CSI': csi, 'POD': pod, 'FAR': far, 'F1': f1, 'HSS': hss
        }

    save_path = f"results_{args.method}_{args.dataname}_Universal.txt"
    with open(save_path, "w", encoding="utf-8") as file:
        file.write(f"====================================================\n")
        file.write(f"  🏆 模型 {args.method.upper()} 深度评测报告 (Top-Tier 格式) \n")
        file.write(f"====================================================\n")
        file.write(f"数据集: {args.dataname.upper()} | 评估标准: {eval_mode}\n\n")

        file.write(f"[1] 图像重建质量指标 (Image Quality Metrics)\n")
        file.write(f"----------------------------------------------------\n")
        file.write(f"  ▶ MSE   (均方误差)      : {global_metrics['MSE']:.4f}\n")
        file.write(f"  ▶ MAE   (平均绝对误差)  : {global_metrics['MAE']:.4f}\n")
        file.write(f"  ▶ PSNR  (峰值信噪比)    : {global_metrics['PSNR']:.4f} dB\n")
        file.write(f"  ▶ PCC   (皮尔逊相关性)  : {global_metrics['PCC']:.4f}\n\n")

        file.write(f"[2] 气象二元分类指标 (Meteorological Binary Metrics)\n")
        file.write(f"----------------------------------------------------\n")
        for th in thresholds:
            m = weather_metrics[f'th_{th}']
            file.write(f"★ 阈值 >= {th} {unit}:\n")
            file.write(f"  - CSI  (临界成功指数, 越大越好): {m['CSI']:.4f}\n")
            file.write(f"  - POD  (命中率, 越大越好)      : {m['POD']:.4f}\n")
            file.write(f"  - FAR  (误报率, 越小越好)      : {m['FAR']:.4f}\n")
            file.write(f"  - F1   (综合F1, 越大越好)      : {m['F1']:.4f}\n")
            file.write(f"  - HSS  (Heidke技巧分, 越大越好): {m['HSS']:.4f}\n")
            file.write(f"\n")

    print(f"\n✅ 学术评测完成！结果已保存至: {os.path.abspath(save_path)}")
    with open(save_path, "r", encoding="utf-8") as file:
        print(file.read())


if __name__ == '__main__':
    main()