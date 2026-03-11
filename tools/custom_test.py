import gc
import os
import torch
import numpy as np
from tqdm import tqdm

from openstl.utils import get_dataset
from openstl.utils import (
    create_parser,
    default_parser,
    load_config,
    update_config,
    canonicalize_dataname,
    build_default_config_path,
    get_cli_override_keys,
)


def pixel_to_rainfall_tensor(pixel_data):
    p = pixel_data * 255.0
    dbz = p * (95.0 / 255.0) - 10.0
    a, b = 58.53, 1.56
    z = 10.0 ** (dbz / 10.0)
    r = (z / a) ** (1.0 / b)
    return torch.clamp(r, min=0.0)


def calculate_pcc(preds, targets):
    preds_mean = preds - preds.mean(dim=[-1, -2], keepdim=True)
    targets_mean = targets - targets.mean(dim=[-1, -2], keepdim=True)

    covariance = (preds_mean * targets_mean).sum(dim=[-1, -2])
    preds_std = torch.sqrt((preds_mean ** 2).sum(dim=[-1, -2]))
    targets_std = torch.sqrt((targets_mean ** 2).sum(dim=[-1, -2]))

    pcc = covariance / (preds_std * targets_std + 1e-8)
    return pcc.mean().item()


def parse_and_merge_config():
    parser = create_parser()
    args = parser.parse_args()
    cli_override_keys = get_cli_override_keys(parser)

    raw_dataname = args.dataname
    args.dataname = canonicalize_dataname(args.dataname)
    if raw_dataname != args.dataname:
        print(f'info: remap dataname alias {raw_dataname} -> {args.dataname}')

    config = args.__dict__
    cfg_path = args.config_file if args.config_file else build_default_config_path(args.dataname, args.method)
    args.config_file = cfg_path
    print(f'loading config: {cfg_path}')

    loaded_cfg = load_config(cfg_path)
    if args.overwrite:
        config = update_config(config, loaded_cfg, cli_override_keys=set(config.keys()))
    else:
        config = update_config(config, loaded_cfg, cli_override_keys=cli_override_keys)

    default_values = default_parser()
    for attribute, value in default_values.items():
        if attribute in config and config[attribute] is None:
            config[attribute] = value

    config['dataname'] = args.dataname

    for k, v in config.items():
        setattr(args, k, v)

    return args, config


def build_model(args, config, device):
    is_official_exprecast = False

    if args.method == 'exprecast':
        from openstl.models.exprecast import exPreCast_Model

        if getattr(args, 'ckpt_path', None) and 'pretrained_sevir' in args.ckpt_path.lower():
            is_official_exprecast = True
            config['embed_dim'] = 96
            config['depths'] = [2, 2, 6, 2]
            config['num_heads'] = [3, 6, 12, 24]
            config['window_size'] = (2, 7, 7)
            config['skip_connection'] = 'add'
            config['patch_norm'] = False

        model = exPreCast_Model(**config).to(device)

    elif args.method == 'alphapre':
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
    else:
        from openstl.methods import method_maps

        if 'metrics' not in config or config['metrics'] is None:
            config['metrics'] = ['mse', 'mae']
        method = method_maps[args.method](**config)
        model = method.model.to(device)

    return model, is_official_exprecast


def load_checkpoint(args, model):
    if not getattr(args, 'ckpt_path', None):
        return

    print(f'loading checkpoint: {args.ckpt_path}')
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint.get('network', checkpoint.get('model', checkpoint)))

    model_state_dict = model.state_dict()
    clean_state_dict = {}
    mismatched = []
    unexpected = []

    for k, v in state_dict.items():
        k = k.replace('model.', '').replace('network.', '').replace('module.', '')
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                clean_state_dict[k] = v
            else:
                mismatched.append((k, tuple(v.shape), tuple(model_state_dict[k].shape)))
        else:
            unexpected.append(k)

    missing_keys = [k for k in model_state_dict.keys() if k not in clean_state_dict]

    if args.method == 'stldm':
        if mismatched or missing_keys or unexpected:
            msg = [
                'Strict checkpoint loading failed for STLDM.',
                f'Mismatched: {len(mismatched)}',
                f'Missing: {len(missing_keys)}',
                f'Unexpected: {len(unexpected)}',
            ]
            if mismatched:
                k, got, need = mismatched[0]
                msg.append(f'Example mismatched key: {k}, ckpt={got}, model={need}')
            if missing_keys:
                msg.append(f'Example missing key: {missing_keys[0]}')
            if unexpected:
                msg.append(f'Example unexpected key: {unexpected[0]}')
            raise RuntimeError(' | '.join(msg))

        model.load_state_dict(clean_state_dict, strict=True)
        print(f'STLDM strict checkpoint load success: {len(clean_state_dict)}/{len(model_state_dict)} keys.')
    else:
        for k, got_shape, need_shape in mismatched:
            print(f'Shape mismatch skipped: {k} (need {need_shape}, ckpt {got_shape})')

        model.load_state_dict(clean_state_dict, strict=False)

        critical_missing = [k for k in missing_keys if 'patch_embed.norm' not in k]
        print(f'Weights loaded: {len(clean_state_dict)}/{len(model_state_dict)}')
        if critical_missing:
            print(f'Warning: {len(critical_missing)} critical keys missing, first few: {critical_missing[:5]}')


def main():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args, config = parse_and_merge_config()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'start robust evaluation | dataset: {args.dataname} | method: {args.method}')

    print('building dataloader...')
    _, _, test_loader = get_dataset(args.dataname, config)

    if 'sevir' in args.dataname.lower():
        eval_mode = 'SEVIR'
        thresholds = [16, 74, 133, 160, 211, 277]
        unit = 'VIL Pixel'
        print('detected SEVIR evaluation mode')
    else:
        eval_mode = 'CIKM'
        thresholds = [0.5, 2, 5, 10, 30]
        unit = 'mm/h'
        print('detected CIKM evaluation mode')

    print('building model...')
    model, is_official_exprecast = build_model(args, config, device)
    load_checkpoint(args, model)
    model.eval()

    print(f'start inference and metrics ({eval_mode})...')

    tps = {th: 0.0 for th in thresholds}
    fns = {th: 0.0 for th in thresholds}
    fps = {th: 0.0 for th in thresholds}
    tns = {th: 0.0 for th in thresholds}

    total_mse = 0.0
    total_mae = 0.0
    total_pcc = 0.0
    total_samples = 0
    total_pixels = 0

    mean_ex, scale_ex = 33.44, 47.54

    with torch.inference_mode():
        for batch_x, batch_y in tqdm(test_loader, desc='Testing'):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            total_frames = torch.cat([batch_x, batch_y], dim=1)
            t_in = args.pre_seq_length
            batch_x = total_frames[:, :t_in]
            batch_y = total_frames[:, t_in:]

            raw_x = batch_x * 255.0 if batch_x.max() <= 2.0 else batch_x
            raw_y = batch_y * 255.0 if batch_y.max() <= 2.0 else batch_y
            targets_255 = torch.clamp(raw_y, 0.0, 255.0)

            if is_official_exprecast:
                batch_x_input = (raw_x - mean_ex) / scale_ex
                preds = model(batch_x_input)
                raw_pred = preds * scale_ex + mean_ex
                preds_255 = torch.clamp(raw_pred, 0.0, 255.0)
            elif args.method == 'alphapre':
                batch_x_input = raw_x / 255.0
                preds, _ = model.predict(frames_in=batch_x_input, compute_loss=False)
                preds_255 = torch.clamp(preds, 0.0, 1.0) * 255.0
            else:
                preds = model(raw_x / 255.0)
                preds_255 = torch.clamp(preds, 0.0, 1.0) * 255.0

            if preds_255.ndim == 4 and targets_255.ndim == 5:
                preds_255 = preds_255.unsqueeze(2)
            elif preds_255.shape == (
                targets_255.shape[0],
                targets_255.shape[2],
                targets_255.shape[1],
                targets_255.shape[3],
                targets_255.shape[4],
            ):
                preds_255 = preds_255.permute(0, 2, 1, 3, 4)

            if preds_255.shape != targets_255.shape:
                raise RuntimeError(f'Prediction shape mismatch: pred={preds_255.shape}, target={targets_255.shape}')

            total_mse += torch.sum((preds_255 - targets_255) ** 2).item()
            total_mae += torch.sum(torch.abs(preds_255 - targets_255)).item()
            total_pcc += calculate_pcc(preds_255, targets_255) * preds_255.shape[0]

            total_pixels += preds_255.numel()
            total_samples += preds_255.shape[0]

            if eval_mode == 'CIKM':
                if preds_255.shape[-1] == 112 and preds_255.shape[-2] == 112:
                    preds_255 = preds_255[..., 5:-6, 5:-6]
                    targets_255 = targets_255[..., 5:-6, 5:-6]
                preds_eval = pixel_to_rainfall_tensor(preds_255 / 255.0)
                targets_eval = pixel_to_rainfall_tensor(targets_255 / 255.0)
            else:
                preds_eval = preds_255
                targets_eval = targets_255

            for th in thresholds:
                b_preds = preds_eval >= th
                b_targets = targets_eval >= th
                tps[th] += torch.sum(b_preds & b_targets).item()
                fns[th] += torch.sum(~b_preds & b_targets).item()
                fps[th] += torch.sum(b_preds & ~b_targets).item()
                tns[th] += torch.sum(~b_preds & ~b_targets).item()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_mse = total_mse / total_pixels
    final_mae = total_mae / total_pixels
    final_pcc = total_pcc / total_samples
    final_psnr = 10 * np.log10((255.0 ** 2) / (final_mse + 1e-8))

    weather_metrics = {}
    for th in thresholds:
        tp, fn, fp, tn = tps[th], fns[th], fps[th], tns[th]
        pod = tp / (tp + fn + 1e-6)
        far = fp / (tp + fp + 1e-6)
        csi = tp / (tp + fn + fp + 1e-6)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)

        total = tp + fn + fp + tn
        exp_hits = (tp + fn) * (tp + fp) / (total + 1e-6)
        exp_cns = (tn + fn) * (tn + fp) / (total + 1e-6)
        hss = (tp + tn - exp_hits - exp_cns) / (total - exp_hits - exp_cns + 1e-6)

        weather_metrics[th] = {'CSI': csi, 'POD': pod, 'FAR': far, 'F1': f1, 'HSS': hss}

    save_path = f"results_{args.method}_{args.dataname}_Clean.txt"
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write('====================================================\n')
        file.write(f'  {args.method.upper()} evaluation report\n')
        file.write('====================================================\n')
        file.write(f'Dataset: {args.dataname.upper()} | Eval mode: {eval_mode}\n\n')

        file.write('[1] Image reconstruction metrics\n')
        file.write('----------------------------------------------------\n')
        file.write(f'MSE : {final_mse:.6f}\n')
        file.write(f'MAE : {final_mae:.6f}\n')
        file.write(f'PSNR: {final_psnr:.6f} dB\n')
        file.write(f'PCC : {final_pcc:.6f}\n\n')

        file.write('[2] Meteorological binary metrics\n')
        file.write('----------------------------------------------------\n')
        for th in thresholds:
            m = weather_metrics[th]
            file.write(f'Threshold >= {th} {unit}:\n')
            file.write(f"  CSI: {m['CSI']:.6f}\n")
            file.write(f"  POD: {m['POD']:.6f}\n")
            file.write(f"  FAR: {m['FAR']:.6f}\n")
            file.write(f"  F1 : {m['F1']:.6f}\n")
            file.write(f"  HSS: {m['HSS']:.6f}\n\n")

    print(f'evaluation done, report saved to: {os.path.abspath(save_path)}')


if __name__ == '__main__':
    main()

