import gc
import os
import re
import time

import numpy as np
import torch
from tqdm import tqdm

from openstl.core.metrics import metric as core_metric
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
#
MAIN_METRICS = ['mse', 'mae', 'rmse', 'csi', 'pod', 'hss']
ALL_METRICS = [
    'mse', 'mae', 'rmse', 'ssim', 'psnr', 'snr', 'lpips',
    'pod', 'sucr', 'csi', 'hss', 'far', 'f1', 'pcc'
]
VALID_METRICS = set(ALL_METRICS)


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


def sanitize_filename_part(value):
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('_')
    return safe if safe else 'na'


def resolve_metric_selection(metric_set):
    metric_set = 'main' if metric_set is None else str(metric_set).strip().lower()
    if metric_set in ('', 'main'):
        return MAIN_METRICS.copy()
    if metric_set == 'all':
        return ALL_METRICS.copy()

    selected = [m.strip().lower() for m in metric_set.split(',') if m.strip()]
    invalid = [m for m in selected if m not in VALID_METRICS]
    if invalid:
        raise ValueError(f'unsupported metrics: {invalid}, valid={sorted(VALID_METRICS)}')
    if len(selected) == 0:
        return MAIN_METRICS.copy()
    return selected


def infer_run_id(run_id):
    if run_id is not None and str(run_id).strip():
        return str(run_id).strip()
    return time.strftime('%m%d%H', time.localtime())


def build_unique_save_path(save_root, dataname, method, run_id):
    os.makedirs(save_root, exist_ok=True)

    safe_dataname = sanitize_filename_part(dataname)
    safe_method = sanitize_filename_part(method)
    safe_run = sanitize_filename_part(run_id)

    filename = f'{safe_dataname}_{safe_method}_{safe_run}.txt'
    path = os.path.join(save_root, filename)
    if not os.path.exists(path):
        return path

    version = 2
    while True:
        path = os.path.join(save_root, f'{safe_dataname}_{safe_method}_{safe_run}_v{version}.txt')
        if not os.path.exists(path):
            return path
        version += 1


def to_pixel_space(tensor, eval_mode, data_mean, data_std):
    if eval_mode == 'SEVIR':
        return tensor * data_std + data_mean
    if eval_mode == 'CIKM':
        return tensor * 255.0
    return tensor * 255.0 if tensor.max() <= 2.0 else tensor


def parse_and_merge_config():
    parser = create_parser()
    parser.add_argument('--save_root', default='./results/custom_test', type=str,
                        help='Root dir for evaluation reports')
    parser.add_argument('--run_id', default=None, type=str,
                        help='Run id used in report file name, default MMDDHH')
    parser.add_argument('--metrics', dest='metric_set', default='main', type=str,
                        help='Metric set: main | all | comma list')
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
    selected_metrics = resolve_metric_selection(args.metric_set)

    runtime_config = dict(config)
    for key in ['save_root', 'run_id', 'metric_set']:
        runtime_config.pop(key, None)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'start robust evaluation | dataset: {args.dataname} | method: {args.method}')
    print(f'metric set: {selected_metrics}')

    print('building dataloader...')
    _, _, test_loader = get_dataset(args.dataname, runtime_config)

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

    data_mean = float(getattr(test_loader.dataset, 'mean', 0.0))
    data_std = float(getattr(test_loader.dataset, 'std', 1.0))

    print('building model...')
    model, is_official_exprecast = build_model(args, runtime_config, device)
    load_checkpoint(args, model)
    model.eval()

    print(f'start inference and metrics ({eval_mode})...')

    need_confusion = any(m in selected_metrics for m in ['pod', 'sucr', 'csi', 'hss', 'far', 'f1'])
    tps = {th: 0.0 for th in thresholds} if need_confusion else {}
    fns = {th: 0.0 for th in thresholds} if need_confusion else {}
    fps = {th: 0.0 for th in thresholds} if need_confusion else {}
    tns = {th: 0.0 for th in thresholds} if need_confusion else {}

    need_mse = any(m in selected_metrics for m in ['mse', 'rmse', 'psnr'])
    need_mae = 'mae' in selected_metrics
    need_pcc = 'pcc' in selected_metrics
    aux_metrics = [m for m in ['ssim', 'snr', 'lpips'] if m in selected_metrics]

    aux_metric_sums = {m: 0.0 for m in aux_metrics}
    aux_metric_skipped = {}

    total_mse = 0.0
    total_mae = 0.0
    total_pcc = 0.0
    total_samples = 0
    total_pixels = 0
    data_check = None

    # Official exPreCast uses this normalization for released checkpoints.
    mean_ex, scale_ex = 33.44, 47.54

    with torch.inference_mode():
        for batch_x, batch_y in tqdm(test_loader, desc='Testing'):
            if data_check is None:
                data_check = {
                    'x_shape': tuple(batch_x.shape),
                    'y_shape': tuple(batch_y.shape),
                    'x_dtype': str(batch_x.dtype),
                    'y_dtype': str(batch_y.dtype),
                    'x_min': float(batch_x.min().item()),
                    'x_max': float(batch_x.max().item()),
                    'x_mean': float(batch_x.mean().item()),
                    'y_min': float(batch_y.min().item()),
                    'y_max': float(batch_y.max().item()),
                    'y_mean': float(batch_y.mean().item()),
                }

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pixel_x = to_pixel_space(batch_x, eval_mode, data_mean, data_std)
            pixel_y = to_pixel_space(batch_y, eval_mode, data_mean, data_std)
            targets_255 = torch.clamp(pixel_y, 0.0, 255.0)

            if is_official_exprecast:
                batch_x_input = (pixel_x - mean_ex) / scale_ex
                preds = model(batch_x_input)
                raw_pred = preds * scale_ex + mean_ex
                preds_255 = torch.clamp(raw_pred, 0.0, 255.0)
            elif args.method == 'alphapre':
                batch_x_input = torch.clamp(pixel_x / 255.0, 0.0, 1.0)
                preds, _ = model.predict(frames_in=batch_x_input, compute_loss=False)
                preds_255 = torch.clamp(preds, 0.0, 1.0) * 255.0
            else:
                batch_x_input = torch.clamp(pixel_x / 255.0, 0.0, 1.0)
                preds = model(batch_x_input)
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

            batch_size = preds_255.shape[0]
            total_samples += batch_size

            if need_mse:
                total_mse += torch.sum((preds_255 - targets_255) ** 2).item()
            if need_mae:
                total_mae += torch.sum(torch.abs(preds_255 - targets_255)).item()
            if need_pcc:
                total_pcc += calculate_pcc(preds_255, targets_255) * batch_size
            if need_mse or need_mae:
                total_pixels += preds_255.numel()

            if need_confusion:
                preds_metric = preds_255
                targets_metric = targets_255
                if eval_mode == 'CIKM':
                    if preds_metric.shape[-1] == 112 and preds_metric.shape[-2] == 112:
                        preds_metric = preds_metric[..., 5:-6, 5:-6]
                        targets_metric = targets_metric[..., 5:-6, 5:-6]
                    preds_eval = pixel_to_rainfall_tensor(preds_metric / 255.0)
                    targets_eval = pixel_to_rainfall_tensor(targets_metric / 255.0)
                else:
                    preds_eval = preds_metric
                    targets_eval = targets_metric

                for th in thresholds:
                    b_preds = preds_eval >= th
                    b_targets = targets_eval >= th
                    tps[th] += torch.sum(b_preds & b_targets).item()
                    fns[th] += torch.sum(~b_preds & b_targets).item()
                    fps[th] += torch.sum(b_preds & ~b_targets).item()
                    tns[th] += torch.sum(~b_preds & ~b_targets).item()

            if aux_metrics:
                preds_unit = torch.clamp(preds_255 / 255.0, 0.0, 1.0)
                targets_unit = torch.clamp(targets_255 / 255.0, 0.0, 1.0)
                preds_np = preds_unit.detach().cpu().numpy()
                targets_np = targets_unit.detach().cpu().numpy()

                for metric_name in aux_metrics:
                    if metric_name in aux_metric_skipped:
                        continue
                    try:
                        eval_res, _ = core_metric(preds_np, targets_np, metrics=[metric_name], return_log=False)
                        if metric_name in eval_res:
                            aux_metric_sums[metric_name] += float(eval_res[metric_name]) * batch_size
                        else:
                            aux_metric_skipped[metric_name] = 'metric not returned by evaluator'
                    except Exception as exc:
                        aux_metric_skipped[metric_name] = str(exc).strip().split('\n')[0]

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    scalar_metrics = {}
    if need_mse:
        final_mse = total_mse / max(total_pixels, 1)
        if 'mse' in selected_metrics:
            scalar_metrics['mse'] = final_mse
        if 'rmse' in selected_metrics:
            scalar_metrics['rmse'] = float(np.sqrt(final_mse))
        if 'psnr' in selected_metrics:
            scalar_metrics['psnr'] = float(10 * np.log10((255.0 ** 2) / (final_mse + 1e-8)))

    if need_mae:
        scalar_metrics['mae'] = total_mae / max(total_pixels, 1)

    if need_pcc:
        scalar_metrics['pcc'] = total_pcc / max(total_samples, 1)

    for metric_name in aux_metrics:
        if metric_name in aux_metric_skipped:
            continue
        scalar_metrics[metric_name] = aux_metric_sums[metric_name] / max(total_samples, 1)

    threshold_metrics = {}
    if need_confusion:
        eps = 1e-6
        for th in thresholds:
            tp, fn, fp, tn = tps[th], fns[th], fps[th], tns[th]
            pod = tp / (tp + fn + eps)
            sucr = tp / (tp + fp + eps)
            far = fp / (tp + fp + eps)
            csi = tp / (tp + fn + fp + eps)
            f1 = 2 * tp / (2 * tp + fp + fn + eps)

            total = tp + fn + fp + tn
            exp_hits = (tp + fn) * (tp + fp) / (total + eps)
            exp_cns = (tn + fn) * (tn + fp) / (total + eps)
            hss = (tp + tn - exp_hits - exp_cns) / (total - exp_hits - exp_cns + eps)

            threshold_metrics[th] = {
                'pod': pod,
                'sucr': sucr,
                'csi': csi,
                'hss': hss,
                'far': far,
                'f1': f1,
            }

    run_id = infer_run_id(args.run_id)
    save_path = build_unique_save_path(args.save_root, args.dataname, args.method, run_id)
    now_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    with open(save_path, 'w', encoding='utf-8') as file:
        file.write('====================================================================\n')
        file.write('Custom Test Evaluation Report\n')
        file.write('====================================================================\n\n')

        file.write('[Run Info]\n')
        file.write('--------------------------------------------------------------------\n')
        file.write(f'timestamp    : {now_str}\n')
        file.write(f'device       : {device}\n')
        file.write(f'dataname     : {args.dataname}\n')
        file.write(f'method       : {args.method}\n')
        file.write(f'run_id       : {run_id}\n')
        file.write(f'ckpt_path    : {os.path.abspath(args.ckpt_path) if getattr(args, "ckpt_path", None) else "None"}\n')
        file.write(f'save_path    : {os.path.abspath(save_path)}\n\n')

        file.write('[Config Snapshot]\n')
        file.write('--------------------------------------------------------------------\n')
        file.write(f'pre_seq_length : {runtime_config.get("pre_seq_length")}\n')
        file.write(f'aft_seq_length : {runtime_config.get("aft_seq_length")}\n')
        file.write(f'total_length   : {runtime_config.get("total_length")}\n')
        file.write(f'in_shape       : {runtime_config.get("in_shape")}\n')
        file.write(f'batch_size     : {runtime_config.get("batch_size")}\n')
        file.write(f'val_batch_size : {runtime_config.get("val_batch_size")}\n')
        file.write(f'num_workers    : {runtime_config.get("num_workers")}\n')
        file.write(f'eval_mode      : {eval_mode}\n')
        file.write(f'thresholds     : {thresholds} ({unit})\n')
        file.write(f'metrics_input  : {args.metric_set}\n')
        file.write(f'metrics_used   : {selected_metrics}\n')
        file.write(f'data_mean/std  : {data_mean:.6f}/{data_std:.6f}\n\n')

        file.write('[Data Check]\n')
        file.write('--------------------------------------------------------------------\n')
        if data_check is None:
            file.write('No batch captured.\n\n')
        else:
            file.write(f"x shape/dtype : {data_check['x_shape']} | {data_check['x_dtype']}\n")
            file.write(f"y shape/dtype : {data_check['y_shape']} | {data_check['y_dtype']}\n")
            file.write(f"x min/max/mean: {data_check['x_min']:.6f} / {data_check['x_max']:.6f} / {data_check['x_mean']:.6f}\n")
            file.write(f"y min/max/mean: {data_check['y_min']:.6f} / {data_check['y_max']:.6f} / {data_check['y_mean']:.6f}\n\n")

        file.write('[Metrics]\n')
        file.write('--------------------------------------------------------------------\n')
        metric_print_order = ['mse', 'mae', 'rmse', 'psnr', 'ssim', 'snr', 'lpips', 'pcc']
        for name in metric_print_order:
            if name not in selected_metrics:
                continue
            if name in scalar_metrics:
                file.write(f'{name.upper():<6}: {scalar_metrics[name]:.6f}\n')
            elif name in aux_metric_skipped:
                file.write(f'{name.upper():<6}: skipped ({aux_metric_skipped[name]})\n')
            else:
                file.write(f'{name.upper():<6}: unavailable\n')

        weather_order = ['csi', 'pod', 'hss', 'sucr', 'far', 'f1']
        selected_weather = [m for m in weather_order if m in selected_metrics]
        if selected_weather:
            file.write('\n[Meteorological Binary Metrics]\n')
            file.write('--------------------------------------------------------------------\n')
            for th in thresholds:
                values = threshold_metrics.get(th, {})
                file.write(f'Threshold >= {th} {unit}\n')
                for m in selected_weather:
                    if m in values:
                        file.write(f'  {m.upper():<4}: {values[m]:.6f}\n')
                file.write('\n')
        else:
            file.write('\nNo meteorological binary metric requested.\n')

    print(f'evaluation done, report saved to: {os.path.abspath(save_path)}')


if __name__ == '__main__':
    main()
