# Copyright (c) CAIRI AI Lab. All rights reserved

import warnings
warnings.filterwarnings('ignore')

from openstl.api import BaseExperiment
from openstl.utils import (
    create_parser,
    default_parser,
    get_dist_info,
    load_config,
    update_config,
    canonicalize_dataname,
    build_default_config_path,
    get_cli_override_keys,
)


if __name__ == '__main__':
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

    loaded_cfg = load_config(cfg_path)
    if args.overwrite:
        config = update_config(config, loaded_cfg, cli_override_keys=set(config.keys()))
    else:
        config = update_config(config, loaded_cfg, cli_override_keys=cli_override_keys)

    default_values = default_parser()
    for attribute, value in default_values.items():
        if config.get(attribute) is None:
            config[attribute] = value

    print('>' * 35 + ' training ' + '<' * 35)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()
    exp.train()

    if rank == 0:
        print('>' * 35 + ' testing  ' + '<' * 35)
    exp.test()
