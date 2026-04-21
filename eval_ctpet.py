import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from models.model_configs import instantiate_model
from training.dataloader import CellDataLoader_Eval
from training.eval_loop import eval_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ctpet", type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--num_workers", default=None, type=int)
    parser.add_argument("--fid_samples", default=None, type=int)
    parser.add_argument("--use_initial", default=None, type=int)
    parser.add_argument("--compute_fid", action="store_true")
    parser.add_argument("--test_run", action="store_true")
    return parser.parse_args()


def load_config(config_name):
    config_path = Path("configs") / f"{config_name}.yaml"
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def merge_args(cli_args, yaml_config):
    merged = dict(yaml_config)
    for key, value in vars(cli_args).items():
        if value is not None:
            merged[key] = value
    merged.setdefault("dataset", "ctpet")
    merged.setdefault("dataset_name", "ctpet")
    merged.setdefault("compute_recon_metrics", True)
    merged.setdefault("compute_fid", False)
    merged.setdefault("num_workers", 4)
    merged.setdefault("batch_size", 8)
    merged.setdefault("fid_samples", 1024)
    merged.setdefault("use_initial", 1)
    merged.setdefault("interpolate", False)
    merged.setdefault("normalize", True)
    merged.setdefault("class_drop_prob", 0.0)
    merged.setdefault("cfg_scale", 0.0)
    merged.setdefault("ode_method", "midpoint")
    merged.setdefault("ode_options", {"step_size": 0.01})
    merged.setdefault("edm_schedule", False)
    merged.setdefault("discrete_flow_matching", False)
    merged.setdefault("sampling_dtype", "float32")
    merged.setdefault("sym_func", False)
    merged.setdefault("sym", 0.0)
    merged.setdefault("save_fid_samples", False)
    merged.setdefault("pin_mem", True)
    merged.setdefault("noise_level", 0.2)
    merged.setdefault("seed", 0)
    return SimpleNamespace(**merged)


def main():
    cli_args = parse_args()
    yaml_config = load_config(cli_args.config)
    args = merge_args(cli_args, yaml_config)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    datamodule = CellDataLoader_Eval(args)
    model = instantiate_model(
        architechture=args.dataset,
        is_discrete=args.discrete_flow_matching,
        use_ema=getattr(args, "use_ema", False),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    eval_stats = eval_model(
        model=model,
        data_loader=datamodule.test_dataloader(),
        device=device,
        epoch=checkpoint.get("epoch", -1) if isinstance(checkpoint, dict) else -1,
        fid_samples=args.fid_samples,
        args=args,
        datamodule=datamodule,
        use_initial=args.use_initial,
        interpolate=args.interpolate,
    )
    print(json.dumps(eval_stats, indent=4))


if __name__ == "__main__":
    main()
