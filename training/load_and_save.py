# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path

import torch
from training.distributed_mode import is_main_process


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def save_model(
    args, epoch, model, model_without_ddp, optimizer, lr_schedule, loss_scaler
):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [
            output_dir / ("checkpoint-%s.pth" % epoch_name),
            output_dir / "checkpoint.pth",
        ]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_schedule": lr_schedule.state_dict(),
                "epoch": epoch,
                "scaler": loss_scaler.state_dict(),
                "args": args,
                "best_metric_name": getattr(args, "best_metric", None),
                "best_metric_value": getattr(args, "best_metric_value", None),
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {"epoch": epoch}
        model.save_checkpoint(
            save_dir=args.output_dir,
            tag="checkpoint-%s" % epoch_name,
            client_state=client_state,
        )


def best_metric_default(metric_name):
    return float("inf") if metric_name in {"mae", "lpips"} else float("-inf")


def is_better_metric(metric_name, metric_value, current_best):
    if metric_name in {"mae", "lpips"}:
        return metric_value < current_best
    return metric_value > current_best


def save_best_model(
    args,
    epoch,
    metric_name,
    metric_value,
    model_without_ddp,
    optimizer,
    lr_schedule,
    loss_scaler,
):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / f"checkpoint_best_{metric_name}.pth"
    to_save = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_schedule": lr_schedule.state_dict(),
        "epoch": epoch,
        "scaler": loss_scaler.state_dict() if loss_scaler is not None else None,
        "args": args,
        "best_metric_name": metric_name,
        "best_metric_value": metric_value,
    }
    save_on_master(to_save, checkpoint_path)


def load_model(args, model_without_ddp, optimizer, loss_scaler, lr_schedule):
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        print("Resume checkpoint %s" % args.resume)
        if "best_metric_value" in checkpoint:
            args.best_metric_value = checkpoint["best_metric_value"]
        elif "args" in checkpoint and hasattr(checkpoint["args"], "best_metric_value"):
            args.best_metric_value = checkpoint["args"].best_metric_value
        if (
            "optimizer" in checkpoint
            and "epoch" in checkpoint
            and not (hasattr(args, "eval") and args.eval)
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_schedule.load_state_dict(checkpoint["lr_schedule"])
            args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            print("With optim & sched!")
