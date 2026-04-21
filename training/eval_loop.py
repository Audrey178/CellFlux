# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import gc
import json
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Iterable

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import save_image
from tqdm import tqdm

from training import distributed_mode
from training.data_utils import convert_5ch_to_3ch, convert_6ch_to_3ch
from training.dataloader import CTPETDataLoader
from training.edm_time_discretization import get_time_discretization
from training.train_loop import MASK_TOKEN

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:  # pragma: no cover - dependency is declared in environment.yml
    FrechetInceptionDistance = None

try:
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
except ImportError:  # pragma: no cover - dependency is declared in environment.yml
    PeakSignalNoiseRatio = None
    StructuralSimilarityIndexMeasure = None

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
except ImportError:  # pragma: no cover - dependency is optional at runtime
    LearnedPerceptualImagePatchSimilarity = None

logger = logging.getLogger(__name__)


class CFGScaledModel(ModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float, extra: dict
    ):
        module = (
            self.model.module
            if isinstance(self.model, DistributedDataParallel)
            else self.model
        )
        is_discrete = isinstance(module, DiscreteUNetModel) or (
            isinstance(module, EMA) and isinstance(module.model, DiscreteUNetModel)
        )
        assert (
            cfg_scale == 0.0 or not is_discrete
        ), f"Cfg scaling does not work for the logit outputs of discrete models. Got cfg weight={cfg_scale} and model {type(self.model)}."
        t = torch.zeros(x.shape[0], device=x.device) + t

        if cfg_scale != 0.0:
            with torch.cuda.amp.autocast(), torch.no_grad():
                conditional = self.model(x, t, extra=extra)
                condition_free = self.model(x, t, extra={})
            result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
        else:
            with torch.cuda.amp.autocast(), torch.no_grad():
                result = self.model(x, t, extra=extra)

        self.nfe_counter += 1
        if is_discrete:
            return torch.softmax(result.to(dtype=torch.float32), dim=-1)
        return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter



def _denormalize_images(images, normalize=True):
    if normalize:
        images = images * 0.5 + 0.5
    return torch.clamp(images, min=0.0, max=1.0)


def _prepare_samples_for_eval(images, normalize=True):
    images = _denormalize_images(images, normalize=normalize)
    return images


def _repeat_grayscale_to_rgb(images):
    if images.shape[1] == 1:
        return images.repeat(1, 3, 1, 1)
    return images


def _compute_manual_ssim(pred, target, data_range=1.0):
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = pred.mean(dim=(-1, -2), keepdim=True)
    mu_y = target.mean(dim=(-1, -2), keepdim=True)
    sigma_x = ((pred - mu_x) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma_y = ((target - mu_y) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=(-1, -2), keepdim=True)
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / torch.clamp(denominator, min=1e-8)
    return ssim_map.mean()


def _compute_manual_psnr(pred, target, data_range=1.0):
    mse = F.mse_loss(pred, target)
    mse = torch.clamp(mse, min=1e-12)
    return 10.0 * torch.log10(torch.tensor(data_range**2, device=pred.device) / mse)


def _dist_reduce_metric_sums(metric_sums, device):
    if not distributed_mode.is_dist_avail_and_initialized():
        return metric_sums

    keys = sorted(metric_sums.keys())
    values = torch.tensor([metric_sums[key] for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {key: values[idx].item() for idx, key in enumerate(keys)}


def _to_uint8_image(image_tensor):
    image = torch.clamp(image_tensor, min=0.0, max=1.0).detach().cpu()
    if image.shape[0] == 1:
        return (image.squeeze(0).numpy() * 255.0).astype(np.uint8)
    return (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def save_ctpet_visualization(ct_image, real_pet, pred_pet, save_path, error_map_mode):
    if error_map_mode != "absolute":
        raise ValueError(f"Unsupported error_map_mode: {error_map_mode}")

    error_map = torch.abs(pred_pet - real_pet)
    images = [ct_image, real_pet, pred_pet]
    titles = ["CT Input", "Real PET", "Pred PET"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, image, title in zip(axes[:3], images, titles):
        img_uint8 = _to_uint8_image(image)
        if img_uint8.ndim == 2:
            axis.imshow(img_uint8, cmap="gray", vmin=0, vmax=255)
        else:
            axis.imshow(img_uint8)
        axis.set_title(title)
        axis.axis("off")

    error_uint8 = _to_uint8_image(error_map)
    axes[3].imshow(error_uint8, cmap="inferno", vmin=0, vmax=255)
    axes[3].set_title("|Pred-Real|")
    axes[3].axis("off")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _maybe_save_ctpet_visualizations(
    args,
    epoch,
    saved_count,
    ct_images,
    real_samples,
    synthetic_samples,
    file_names,
):
    if not getattr(args, "save_visualizations", False):
        return saved_count
    if not distributed_mode.is_main_process():
        return saved_count
    if args.output_dir is None:
        return saved_count

    remaining = max(args.num_visual_samples - saved_count, 0)
    if remaining == 0:
        return saved_count

    epoch_dir = Path(args.output_dir) / "visualizations" / f"epoch_{epoch}"
    for batch_index in range(min(remaining, real_samples.shape[0])):
        file_name = Path(file_names[batch_index]).stem
        save_path = epoch_dir / f"sample_{saved_count + batch_index:04d}_{file_name}.png"
        save_ctpet_visualization(
            ct_image=ct_images[batch_index],
            real_pet=real_samples[batch_index],
            pred_pet=synthetic_samples[batch_index],
            save_path=save_path,
            error_map_mode=args.error_map_mode,
        )
    return saved_count + min(remaining, real_samples.shape[0])


def _save_metrics_json(args, epoch, metrics):
    if args.output_dir is None or not distributed_mode.is_main_process():
        return
    metrics_dir = Path(args.output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"epoch_{epoch}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def eval_model(
    model: DistributedDataParallel,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    fid_samples: int,
    args: Namespace,
    datamodule: CTPETDataLoader,
    use_initial: int = 0,
    interpolate: bool = False,
):
    gc.collect()
    cfg_scaled_model = CFGScaledModel(model=model)
    cfg_scaled_model.train(False)

    if args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
        p = torch.zeros(size=[257], dtype=torch.float32, device=device)
        p[256] = 1.0
        solver = MixtureDiscreteEulerSolver(
            model=cfg_scaled_model,
            path=path,
            vocabulary_size=257,
            source_distribution_p=p,
        )
    else:
        solver = ODESolver(velocity_model=cfg_scaled_model)
        ode_opts = args.ode_options

    compute_recon_metrics = bool(getattr(args, "compute_recon_metrics", False))
    ssim_metric = None
    psnr_metric = None
    lpips_metric = None
    if compute_recon_metrics:
        if StructuralSimilarityIndexMeasure is not None:
            ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        if PeakSignalNoiseRatio is not None:
            psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        if LearnedPerceptualImagePatchSimilarity is not None:
            lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
        else:
            logger.warning("LPIPS dependency is unavailable; eval_lpips will be reported as NaN.")

    num_synthetic = 0
    snapshots_saved = False
    visuals_saved = 0
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    trt2ctrl_idx = {}
    metric_sums = {"count": 0.0, "mae": 0.0, "ssim": 0.0, "psnr": 0.0, "lpips": 0.0, "lpips_count": 0.0}

    for data_iter_step, batch in tqdm(enumerate(data_loader)):
        x_real = batch["X"]
        x_real_ct, x_real_pet = x_real
        x_real_ct = x_real_ct.to(device)
        x_real_pet = x_real_pet.to(device)

        samples = None
        labels = None

        if compute_recon_metrics or getattr(args, "save_visualizations", False):
            cfg_scaled_model.reset_nfe_counter()
            if args.discrete_flow_matching:
                x_0 = torch.zeros(samples.shape, dtype=torch.long, device=device) + MASK_TOKEN
                if args.sym_func:
                    sym = lambda t: 12.0 * torch.pow(t, 2.0) * torch.pow(1.0 - t, 0.25)
                else:
                    sym = args.sym
                dtype = torch.float32 if args.sampling_dtype == "float32" else torch.float64
                synthetic_samples = solver.sample(
                    x_init=x_0,
                    step_size=1.0 / args.discrete_fm_steps,
                    verbose=False,
                    div_free=sym,
                    dtype_categorical=dtype,
                    label=labels,
                    cfg_scale=args.cfg_scale,
                )
            else:
                if use_initial == 1:
                    x_0 = x_real_ct
                elif use_initial == 2:
                    x_0 = x_real_pet + torch.randn(
                        x_real_pet.shape, dtype=torch.float32, device=device
                    ) * args.noise_level
                else:
                    x_0 = torch.randn(x_real_ct.shape, dtype=torch.float32, device=device)

                if args.edm_schedule:
                    time_grid = get_time_discretization(nfes=ode_opts["nfe"])
                else:
                    time_grid = torch.tensor([0.0, 1.0], device=device)

                synthetic_samples = solver.sample(
                    time_grid=time_grid,
                    x_init=x_0,
                    method=args.ode_method,
                    return_intermediates=interpolate,
                    atol=ode_opts["atol"] if "atol" in ode_opts else 1e-5,
                    rtol=ode_opts["rtol"] if "rtol" in ode_opts else 1e-5,
                    step_size=ode_opts["step_size"] if "step_size" in ode_opts else None,
                    cfg_scale=args.cfg_scale,
                )
                if interpolate:
                    save_interpolation_grid(
                        synthetic_samples,
                        x_real_ct,
                        x_real_pet,
                        time_grid,
                        save_dir=Path(args.output_dir) / "interpolation",
                        title="Interpolation Visualization",
                        normalize=getattr(args, "normalize", True),
                    )
                    return {}

            logger.info(
                f"{x_real_ct.shape[0]} samples generated in {cfg_scaled_model.get_nfe()} evaluations."
            )

            pet_samples = _prepare_samples_for_eval(
                x_real_pet,
                dataset_name=args.dataset_name,
                normalize=getattr(args, "normalize", True),
            )
            synthetic_samples = _prepare_samples_for_eval(
                synthetic_samples,
                dataset_name=args.dataset_name,
                normalize=getattr(args, "normalize", True),
            )
            ct_samples = _prepare_samples_for_eval(
                x_real_ct,
                dataset_name=args.dataset_name,
                normalize=getattr(args, "normalize", True),
            )

            if compute_recon_metrics:
                batch_size = pet_samples.shape[0]
                metric_sums["count"] += batch_size
                metric_sums["mae"] += torch.mean(torch.abs(synthetic_samples - pet_samples)).item() * batch_size

                if ssim_metric is not None:
                    batch_ssim = ssim_metric(synthetic_samples, pet_samples).item()
                    ssim_metric.reset()
                else:
                    batch_ssim = _compute_manual_ssim(synthetic_samples, pet_samples).item()
                metric_sums["ssim"] += batch_ssim * batch_size

                if psnr_metric is not None:
                    batch_psnr = psnr_metric(synthetic_samples, pet_samples).item()
                    psnr_metric.reset()
                else:
                    batch_psnr = _compute_manual_psnr(synthetic_samples, pet_samples).item()
                metric_sums["psnr"] += batch_psnr * batch_size

                if lpips_metric is not None:
                    pred_rgb = _repeat_grayscale_to_rgb(synthetic_samples) * 2.0 - 1.0
                    real_rgb = _repeat_grayscale_to_rgb(pet_samples) * 2.0 - 1.0
                    batch_lpips = lpips_metric(pred_rgb, real_rgb).item()
                    lpips_metric.reset()
                    metric_sums["lpips"] += batch_lpips * batch_size
                    metric_sums["lpips_count"] += batch_size

            visuals_saved = _maybe_save_ctpet_visualizations(
                args=args,
                epoch=epoch,
                saved_count=visuals_saved,
                ct_images=ct_samples,
                real_samples=pet_samples,
                synthetic_samples=synthetic_samples,
                file_names=batch["file_names"][1],
            )

        if args.test_run:
            break

    metric_sums = _dist_reduce_metric_sums(metric_sums, device)
    eval_stats = {}
    if compute_recon_metrics and metric_sums["count"] > 0:
        denom = metric_sums["count"]
        eval_stats["mae"] = metric_sums["mae"] / denom
        eval_stats["ssim"] = metric_sums["ssim"] / denom
        eval_stats["psnr"] = metric_sums["psnr"] / denom
        eval_stats["lpips"] = (
            metric_sums["lpips"] / metric_sums["lpips_count"]
            if metric_sums["lpips_count"] > 0
            else float("nan")
        )

    _save_metrics_json(args, epoch, eval_stats)
    return eval_stats


def save_interpolation_grid(
    intermediate_images: torch.Tensor,
    real_ct: torch.Tensor,
    real_pet: torch.Tensor,
    time_grid: torch.Tensor,
    save_dir: Path,
    title: str = "Interpolation Visualization",
    normalize: bool = True,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    def to_numpy(image_tensor):
        img = _denormalize_images(image_tensor, normalize=normalize)
        if img.shape[0] == 1:
            return (img.squeeze(0).cpu().numpy() * 255.0).astype("uint8")
        return (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")

    batch_size = real_ct.shape[0]
    for batch_index in range(batch_size):
        real_ct_img = to_numpy(real_ct[batch_index])
        real_pet_img = to_numpy(real_pet[batch_index])
        intermediate_imgs = [
            to_numpy(intermediate_images[timestep_index, batch_index])
            for timestep_index in range(intermediate_images.shape[0])
        ]

        images = [real_ct_img] + intermediate_imgs + [real_pet_img]
        labels = ["Real CT"] + [f"t={t:.2f}" for t in time_grid] + ["Real PET"]

        num_images = len(images)
        num_cols = max(1, (num_images + 4) // 5)
        fig, axes = plt.subplots(5, num_cols, figsize=(2 * num_cols, 10))
        axes = np.array(axes, dtype=object).reshape(5, num_cols)
        fig.suptitle(
            f"{title} - Sample {batch_index}",
            fontsize=16,
        )

        for image_index in range(5 * num_cols):
            row, col = divmod(image_index, num_cols)
            if image_index < num_images:
                if images[image_index].ndim == 2:
                    axes[row, col].imshow(images[image_index], cmap="gray", vmin=0, vmax=255)
                else:
                    axes[row, col].imshow(images[image_index])
                axes[row, col].set_title(labels[image_index], fontsize=8)
            axes[row, col].axis("off")

        sample_save_path = save_dir / f"sample_{batch_index}_interpolation_grid.png"
        plt.tight_layout()
        plt.savefig(sample_save_path, dpi=300)
        plt.close(fig)
