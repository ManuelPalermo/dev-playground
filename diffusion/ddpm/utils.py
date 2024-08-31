"""Utility functions to process, visualize and log data."""

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torcheval.metrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid

from ddpm.diffusion import GaussianDiffusion

try:
    from moviepy.editor import ImageSequenceClip
except RuntimeError:
    os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
    from moviepy.editor import ImageSequenceClip


def inverse_transform(tensors, max_val: float = 255.0):
    """Convert tensors from [-1., 1.] to [0., max_val]."""
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * max_val


def min_max_normalize(tensors: torch.Tensor, axis: int = -1, max_val: float = 1.0):
    """Convert tensors from range to [min, max]."""
    t_min = tensors.min(dim=axis, keepdim=True)[0]
    t_max = tensors.max(dim=axis, keepdim=True)[0]
    return ((tensors - t_min) / (t_max - t_min)) * max_val


def prepare_fid_frames(frames: torch.Tensor) -> torch.Tensor:
    """Convert frames from [-1,1] to [0,1]."""
    return inverse_transform(torch.nan_to_num(frames), max_val=1.0).to(torch.float32)


def project_pointcloud_to_bev_image(
    pcs: torch.Tensor, res: float = 0.01, space_m: tuple[float, ...] = (-0.75, 0.75)
) -> torch.Tensor:
    """Projects pointcloud xyz to BeV image."""
    B, C, _ = pcs.shape

    # assumes square region to project
    img_shape = (int((space_m[1] - space_m[0]) / res), int((space_m[1] - space_m[0]) / res))

    image = -torch.ones((B, img_shape[1], img_shape[0]), dtype=pcs.dtype, device=pcs.device)

    for b_idx, sample_pc in enumerate(pcs):
        x_points = sample_pc[0, :]
        y_points = sample_pc[1, :]
        z_points = sample_pc[2, :]
        intensity = sample_pc[3, :] if C == 4 else z_points  # if no intensity data then visualize height

        # filter invalid points (xyz=0 or xyz=nan) and points outside space
        valid_point_mask = ~(
            ((x_points == 0.0).to(torch.bool) & (y_points == 0.0).to(torch.bool) & (z_points == 0.0).to(torch.bool))
            | (
                (x_points == torch.nan).to(torch.bool)
                & (y_points == torch.nan).to(torch.bool)
                & (z_points == torch.nan).to(torch.bool)
            )
        )
        inside_space_mask = ~(
            (x_points >= space_m[1]).to(torch.bool)
            | (x_points <= space_m[0]).to(torch.bool)
            | (y_points >= space_m[1]).to(torch.bool)
            | (y_points <= space_m[0]).to(torch.bool)
            | (z_points <= 0).to(torch.bool)
        )
        x_points = x_points[valid_point_mask & inside_space_mask]
        y_points = y_points[valid_point_mask & inside_space_mask]
        z_points = z_points[valid_point_mask & inside_space_mask]
        intensity = intensity[valid_point_mask & inside_space_mask]

        if len(intensity) == 0:  # all points have been removed
            continue

        # convert from world to pixel positions (swaps x/y axis)
        x_idx = ((-y_points / res) - ((img_shape[0] // 2) - 1)).to(torch.int32)
        y_idx = ((-x_points / res) + ((img_shape[1] // 2) - 1)).to(torch.int32)

        # normalize pixel values
        pixel_values = min_max_normalize(intensity)

        # scater pixels to image
        image[b_idx, x_idx, y_idx] = pixel_values

    return image[..., None, :, :]  # add channel dim: B, C, H, W


def prepare_vis_frames(frames: torch.Tensor, nrow: int | None = 10, data_type: str = "img") -> np.ndarray:
    """Prepares torch frames for visualization."""
    if data_type == "pcd":
        frames = project_pointcloud_to_bev_image(frames)

    frames_gen_vis = inverse_transform(torch.nan_to_num(frames), max_val=255.0)

    if nrow is not None:
        # GW x GH x C
        frames_gen_vis = torch.permute(make_grid(frames_gen_vis, nrow=nrow, pad_value=255), dims=(1, 2, 0))
    else:
        # B x W x H x C
        frames_gen_vis = torch.permute(frames_gen_vis, dims=(0, 2, 3, 1))

    return frames_gen_vis.to(torch.uint8).to("cpu").numpy()


def collect_n_samples_from_dataloader(dataloader: DataLoader, num: int) -> torch.Tensor:
    """Collects desired number of samples from dataloader."""
    frames_gt = []
    n_to_sample = np.ceil(num / dataloader.batch_size)
    for idx, (x0s, _) in enumerate(dataloader):
        x0s = x0s.to(device="cpu", dtype=torch.float32)
        frames_gt.append(x0s)
        if idx >= n_to_sample:
            break
    return torch.cat(frames_gt, dim=0)[:num]


def compute_fid_metric(frames_gen: torch.Tensor, dataloader: DataLoader) -> float:
    """Calculate FID metric from generated frames + sampled gt from a dataloader."""
    assert frames_gen.shape
    is_rgb_img = (frames_gen.shape[1] == 3) and (len(frames_gen.shape) == 4)
    if is_rgb_img:
        frames_gt = collect_n_samples_from_dataloader(dataloader=dataloader, num=len(frames_gen))

        fid_metric = FrechetInceptionDistance(device="cpu")
        fid_metric.update(images=prepare_fid_frames(frames_gt), is_real=True)
        fid_metric.update(images=prepare_fid_frames(frames_gen), is_real=False)
        fid_score = float(fid_metric.compute().numpy())

    else:
        fid_score = np.nan

    return fid_score


def log_generation_examples(
    frames_steps: list[torch.Tensor],
    epoch: int,
    eval_examples_dir: str,
    timesteps: int,
    num_classes: int,
    data_type: str,
    steps_to_vis: int = 15,
) -> None:
    """Visualizes frames from reverse diffusion and saves some timesteps for visualization."""
    # prepare generated frames to be visualized
    frames_grid_gen_vis = [
        prepare_vis_frames(frames_gen, nrow=(len(frames_gen) // num_classes), data_type=data_type)
        for frames_gen in frames_steps
    ]

    # save final denoised frames
    plt.imsave(
        fname=os.path.join(eval_examples_dir, f"frames_{epoch}_final.png"),
        arr=frames_grid_gen_vis[-1],
    )

    # save snapshots of frames at different timesteps
    vis_tsteps = np.linspace(start=0, stop=timesteps - 1, num=steps_to_vis, dtype=np.int64)
    vis_images = torch.linspace(
        0,
        frames_steps[0].shape[0] - 1,
        num_classes,
        dtype=torch.long,
        device="cpu",
    )
    frames_gen_vis = [
        prepare_vis_frames(frames_gen[vis_images, ...], nrow=1, data_type=data_type)
        for tstep, frames_gen in enumerate(frames_steps)
        if tstep in vis_tsteps
    ]

    figsize = (min(max(int(num_classes * 2), 12), 30), min(max(num_classes, 6), 15))
    fig, ax = plt.subplots(1, len(frames_gen_vis), figsize=figsize, facecolor="white")
    for i, (timestep, sample) in enumerate(zip(vis_tsteps[::-1], frames_gen_vis)):
        ax[i].imshow(sample)
        ax[i].set_title(f"t={timestep}", fontsize=8)
        ax[i].axis("off")
        ax[i].grid(False)
    fig.suptitle("Reverse Diffusion Process", y=0.98)
    fig.tight_layout()
    fig.savefig(fname=os.path.join(eval_examples_dir, f"frames_{epoch}_reverse_diff.png"))
    plt.close()

    # save gif with reverse diffusion
    gif_name = f"frames_{epoch}_reverse_diff.gif"
    clip = ImageSequenceClip(frames_grid_gen_vis, fps=timesteps).speedx(10.0)
    clip.write_gif(gif_name)
    shutil.move(src=gif_name, dst=os.path.join(eval_examples_dir, gif_name))


def log_forward_diffusion_examples(
    dataloader: DataLoader,
    diffusion: GaussianDiffusion,
    num_samples: int,
    eval_examples_dir: str,
    steps_to_vis: int = 15,
    data_type: str = "img",
) -> None:
    """Performs forward diffusion on samples from dataloader and saves given timesteps for visualization."""
    # get a batch of samples
    frames_gt = collect_n_samples_from_dataloader(dataloader=dataloader, num=num_samples)

    specific_timesteps = torch.linspace(0, diffusion.num_diffusion_timesteps - 1, steps_to_vis, dtype=torch.long)

    noisy_samples: list[torch.Tensor] = []

    for timestep in specific_timesteps:
        ts = torch.as_tensor(timestep, dtype=torch.long)
        xts, _ = diffusion.forward_diffusion(frames_gt, ts)
        noisy_samples.append(xts.to("cpu"))

    frames_gt_vis = [prepare_vis_frames(frames_gt, nrow=1, data_type=data_type) for frames_gt in noisy_samples]

    figsize = (min(max(int(num_samples * 2), 12), 30), min(max(num_samples, 6), 15))
    fig, ax = plt.subplots(1, len(frames_gt_vis), figsize=figsize, facecolor="white")
    for i, (timestep, sample) in enumerate(zip(specific_timesteps, frames_gt_vis)):
        ax[i].imshow(sample)
        ax[i].set_title(f"t={timestep}", fontsize=8)
        ax[i].axis("off")
        ax[i].grid(False)
    fig.suptitle("Forward Diffusion Process", y=0.98)
    fig.tight_layout()
    os.makedirs(eval_examples_dir, exist_ok=True)
    fig.savefig(fname=os.path.join(eval_examples_dir, "frames_0_forward_diff.png"))
    plt.close()


def _debug_tensor(t, name=""):
    print(
        f"{name}:",
        t.shape,
        t.min(),
        t.max(),
        t.mean(),
        torch.isnan(t.reshape(-1)).sum().item(),
    )
