import os
import shutil
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from ddpm.diffusion import GaussianDiffusion
from torcheval.metrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid

try:
    from moviepy.editor import ImageSequenceClip
except RuntimeError:
    os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
    from moviepy.editor import ImageSequenceClip


def inverse_transform(tensors, max_val: float = 255.0):
    """Convert tensors from [-1., 1.] to [0., max_val]"""
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * max_val


def prepare_fid_frames(frames: torch.Tensor) -> torch.Tensor:
    """Convert frames from [-1,1] to [0,1]."""
    return inverse_transform(torch.nan_to_num(frames), max_val=1.0).to(torch.float32)


def prepare_vis_frames(frames: torch.Tensor, nrow: Optional[int] = 10) -> np.ndarray:
    frames_gen_vis = inverse_transform(torch.nan_to_num(frames), max_val=255.0)

    if nrow is not None:
        frames_gen_vis = torch.permute(make_grid(frames_gen_vis, nrow=nrow), dims=(1, 2, 0))  # GW x GH x C
    else:
        frames_gen_vis = torch.permute(frames_gen_vis, dims=(0, 2, 3, 1))  # B x W x H x C

    return frames_gen_vis.to(torch.uint8).to("cpu").numpy()


def collect_n_samples_from_dataloader(dataloader, num: int) -> torch.Tensor:
    frames_gt = []
    n_to_sample = np.ceil(num / dataloader.batch_size)
    for idx, (x0s, _) in enumerate(dataloader):
        x0s = x0s.to(device="cpu", dtype=torch.float32)
        frames_gt.append(x0s)
        if idx >= n_to_sample:
            break
    return torch.cat(frames_gt, dim=0)[:num]


def compute_fid_metric(
    frames_gen: torch.Tensor,
    dataloader,
) -> float:
    """Calculate FID metric from generated frames + sampled gt from a dataloader."""

    is_rgb = frames_gen.shape[1] == 3
    if is_rgb:
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
) -> None:
    # prepare generated frames to be visualized
    frames_grid_gen_vis = [prepare_vis_frames(frames_gen, nrow=10) for frames_gen in frames_steps]

    # save final denoised frames
    plt.imsave(
        fname=os.path.join(eval_examples_dir, f"frames_{epoch}_final.png"),
        arr=frames_grid_gen_vis[-1],
    )

    # save snapshots of frames at different timesteps
    vis_tsteps = np.linspace(start=0, stop=timesteps - 1, num=15, dtype=np.int64)
    vis_images = torch.linspace(
        0,
        frames_steps[0].shape[0] - 1,
        min(num_classes, 10),
        dtype=torch.long,
        device="cpu",
    )
    frames_gen_vis = [
        prepare_vis_frames(frames_gen[vis_images, ...], nrow=1)
        for tstep, frames_gen in enumerate(frames_steps)
        if tstep in vis_tsteps
    ]

    fig, ax = plt.subplots(1, len(frames_gen_vis), figsize=(10, 5), facecolor="white")
    for i, (timestep, sample) in enumerate(zip(vis_tsteps[::-1], frames_gen_vis)):
        ax[i].imshow(sample)
        ax[i].set_title(f"t={timestep}", fontsize=8)
        ax[i].axis("off")
        ax[i].grid(False)
    fig.suptitle("Reverse Diffusion Process", y=0.98)
    fig.tight_layout()
    fig.savefig(fname=os.path.join(eval_examples_dir, f"frames_{epoch}_reverse_diff.png"))

    # save gif with reverse diffusion
    gif_name = f"frames_{epoch}_reverse_diff.gif"
    clip = ImageSequenceClip(frames_grid_gen_vis, fps=timesteps).speedx(10.0)
    clip.write_gif(gif_name)
    shutil.move(src=gif_name, dst=os.path.join(eval_examples_dir, gif_name))


def log_forward_diffusion_examples(
    dataloader,
    diffusion: GaussianDiffusion,
    num_samples: int,
    eval_examples_dir: str,
    num_diffusion_timesteps: int = 1000,
    steps_to_vis: int = 15,
) -> None:
    # get a batch of samples
    frames_gt = collect_n_samples_from_dataloader(dataloader=dataloader, num=num_samples)

    specific_timesteps = torch.linspace(0, num_diffusion_timesteps - 1, steps_to_vis, dtype=torch.long)

    noisy_images: list[torch.Tensor] = []

    for timestep in specific_timesteps:
        ts = torch.as_tensor(timestep, dtype=torch.long)
        xts, _ = diffusion.forward_diffusion(frames_gt, ts)

        noisy_images.append(xts.to("cpu"))

    frames_gt_vis = [prepare_vis_frames(frames_gt, nrow=1) for frames_gt in noisy_images]

    fig, ax = plt.subplots(1, len(frames_gt_vis), figsize=(10, 5), facecolor="white")
    for i, (timestep, sample) in enumerate(zip(specific_timesteps, frames_gt_vis)):
        ax[i].imshow(sample)
        ax[i].set_title(f"t={timestep}", fontsize=8)
        ax[i].axis("off")
        ax[i].grid(False)
    fig.suptitle("Forward Diffusion Process", y=0.98)
    fig.tight_layout()
    fig.savefig(fname=os.path.join(eval_examples_dir, f"frames_0_forward_diff.png"))


def _debug_tensor(t, name=""):
    print(
        f"{name}:",
        t.shape,
        t.min(),
        t.max(),
        t.mean(),
        torch.isnan(t.view(-1)).sum().item(),
    )
