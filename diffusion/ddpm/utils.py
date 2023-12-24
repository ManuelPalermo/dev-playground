import numpy as np
import torch
from torchvision.utils import make_grid


def inverse_transform(tensors, max_val: float = 255.0):
    """Convert tensors from [-1., 1.] to [0., max_val]"""
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * max_val


def prepare_fid_frames(frames: torch.Tensor) -> torch.Tensor:
    """Convert frames from [-1,1] to [0,1]."""
    return inverse_transform(torch.nan_to_num(frames), max_val=1.0).to(torch.float32)


def prepare_vis_frames(frames: torch.Tensor, nrow: int = 10) -> np.ndarray:
    frames_gen_vis = (
        (
            torch.permute(
                make_grid(
                    inverse_transform(torch.nan_to_num(frames), max_val=255.0),
                    nrow=nrow,
                ),
                dims=(1, 2, 0),
            )
        )
        .to(torch.uint8)
        .to("cpu")
        .numpy()
    )

    return frames_gen_vis


def _debug_tensor(t, name=""):
    print(
        f"{name}:",
        t.shape,
        t.min(),
        t.max(),
        t.mean(),
        torch.isnan(t.view(-1)).sum().item(),
    )
