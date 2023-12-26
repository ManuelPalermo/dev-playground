import os

import hydra
import matplotlib.pyplot as plt
import torch
from ddpm.data_generator import get_dataloader
from ddpm.diffusion import GaussianDiffusion
from ddpm.utils import inverse_transform
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import make_grid


@hydra.main(config_path="./../config/")
def main(cfg: DictConfig) -> None:
    print(f"---------- Config ----------\n{OmegaConf.to_yaml(cfg)}------------------------------")
    dtype = torch.bfloat16 if cfg.experiment.dtype == "bfloat16" else torch.float32

    # get a batch of samples
    x0s = next(
        iter(
            get_dataloader(
                dataset_name=cfg.dataset.name,
                directory=cfg.dataset.directory,
                data_shape=cfg.experiment.data_shape,
                batch_size=cfg.experiment.batch_size,
                shuffle=cfg.dataset.shuffle,
                num_workers=cfg.dataset.num_workers,
                pin_memory=cfg.dataset.pin_memory,
            )[0]
        )
    )[0]

    diffusion = GaussianDiffusion(
        num_diffusion_timesteps=cfg.diffusion.num_steps,
        schedule=cfg.diffusion.noise_schedule,
        device="cpu",
        dtype=dtype,
    )

    noisy_images = []
    specific_timesteps = torch.linspace(0, cfg.diffusion.num_steps - 1, steps=15, dtype=torch.long)

    for timestep in specific_timesteps:
        timestep = torch.as_tensor(timestep, dtype=torch.long)

        xts, _ = diffusion.forward_diffusion(x0s, timestep)
        xts = inverse_transform(xts, max_val=1.0)
        xts = make_grid(xts, nrow=1, padding=1)
        noisy_images.append(xts)

    fig, ax = plt.subplots(1, len(noisy_images), figsize=(10, 5), facecolor="white")
    for i, (timestep, noisy_sample) in enumerate(zip(specific_timesteps, noisy_images)):
        ax[i].imshow(noisy_sample.squeeze(0).permute(1, 2, 0))
        ax[i].set_title(f"t={timestep}", fontsize=8)
        ax[i].axis("off")
        ax[i].grid(False)

    fig.suptitle("Forward Diffusion Process", y=0.98)
    fig.tight_layout()
    os.makedirs(cfg.experiment.eval_dir, exist_ok=True)
    fig.savefig(fname=os.path.join(cfg.experiment.eval_dir, "frames_forward_diff.png"))


if __name__ == "__main__":
    main()
