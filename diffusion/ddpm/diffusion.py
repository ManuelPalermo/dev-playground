import os
from typing import Optional

from tqdm import tqdm

import numpy as np
import torch

from moviepy.editor import ImageSequenceClip

from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from ddpm.data_generator import inverse_transform


class SimpleDiffusion:
    def __init__(
        self,
        num_diffusion_timesteps=1000,
        data_shape=(3, 64, 64),
        device="cpu",
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.data_shape = data_shape
        self.device = device
        self.initialize()

    def initialize(self):
        # BETAs & ALPHAs required at different places in the Algorithm.
        self.beta = self.get_betas()
        self.alpha = 1 - self.beta

        self.sqrt_beta = torch.sqrt(self.beta)
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha = 1.0 / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)

    def get_betas(self):
        """Linear schedule, proposed in original ddpm paper"""
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            start=beta_start,
            end=beta_end,
            steps=self.num_diffusion_timesteps,
            dtype=torch.float32,
            device=self.device,
        )

    def forward_diffusion(self, x0: torch.Tensor, timesteps: torch.Tensor):
        B, C, *rest = x0.shape
        view_shape = (B, C, *[1 for _ in rest])
        eps = torch.randn_like(x0)  # Noise
        mean = self.sqrt_alpha_cumulative[timesteps].view(view_shape) * x0
        std_dev = self.sqrt_one_minus_alpha_cumulative[timesteps].view(view_shape)
        sample = mean + std_dev * eps  # scaled inputs * scaled noise
        return sample, eps


class SimpleDiffusionModel:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        sd: SimpleDiffusion,
        loss_fn: torch.nn.Module,
        data_shape: tuple[int, ...],
        device: str,
    ):

        self.model = model
        self.optimizer = optimizer
        self.sd = sd
        self.loss_fn = loss_fn
        self.data_shape = data_shape
        self.device = device

        self.timesteps = sd.num_diffusion_timesteps

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int,
        eval_interval: Optional[int] = 10,
        eval_dir: str = "./outputs/reverse_diffusion",
        checkpoints_dir: str = "./outputs/checkpoints",
    ):

        self.model.train()
        self.model.to(device=self.device)

        for epoch in range(num_epochs):

            # Train
            tq = tqdm(total=len(dataloader))
            tq.set_description(f"Train :: Epoch: {epoch}/{num_epochs}")
            epochs_loss = self.train_epoch(dataloader=dataloader)
            tq.set_postfix_str(s=f"Epoch Loss: {epochs_loss:.4f}")

            # Eval example logging
            if eval_interval is not None and epoch % eval_interval == 0:
                with torch.no_grad():
                    frames = self.reverse_diffusion(
                        img_shape=self.data_shape, num_images=32, nrow=8, generate_video=False
                    )
                    # save_path = os.path.join(eval_dir)
                    # os.makedirs(save_path, exist_ok=True)
                    ImageSequenceClip(frames, fps=self.timesteps // 25).write_gif(f"{epoch}.gif", fps=20)

            # save model checkpoint
            checkpoint_dict = {
                "model": self.model.state_dict(),
                "opt": self.optimizer.state_dict(),
            }
            os.makedirs(checkpoints_dir, exist_ok=True)
            torch.save(checkpoint_dict, os.path.join(checkpoints_dir, f"ckpt_{epoch}_{epochs_loss:.03f}.pt"))
            del checkpoint_dict

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> float:

        epoch_losses: list[float] = []
        self.model.train()

        for (x0s, _) in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True):

            x0s = x0s.to(device=self.device)

            # forward diffusion (gt generation at random T steps)
            ts = torch.randint(low=1, high=self.timesteps, size=(x0s.shape[0],), device=self.device)
            xts, gt_noise = self.sd.forward_diffusion(x0s, ts)

            # predict noise at given timestep and calculate loss
            pred_noise = self.model(xts, ts)
            loss = self.loss_fn(gt_noise, pred_noise)

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # log loss
            loss_value = loss.detach().item()
            epoch_losses.append(loss_value)

        epoch_loss = float(sum(epoch_losses) / len(epoch_losses))
        return epoch_loss

    @torch.no_grad()
    def reverse_diffusion(
        self,
        img_shape: tuple[int, ...] = (3, 64, 64),
        num_images: int = 5,
        nrow: int = 8,
        generate_video: bool = True,
    ) -> list[np.ndarray]:

        self.model.eval()
        self.model.to(device=self.device)

        outs: list[np.ndarray] = []

        # generate random noise
        x = torch.randn((num_images, *img_shape), device=self.device)

        for time_step in tqdm(
            iterable=reversed(range(1, self.timesteps)),
            total=self.timesteps - 1,
            dynamic_ncols=False,
            desc="Generating :: ",
            position=0,
        ):

            ts = torch.ones(num_images, dtype=torch.long, device=self.device) * time_step
            z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

            predicted_noise = self.model(x, ts)

            beta_t = self.sd.beta[ts]
            one_by_sqrt_alpha_t = self.sd.one_by_sqrt_alpha[ts]
            sqrt_one_minus_alpha_cumulative_t = self.sd.sqrt_one_minus_alpha_cumulative[ts]

            x = (
                one_by_sqrt_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
                + torch.sqrt(beta_t) * z
            )

            # prepare outputs for visu
            x_inv = inverse_transform(x).type(torch.uint8)
            grid = make_grid(x_inv, nrow=nrow, pad_value=255.0).to("cpu")
            ndarr = torch.permute(grid, (1, 2, 0)).numpy()[:, :, ::-1]
            outs.append(ndarr)

        return outs
