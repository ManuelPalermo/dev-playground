import os
import shutil
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from ddpm.utils import prepare_fid_frames, prepare_vis_frames
from moviepy.editor import ImageSequenceClip
from torcheval.metrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm


class SimpleDiffusion:
    def __init__(
        self,
        num_diffusion_timesteps=1000,
        data_shape=(3, 64, 64),
        device="cpu",
        dtype=torch.float32,
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.data_shape = data_shape
        self.device = device
        self.dtype = dtype
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
            dtype=self.dtype,
            device=self.device,
        )

    def forward_diffusion(self, x0: torch.Tensor, timesteps: torch.Tensor):
        # some shape magic to deal with one or vector of timesteps
        B, _, *rest = x0.shape
        view_shape = (B, 1, *[1 for _ in rest])
        ts = timesteps.to(dtype=torch.long)
        if len(timesteps.shape) == 0:
            ts = ts.tile(B)

        eps = torch.randn_like(x0)  # Noise

        mean = self.sqrt_alpha_cumulative[ts].view(view_shape).expand_as(x0) * x0
        std_dev = (
            self.sqrt_one_minus_alpha_cumulative[ts].view(view_shape).expand_as(x0)
        )
        sample = mean + std_dev * eps  # scaled inputs * scaled noise
        return sample, eps


class SimpleDiffusionExperiment:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        sd: SimpleDiffusion,
        loss_fn: torch.nn.Module,
        data_shape: tuple[int, ...],
        num_classes: int,
        device,
        dtype,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.sd = sd
        self.loss_fn = loss_fn
        self.data_shape = data_shape
        self.num_classes = num_classes
        self.device = device
        self.dtype = dtype

        self.timesteps = sd.num_diffusion_timesteps

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int,
        eval_interval: Optional[int] = 10,
        eval_dir: str = "./outputs/reverse_diffusion",
        checkpoints_dir: str = "./outputs/checkpoints",
    ):
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)

        epochs_loss_list: list[float] = []
        fid_score_list: list[float] = []
        for epoch in range(num_epochs):
            # Train
            tq = tqdm(total=len(dataloader))
            tq.set_description(f"Train :: Epoch: {epoch}/{num_epochs}")
            epochs_loss = self.train_epoch(dataloader=dataloader)
            tq.set_postfix_str(s=f"Epoch Loss: {epochs_loss:.4f}")

            # Eval example logging
            fid_score = np.nan
            if eval_interval is not None and epoch % eval_interval == 0:
                # create grid of example reverse diffusion steps
                fid_score = self.eval_epoch(
                    epoch=epoch,
                    dataloader=dataloader,
                    num_images=100,
                    eval_examples_dir=eval_dir,
                    calculate_fid=True,
                )

                # save model checkpoint
                checkpoint_dict = {
                    "model": self.model.state_dict(),
                    "opt": self.optimizer.state_dict(),
                }
                torch.save(
                    checkpoint_dict,
                    os.path.join(
                        checkpoints_dir, f"ckpt_{epoch}_{epochs_loss:.03f}.pt"
                    ),
                )

            epochs_loss_list.append(epochs_loss)
            fid_score_list.append(fid_score)

            # log losses
            plt.figure(figsize=(8, 5))
            plt.plot(
                list(range(0, len(epochs_loss_list))),
                epochs_loss_list,
                label="Training Loss",
            )
            plt.savefig(os.path.join(eval_dir, "Plot_loss.png"))

            mask = np.isfinite(fid_score_list)
            plt.figure(figsize=(8, 5))
            plt.plot(
                np.arange(0, len(fid_score_list))[mask],
                np.array(fid_score_list)[mask],
                label="Eval FID",
            )
            plt.savefig(os.path.join(eval_dir, "Plot_FID.png"))

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> float:
        self.model.train()
        self.model.to(device=self.device, dtype=self.dtype)

        epoch_losses: list[float] = []
        for x0s, cls in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True):
            x0s = x0s.to(device=self.device, dtype=self.dtype)
            cls = cls.to(device=self.device, dtype=self.dtype)

            # forward diffusion (gt generation at random T steps)
            ts = torch.randint(
                low=1,
                high=self.timesteps,
                size=(x0s.shape[0],),
                device=self.device,
                dtype=self.dtype,
            )
            xts, gt_noise = self.sd.forward_diffusion(x0s, ts)

            # predict noise at given timestep (and optionaly class conditioning) and calculate loss
            pred_noise = self.model(xts, ts, cls)
            loss = self.loss_fn(gt_noise, pred_noise)

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # log loss
            loss_value = loss.detach().item()
            epoch_losses.append(loss_value)

        self.lr_scheduler.step()
        epoch_loss = float(sum(epoch_losses) / len(epoch_losses))
        return epoch_loss

    @torch.inference_mode()
    def eval_epoch(
        self,
        epoch: int,
        dataloader,
        num_images: int = 32,
        eval_examples_dir: Optional[str] = None,
        calculate_fid: bool = True,
    ) -> float:
        # generate frames through reverse diffusion
        # frames: timesteps[Tensor[B,C,H,W]]
        frames_gen_steps = self.reverse_diffusion(
            img_shape=self.data_shape,
            num_images=num_images,
        )

        frames_gen = frames_gen_steps[-1]

        fid_score = np.nan
        is_rgb = frames_gen.shape[1] == 3
        if calculate_fid and is_rgb:
            # calculate FID metric from frames (gen + gt)
            frames_gt = []
            n_to_sample = np.ceil(len(frames_gen) / dataloader.batch_size)
            for idx, (x0s, _) in enumerate(dataloader):
                x0s = x0s.to(device="cpu", dtype=self.dtype)
                frames_gt.append(x0s)
                if idx >= n_to_sample:
                    break
            frames_gt = torch.cat(frames_gt, dim=0)[: len(frames_gen)]

            fid_metric = FrechetInceptionDistance(device="cpu")
            fid_metric.update(
                images=prepare_fid_frames(frames_gt).to("cpu"), is_real=True
            )
            fid_metric.update(
                images=prepare_fid_frames(frames_gen).to("cpu"), is_real=False
            )
            fid_score = float(fid_metric.compute().cpu().numpy())

        if eval_examples_dir is not None:
            # prepare genered frames to be visualized
            frames_gen_vis = [
                prepare_vis_frames(frames_gen) for frames_gen in frames_gen_steps
            ]

            # save final denoised frame
            plt.imsave(
                fname=os.path.join(eval_examples_dir, f"frames_{epoch}.png"),
                arr=frames_gen_vis[-1],
            )
            # TODO: add visu with sampled timesteps of noise to last image (similar to existing for diffusion process)

            # save gif with reverse diffusion
            gif_name = f"frame_{epoch}.gif"
            clip = ImageSequenceClip(frames_gen_vis, fps=self.timesteps).speedx(10.0)
            clip.write_gif(gif_name)
            shutil.move(src=gif_name, dst=os.path.join(eval_examples_dir, gif_name))

        return fid_score

    @torch.inference_mode()
    def reverse_diffusion(
        self,
        img_shape: tuple[int, ...] = (3, 32, 32),
        num_images: int = 32,
    ) -> list[torch.Tensor]:
        self.model.eval()
        self.model.to(device=self.device, dtype=self.dtype)

        # generate random noise
        x = torch.randn((num_images, *img_shape), device=self.device, dtype=self.dtype)

        outs: list[torch.Tensor] = []

        for time_step in tqdm(
            iterable=reversed(range(1, self.timesteps)),
            total=self.timesteps - 1,
            dynamic_ncols=False,
            desc=f"Generating (n:{num_images}):: ",
            position=0,
        ):
            ts = (
                torch.ones(num_images, dtype=torch.long, device=self.device) * time_step
            )
            z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

            # try to condition generation of equaly number of samples for each class
            cls = torch.arange(self.num_classes, dtype=torch.long, device=self.device)
            cls = cls.repeat_interleave((num_images // self.num_classes))
            cls = torch.nn.functional.pad(cls, pad=(0, num_images - len(cls)))

            predicted_noise = self.model(x, ts, cls)
            predicted_noise = torch.nan_to_num(predicted_noise)

            B, *rest = predicted_noise.shape
            view_shape = (B, *[1 for _ in rest])
            beta_t = self.sd.beta[ts].view(view_shape).expand_as(predicted_noise)
            one_by_sqrt_alpha_t = (
                self.sd.one_by_sqrt_alpha[ts]
                .view(view_shape)
                .expand_as(predicted_noise)
            )
            sqrt_one_minus_alpha_cumulative_t = (
                self.sd.sqrt_one_minus_alpha_cumulative[ts]
                .view(view_shape)
                .expand_as(predicted_noise)
            )

            # TODO: fix issue with num_images has to be =32, or else shape fail (prob some deeper issue?)
            x = (
                one_by_sqrt_alpha_t
                * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
                + torch.sqrt(beta_t) * z
            )

            x = torch.clamp(x, min=-1, max=1)
            outs.append(x)

        return outs
