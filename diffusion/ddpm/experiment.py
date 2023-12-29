import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from ddpm.diffusion import GaussianDiffusion
from ddpm.utils import compute_fid_metric, log_forward_diffusion_examples, log_generation_examples
from tqdm import tqdm


class DiffusionExperiment:
    """DDMP experiment, supports training and generation of samples."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        diffusion: GaussianDiffusion,
        loss_fn: torch.nn.Module,
        data_type: str,
        data_shape: tuple[int, ...],
        num_classes: int,
        device,
        dtype,
        ema_decay: Optional[bool],
        torch_compile: bool = False,  # giving some issues
    ):
        """Constructor."""
        if torch_compile:
            torch.backends.cudnn.benchmark = True
            torch._dynamo.reset()
            model = torch.compile(model)

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.diffusion = diffusion
        self.loss_fn = loss_fn
        self.data_type = data_type
        self.data_shape = data_shape
        self.num_classes = num_classes
        self.device = device
        self.dtype = dtype
        self.ema_decay = ema_decay

        self.timesteps = diffusion.num_diffusion_timesteps

        self.ema_model = (
            torch.optim.swa_utils.AveragedModel(
                model,
                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay),
            )
            if ema_decay is not None
            else self.model
        )

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int,
        eval_interval: Optional[int] = 10,
        eval_num: int = 100,
        eval_dir: str = "./outputs/reverse_diffusion",
        checkpoints_dir: str = "./outputs/checkpoints",
    ):
        """Train loop."""
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)

        # log some forward diffusion samples from the dataloader
        log_forward_diffusion_examples(
            dataloader=dataloader,
            diffusion=self.diffusion,
            num_samples=10,
            eval_examples_dir=eval_dir,
            steps_to_vis=15,
            data_type=self.data_type,
        )

        epochs_loss_list: list[float] = []
        fid_score_list: list[float] = []
        for epoch in range(num_epochs):
            # Train
            tq = tqdm(total=len(dataloader), desc=f"Train :: Epoch: {epoch}/{num_epochs-1}")
            epochs_loss = self.train_epoch(dataloader=dataloader)
            tq.set_postfix_str(s=f"Epoch Loss: {epochs_loss:.4f}")

            # Eval example logging
            fid_score = np.nan
            if eval_interval is not None and epoch % eval_interval == 0:
                # save model checkpoints (current model and EMA model)
                checkpoint_dict = {
                    "model": self.model.state_dict(),
                    "opt": self.optimizer.state_dict(),
                }
                if self.ema_decay is not None:
                    # torch.optim.swa_utils.update_bn(dataloader, self.ema_model, device=self.device)
                    checkpoint_dict["model_ema"] = self.ema_model.state_dict()

                torch.save(
                    checkpoint_dict,
                    os.path.join(checkpoints_dir, f"ckpt_{epoch}_{epochs_loss:.03f}.pt"),
                )

                # create grid of example reverse diffusion steps
                frames_steps_gen = self.reverse_diffusion(
                    model=self.ema_model,  # use EMA model for generation
                    data_shape=self.data_shape,
                    num=eval_num,
                )

                # try to calculate fid score
                fid_score = compute_fid_metric(frames_gen=frames_steps_gen[-1], dataloader=dataloader)

                # log some generated samples
                log_generation_examples(
                    frames_steps=frames_steps_gen,
                    epoch=epoch,
                    eval_examples_dir=eval_dir,
                    timesteps=self.timesteps,
                    num_classes=self.num_classes,
                    data_type=self.data_type,
                )

            epochs_loss_list.append(epochs_loss)
            fid_score_list.append(fid_score)

            # log losses
            self.log_losses_metrics(
                eval_dir=eval_dir,
                epochs_loss_list=epochs_loss_list,
                fid_score_list=fid_score_list,
            )

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> float:
        """Train epoch."""
        self.model.train()
        self.model.to(device=self.device, dtype=self.dtype)
        if self.ema_decay is not None:
            self.ema_model.train()
            self.ema_model.to(device=self.device, dtype=self.dtype)

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
            loss = self.train_step(x0s, ts, cls)

            # log loss
            epoch_losses.append(loss.detach().item())

        self.lr_scheduler.step()
        epoch_loss = float(sum(epoch_losses) / len(epoch_losses))
        return epoch_loss

    def train_step(self, x0s, ts, cls) -> torch.Tensor:
        """Train step."""
        # forward diffusion (gt generation at random T steps)
        xts, gt_noise = self.diffusion.forward_diffusion(x0s, ts)

        # predict noise at given timestep (and optionaly class conditioning) and calculate loss
        pred_noise = self.model(xts, ts, cls)
        loss = self.loss_fn(gt_noise, pred_noise)

        # take gradient step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.ema_decay is not None:
            self.ema_model.update_parameters(self.model)

        return loss

    @torch.inference_mode()
    def reverse_diffusion(
        self,
        model: torch.nn.Module,
        data_shape: tuple[int, ...] = (3, 32, 32),
        num: int = 32,
    ) -> list[torch.Tensor]:
        """Performs reverse diffusion."""
        model.eval()
        model.to(device=self.device, dtype=self.dtype)

        # generate initial random noise
        x = torch.randn((num, *data_shape), device=self.device, dtype=self.dtype)

        outs: list[torch.Tensor] = []

        for time_step in tqdm(
            iterable=reversed(range(0, self.timesteps)),
            total=self.timesteps,
            dynamic_ncols=False,
            desc=f"Generating (n:{num}):: ",
            position=0,
        ):
            ts = torch.ones(num, dtype=torch.long, device=self.device) * time_step
            z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

            # try to condition generation of equaly number of samples for each class
            cls = torch.arange(self.num_classes, dtype=torch.long, device=self.device)
            cls = cls.repeat_interleave((num // self.num_classes))
            cls = torch.nn.functional.pad(cls, pad=(0, num - len(cls)))

            predicted_noise = model(x, ts, cls)
            predicted_noise = torch.nan_to_num(predicted_noise)

            B, *rest = predicted_noise.shape
            view_shape = (B, *[1 for _ in rest])
            beta_t = self.diffusion.beta[ts].view(view_shape).expand_as(predicted_noise)
            one_by_sqrt_alpha_t = self.diffusion.one_by_sqrt_alpha[ts].view(view_shape).expand_as(predicted_noise)
            sqrt_one_minus_alpha_cumulative_t = (
                self.diffusion.sqrt_one_minus_alpha_cumulative[ts].view(view_shape).expand_as(predicted_noise)
            )

            x = (
                one_by_sqrt_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
                + torch.sqrt(beta_t) * z
            )

            x = torch.clamp(x, min=-1, max=1)
            outs.append(x.detach().to("cpu"))

        return outs

    @staticmethod
    def log_losses_metrics(
        eval_dir: str,
        epochs_loss_list: list[float],
        fid_score_list: list[float],
    ) -> None:
        """Logs losses and metrics."""
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
        plt.close()
