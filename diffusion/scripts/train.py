import hydra
import torch
from ddpm.data_generator import get_dataloader
from ddpm.diffusion import GaussianDiffusion
from ddpm.experiment import DiffusionExperiment
from ddpm.model import UNet
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary


@hydra.main(config_path="./../config/")
def main(cfg: DictConfig) -> None:
    print(f"---------- Config ----------\n{OmegaConf.to_yaml(cfg)}------------------------------")
    dtype = torch.bfloat16 if cfg.experiment.dtype == "bfloat16" else torch.float32

    # get a batch of samples
    dataloader, num_classes = get_dataloader(
        dataset_name=cfg.dataset.name,
        directory=cfg.dataset.directory,
        batch_size=cfg.experiment.batch_size,
        shuffle=cfg.dataset.shuffle,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )

    model = UNet(
        in_channels=cfg.experiment.data_shape[0],
        image_size=cfg.experiment.data_shape[-1],
        hidden_dims=cfg.model.unet_hidden_dims,
        num_classes=num_classes if cfg.model.class_cond else 1,
        dropout=0.05,
    )
    summary(
        model,
        input_size=(
            (cfg.experiment.batch_size, *cfg.experiment.data_shape),
            (cfg.experiment.batch_size,),
            (cfg.experiment.batch_size,),
        ),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.experiment.optim_lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.experiment.num_epochs,
        eta_min=cfg.experiment.optim_lr * cfg.experiment.lr_schedule_mult,
        last_epoch=-1,
    )

    diffusion = GaussianDiffusion(
        num_diffusion_timesteps=cfg.diffusion.num_steps,
        schedule=cfg.diffusion.noise_schedule,
        device=cfg.experiment.device,
        dtype=dtype,
    )

    loss_fn = torch.nn.MSELoss()

    diffusion_model = DiffusionExperiment(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        diffusion=diffusion,
        loss_fn=loss_fn,
        data_shape=cfg.experiment.data_shape,
        num_classes=num_classes,
        device=cfg.experiment.device,
        dtype=dtype,
        ema_decay=cfg.model.ema_decay,
    )

    diffusion_model.train(
        dataloader=dataloader,
        num_epochs=cfg.experiment.num_epochs,
        eval_interval=cfg.experiment.eval_interval,
        eval_dir=cfg.experiment.eval_dir,
        checkpoints_dir=cfg.experiment.checkpoints_dir,
    )


if __name__ == "__main__":
    main()
