import hydra
import torch
from ddpm.data_generator import get_dataloader
from ddpm.diffusion import GaussianDiffusion
from ddpm.utils import log_forward_diffusion_examples
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="./../config/")
def main(cfg: DictConfig) -> None:
    print(f"---------- Config ----------\n{OmegaConf.to_yaml(cfg)}------------------------------")
    dtype = torch.bfloat16 if cfg.experiment.dtype == "bfloat16" else torch.float32

    # get a batch of samples
    dataloader, _ = get_dataloader(
        dataset_name=cfg.dataset.name,
        directory=cfg.dataset.directory,
        data_shape=cfg.experiment.data_shape,
        batch_size=cfg.experiment.batch_size,
        shuffle=cfg.dataset.shuffle,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )

    diffusion = GaussianDiffusion(
        num_diffusion_timesteps=cfg.diffusion.num_steps,
        schedule=cfg.diffusion.noise_schedule,
        beta_1=cfg.diffusion.beta_1,
        beta_T=cfg.diffusion.beta_T,
        device="cpu",
        dtype=dtype,
    )

    log_forward_diffusion_examples(
        dataloader=dataloader,
        diffusion=diffusion,
        num_samples=10,
        eval_examples_dir=cfg.experiment.eval_dir,
        steps_to_vis=15,
        data_type=cfg.experiment.data_type,
    )


if __name__ == "__main__":
    main()
