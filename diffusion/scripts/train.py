import torch
from ddpm.data_generator import get_dataloader
from ddpm.diffusion import SimpleDiffusion, SimpleDiffusionExperiment
from ddpm.model import UNet
from torchinfo import summary


def main(
    dataset_name: str,
    directory: str = "./data",
    conditional_gen: bool = True,
    num_diffusion_timesteps: int = 1000,
    num_epochs: int = 50,
    batch_size: int = 64,
    optim_lr: float = 2e-4,
    unet_hidden_dims: tuple[int, ...] = (32, 64, 128, 256),
    data_shape: tuple[int, ...] = (1, 32, 32),
    device: str = "cuda",
    dtype=torch.float32,  # NOTE: bfloat16 is possible, but results are not stable for some reason :(
):
    # get a batch of samples
    dataloader, num_classes = get_dataloader(
        dataset_name=dataset_name,
        directory=directory,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
    )

    model = UNet(
        in_channels=data_shape[0],
        image_size=data_shape[-1],
        hidden_dims=unet_hidden_dims,
        num_classes=num_classes if conditional_gen else 1,
    )
    summary(model, input_size=((batch_size, *data_shape), (batch_size,), (batch_size,)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=optim_lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=optim_lr * 0.01,
        last_epoch=-1,
    )

    sd = SimpleDiffusion(
        num_diffusion_timesteps=num_diffusion_timesteps,
        data_shape=data_shape,
        device=device,
        dtype=dtype,
    )

    loss_fn = torch.nn.MSELoss()

    diffusion_model = SimpleDiffusionExperiment(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        sd=sd,
        loss_fn=loss_fn,
        data_shape=data_shape,
        num_classes=num_classes,
        device=device,
        dtype=dtype,
    )

    diffusion_model.train(
        dataloader=dataloader,
        num_epochs=num_epochs,
        eval_interval=5,
        eval_dir=f"./outputs/{dataset_name}/reverse_diffusion",
        checkpoints_dir=f"./outputs/{dataset_name}/checkpoints",
    )


if __name__ == "__main__":
    main(
        dataset_name="Cifar-10",
        data_shape=(3, 32, 32),
        num_diffusion_timesteps=1000,
        num_epochs=1,
    )
    main(
        dataset_name="MNIST",
        data_shape=(1, 32, 32),
        num_diffusion_timesteps=500,
        num_epochs=51,
    )
