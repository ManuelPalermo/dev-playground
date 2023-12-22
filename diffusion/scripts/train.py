import torch

from torchinfo import summary

from ddpm.model import UNet
from ddpm.diffusion import SimpleDiffusion, SimpleDiffusionModel
from ddpm.data_generator import get_dataloader


def main(
    dataset_name: str,
    directory: str = "./data",
    num_diffusion_timesteps: int = 1000,
    batch_size: int = 64,
    optim_lr: float = 1e-3,
    data_shape: tuple[int, ...] = (1, 32, 32),
    device: str = "cuda",
):

    model = UNet(in_channels=data_shape[0], image_size=data_shape[-1], hidden_dims=[8, 16, 32, 64])
    summary(model, input_size=((batch_size, *data_shape), (batch_size,)))  # (sample, timestamp)

    optimizer = torch.optim.AdamW(model.parameters(), lr=optim_lr)

    # get a batch of samples
    dataloader = get_dataloader(
        dataset_name=dataset_name,
        directory=directory,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
    )

    sd = SimpleDiffusion(num_diffusion_timesteps=num_diffusion_timesteps, data_shape=data_shape, device=device)

    loss_fn = torch.nn.MSELoss()

    diffusion_model = SimpleDiffusionModel(
        model=model,
        optimizer=optimizer,
        sd=sd,
        loss_fn=loss_fn,
        data_shape=data_shape,
        device=device,
    )

    diffusion_model.train(dataloader=dataloader, num_epochs=20, eval_interval=1)


if __name__ == "__main__":
    main(dataset_name="MNIST", data_shape=(1, 32, 32))
