import matplotlib.pyplot as plt
import torch
from ddpm.data_generator import get_dataloader
from ddpm.diffusion import GaussianDiffusion
from ddpm.utils import inverse_transform
from torchvision.utils import make_grid


def visualize_forward_diffusion(
    dataset_name: str = "custom",
    directory: str = "./data",
    num_diffusion_timesteps: int = 500,
    steps_to_vis: int = 15,
):
    # get a batch of samples
    x0s = next(
        iter(
            get_dataloader(
                dataset_name=dataset_name,
                directory=directory,
                batch_size=16,
                shuffle=True,
            )[0]
        )
    )[0]

    diffusion = GaussianDiffusion(
        num_diffusion_timesteps=num_diffusion_timesteps, device="cpu"
    )

    noisy_images = []
    specific_timesteps = torch.linspace(
        0, num_diffusion_timesteps - 1, steps_to_vis, dtype=torch.long
    )

    for timestep in specific_timesteps:
        timestep = torch.as_tensor(timestep, dtype=torch.long)

        xts, _ = diffusion.forward_diffusion(x0s, timestep)
        xts = inverse_transform(xts, max_val=1.0)
        xts = make_grid(xts, nrow=1, padding=1)

        noisy_images.append(xts)

    _, ax = plt.subplots(1, len(noisy_images), figsize=(10, 5), facecolor="white")

    for i, (timestep, noisy_sample) in enumerate(zip(specific_timesteps, noisy_images)):
        ax[i].imshow(noisy_sample.squeeze(0).permute(1, 2, 0))
        ax[i].set_title(f"t={timestep}", fontsize=8)
        ax[i].axis("off")
        ax[i].grid(False)

    plt.suptitle("Forward Diffusion Process", y=0.9)
    plt.axis("off")
    plt.show()


def main(dataset_name, directory="./data/"):
    visualize_forward_diffusion(
        dataset_name=dataset_name,
        directory=directory,
    )


if __name__ == "__main__":
    main(dataset_name="Cifar-10", directory="./data/")
