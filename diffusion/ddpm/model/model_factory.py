import torch
from torchinfo import summary

from ddpm.model.model_2d.unet_attn import UNet
from ddpm.model.model_3d.point_e import PointDiffusionTransformer
from ddpm.model.model_3d.point_net import PointNet


def get_model(
    model_name: str,
    hidden_dims: tuple[int, ...],
    data_type: str,
    data_shape: tuple[int, ...],
    num_classes: int,
    class_cond: bool = True,
    dropout: float = 0.0,
    batch_size: int = 1,
    context_dim: int | None = None,
) -> torch.nn.Module:
    """Creates model based opn config, supports 2D image models as well as 3D pointcloud models."""
    # 2D models:
    if model_name == "UNet2d":
        assert data_type == "img"
        model = UNet(
            in_channels=data_shape[0],
            image_size=data_shape[-1],
            hidden_dims=hidden_dims,
            num_classes=num_classes if class_cond else 1,
            dropout=dropout,
        )

    # 3D models:
    elif model_name == "PointNet":
        assert data_type == "pcd"
        model = PointNet(
            input_dim=data_shape[0],
            context_dim=context_dim if context_dim else hidden_dims[0],
            hidden_dims=hidden_dims,
            class_dim=data_shape[0],
            num_classes=num_classes if class_cond else 1,
            residual=False,
        )

    elif model_name == "PointDiffusionTransformer":
        assert data_type == "pcd"
        model = PointDiffusionTransformer(
            input_channels=data_shape[0],
            output_channels=data_shape[0],
            n_ctx=context_dim if context_dim else hidden_dims[0],
            width=hidden_dims[0],
            layers=hidden_dims,
            heads=4,
            init_scale=0.25,
            num_classes=num_classes if class_cond else 1,
            time_token_cond=True,
        )

    else:
        raise NotImplementedError(f"Unknow model config: {model_name}")

    input_example_shape = (
        (batch_size, *data_shape),
        (batch_size,),
        (batch_size,),
    )

    summary(model, input_size=input_example_shape)
    return model
