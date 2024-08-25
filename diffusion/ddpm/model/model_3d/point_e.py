import math
from typing import Callable, Iterable, Sequence, Union

import torch


def checkpoint(
    func: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor]]],
    inputs: Sequence[torch.Tensor],
    params: Iterable[torch.Tensor],
    flag: bool,
):
    """Evaluate a function without caching intermediate activations.

    Allows for reduced memory at the expense of extra compute in the backward pass.

    Args:
        func: the function to evaluate.
        inputs: the argument sequence to pass to `func`.
        params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
        flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def init_linear(layer, stddev):
    torch.nn.init.normal_(layer.weight, std=stddev)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, 0.0)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.

    Returns:
        an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class MultiheadAttention(torch.nn.Module):
    def __init__(
        self,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads

        self.c_qkv = torch.nn.Linear(width, width * 3)
        self.c_proj = torch.nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads=heads, n_ctx=n_ctx)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)  # TODO:check this
        x = self.c_proj(x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, width: int, init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = torch.nn.Linear(width, width * 4)
        self.c_proj = torch.nn.Linear(width * 4, width)
        self.gelu = torch.nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(torch.nn.Module):
    def __init__(self, heads: int, n_ctx: int):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        weight = torch.einsum("bthc,bshc->bhts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).to(qkv.dtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(torch.nn.Module):
    def __init__(
        self,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_1 = torch.nn.LayerNorm(width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_2 = torch.nn.LayerNorm(width)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        n_ctx: int,
        width: int,
        layers: Sequence[int],
        heads: int,
        init_scale: float = 0.25,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = torch.nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx=n_ctx,
                    width=w,
                    heads=heads,
                    init_scale=init_scale * math.sqrt(1.0 / w),
                )
                for w in layers
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class PointDiffusionTransformer(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        n_ctx: int = 1024,
        width: int = 512,
        layers: Sequence[int] = tuple(512 for _ in range(12)),
        heads: int = 8,
        init_scale: float = 0.25,
        num_classes: int = 1,
        time_token_cond: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx + width
        self.time_token_cond = time_token_cond
        self.num_classes = num_classes

        self.time_embed = MLP(width=width, init_scale=init_scale * math.sqrt(1.0 / width))
        self.class_embed = torch.nn.Embedding(num_embeddings=num_classes, embedding_dim=width)

        self.ln_pre = torch.nn.LayerNorm(width)
        self.backbone = Transformer(
            n_ctx=n_ctx + int(time_token_cond) + width,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_post = torch.nn.LayerNorm(width)
        self.input_proj = torch.nn.Linear(input_channels, width)
        self.output_proj = torch.nn.Linear(width, output_channels)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def forward(self, x: torch.Tensor, ts: torch.Tensor, cls: torch.Tensor):
        """Forward.

        Args:
            x: an [B x C x N] tensor.
            ts: an [B] tensor.
            cls: an [B] tensor.

        Returns:
            an [B x C' x N] tensor.
        """
        # assert x.shape[-1] == self.n_ctx

        t_emb = self.time_embed(timestep_embedding(ts, self.backbone.width))

        if self.num_classes > 1:
            # add class information to embeddings
            cls = cls.to(dtype=torch.long, device=t_emb.device)
            t_emb = t_emb + self.class_embed(cls).to(dtype=t_emb.dtype)

        return self._forward_with_cond(x, [(t_emb, self.time_token_cond)])

    def _forward_with_cond(self, x: torch.Tensor, cond_as_token: list[tuple[torch.Tensor, bool]]) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 2, 1))  # NCL -> NLC
        for emb, as_token in cond_as_token:
            if not as_token:
                h = h + emb[:, None]
        extra_tokens = [(emb[:, None] if len(emb.shape) == 2 else emb) for emb, as_token in cond_as_token if as_token]

        if len(extra_tokens):
            h = torch.cat(extra_tokens + [h], dim=1)

        h = self.ln_pre(h)
        h = self.backbone(h)
        h = self.ln_post(h)

        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens) :]
        h = self.output_proj(h)
        return h.permute(0, 2, 1)
