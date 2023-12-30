import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleList


class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()

        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        x = self._layer(x) * gate + bias
        return x


class PointwiseNet(Module):
    def __init__(self, point_dim, hidden_dims, context_dim, num_classes, class_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual

        ctx_feats = context_dim + 3 + class_dim  # context_embed + time_embed + class_embed
        self.layers = ModuleList(
            [
                ConcatSquashLinear(point_dim, hidden_dims[0], ctx_feats),
                *[
                    ConcatSquashLinear(hidden_dims[h], hidden_dims[h + 1], ctx_feats)
                    for h in range(len(hidden_dims) - 1)
                ],
                ConcatSquashLinear(hidden_dims[-1], point_dim, ctx_feats),
            ]
        )

        self.num_classes = num_classes
        self.class_embedding = torch.nn.Embedding(num_embeddings=num_classes, embedding_dim=class_dim)

    def forward(self, x, ctx, ts, cls):
        """Forward.

        Args:
            x:   Point clouds at some timestep ts, (B, d, N).
            ctx: Context (B, F).
            ts:  Time. (B, ).
            cls: Class. (B, ).
        """

        batch_size = x.size(0)

        ctx = ctx[:, None, :]  # (B, 1, F)
        ts = ts.view(batch_size, 1, 1)  # (B, 1, 1)

        ts_emb = torch.cat([ts, torch.sin(ts), torch.cos(ts)], dim=-1)  # (B, 1, 3)

        if self.num_classes > 1:
            cls = cls.to(dtype=torch.long, device=x.device)
            cls_emb = self.class_embedding(cls).to(dtype=ts_emb.dtype)[..., None, :]  # (B, 1, C)
            ctx = torch.cat([ctx, ts_emb, cls_emb], dim=-1)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            out = x + out

        return out


class PointNetEncoder(torch.nn.Module):
    def __init__(self, zdim, input_dim=3):
        super().__init__()
        self.zdim = zdim
        self.conv1 = torch.nn.Conv1d(input_dim, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.bn4 = torch.nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = torch.nn.Linear(512, 256)
        self.fc2_m = torch.nn.Linear(256, 128)
        self.fc3_m = torch.nn.Linear(128, zdim)
        self.fc_bn1_m = torch.nn.BatchNorm1d(256)
        self.fc_bn2_m = torch.nn.BatchNorm1d(128)

        # Mapping to [v], cvar
        self.fc1_v = torch.nn.Linear(512, 256)
        self.fc2_v = torch.nn.Linear(256, 128)
        self.fc3_v = torch.nn.Linear(128, zdim)
        self.fc_bn1_v = torch.nn.BatchNorm1d(256)
        self.fc_bn2_v = torch.nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        # v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        # v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        # v = self.fc3_v(v)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m  # , v


class PointNet(Module):
    def __init__(self, context_dim: int, input_dim: int, hidden_dims, num_classes, class_dim, residual):
        super().__init__()
        self.encoder = PointNetEncoder(zdim=context_dim, input_dim=input_dim)
        self.pointwise = PointwiseNet(
            point_dim=input_dim,
            hidden_dims=hidden_dims,
            context_dim=context_dim,
            num_classes=num_classes,
            class_dim=class_dim,
            residual=residual,
        )

    def forward(self, x, ts, cls):
        """Forward.

        Args:
            x:   Point clouds at some timestep ts, (B, N, d).
            ts:  Time. (B, ).
            cls: Class. (B, ).
        """

        x = x.permute((0, 2, 1))  # (B, D, N) -> (B, N, D)

        ctx = self.encoder(x)
        out = self.pointwise(x, ctx, ts, cls)

        return out.permute((0, 2, 1))
