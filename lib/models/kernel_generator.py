import torch
import torch.nn as nn
import torch.nn.functional as F

def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Conv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        norm (nn.Module, optional): a normalization layer
        activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        super().__init__()
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)

        self.conv = nn.Conv2d(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # if x.numel() == 0:
        #     output_shape = [
        #         (i + 2 * p - (di * (k - 1) + 1)) // s + 1
        #         for i, p, di, k, s in zip(
        #             x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
        #         )
        #     ]
        #     output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        #     empty = _NewEmptyTensorOp.apply(x, output_shape)
        #     if self.training:
        #         assert not isinstance(
        #             self.norm, torch.nn.SyncBatchNorm
        #         ), "SyncBatchNorm does not support empty inputs!"
        #         _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
        #         return empty + _dummy
        #     else:
        #         return empty

        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class ContextKernelPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ContextKernelPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = super(ContextKernelPooling, self).forward(x)
        return x

class SingleHead(nn.Module):
    """
    Build single head with convolutions and coord conv.
    """

    def __init__(self, in_channel, conv_dims, num_convs, coord=False, norm=True):
        super().__init__()
        self.coord = coord
        conv_norm_relus = []
        for k in range(num_convs):
            if coord:
                in_channel += 2
            conv_norm_relus.append(nn.Conv2d(
                in_channel if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ))
            if norm:
                conv_norm_relus.append(nn.BatchNorm2d(conv_dims))
            conv_norm_relus.append(nn.ReLU())
        self.conv_norm_relus = nn.Sequential(*conv_norm_relus)

    def forward(self, x):
        if self.coord:
            x = self.coord_conv(x)
        x = self.conv_norm_relus(x)
        return x

    def coord_conv(self, feat):
        with torch.no_grad():
            x_pos = torch.linspace(-1, 1, feat.shape[-2])
            y_pos = torch.linspace(-1, 1, feat.shape[-1])
            grid_x, grid_y = torch.meshgrid(x_pos, y_pos)
            grid_x = grid_x.unsqueeze(0).unsqueeze(1).expand(feat.shape[0], -1, -1, -1).to(feat.device)
            grid_y = grid_y.unsqueeze(0).unsqueeze(1).expand(feat.shape[0], -1, -1, -1).to(feat.device)
        feat = torch.cat([feat, grid_x, grid_y], dim=1)
        return feat
