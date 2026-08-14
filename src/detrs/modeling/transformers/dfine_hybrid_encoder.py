"""D-FINE family hybrid encoder with an optional training-only F5 projection."""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.workspace import register, serializable
from ..shape_spec import ShapeSpec


def _activation(name):
    if name is None:
        return nn.Identity()
    if isinstance(name, nn.Module):
        return name
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "swish": nn.SiLU}[
        name.lower()
    ]()


class ConvNormLayer_fuse(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        kernel_size,
        stride,
        g=1,
        padding=None,
        bias=False,
        act=None,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride, padding=padding, groups=g, bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = _activation(act)
        self.ch_in, self.ch_out = ch_in, ch_out
        self.kernel_size, self.stride, self.g = kernel_size, stride, g
        self.padding = padding

    def forward(self, value):
        if hasattr(self, "conv_bn_fused"):
            return self.act(self.conv_bn_fused(value))
        return self.act(self.norm(self.conv(value)))

    def convert_to_deploy(self):
        if hasattr(self, "conv_bn_fused"):
            return
        fused = nn.Conv2d(
            self.ch_in,
            self.ch_out,
            self.kernel_size,
            self.stride,
            padding=self.padding,
            groups=self.g,
            bias=True,
        ).to(self.conv.weight)
        running_var = self.norm.running_var
        running_mean = self.norm.running_mean
        weight = self.norm.weight
        bias = self.norm.bias
        if any(value is None for value in (running_var, running_mean, weight, bias)):
            raise ValueError("deploy fusion requires affine BatchNorm running state")
        assert running_var is not None
        assert running_mean is not None
        assert weight is not None
        assert bias is not None
        std = (running_var + self.norm.eps).sqrt()
        scale = (weight / std).reshape(-1, 1, 1, 1)
        if fused.bias is None:
            raise RuntimeError("deploy fusion requires a convolution bias")
        with torch.no_grad():
            fused.weight.copy_(self.conv.weight * scale)
            fused.bias.copy_(bias - running_mean * weight / std)
        self.conv_bn_fused = fused
        del self.conv
        del self.norm


class ConvNormLayer(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        kernel_size,
        stride,
        g=1,
        padding=None,
        bias=False,
        act=None,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = _activation(act)

    def forward(self, value):
        return self.act(self.norm(self.conv(value)))


class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, value):
        return self.cv2(self.cv1(value))


class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.ch_in, self.ch_out = ch_in, ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0)
        self.act = _activation(act)

    def forward(self, value):
        if hasattr(self, "conv"):
            return self.act(self.conv(value))
        return self.act(self.conv1(value) + self.conv2(value))

    def convert_to_deploy(self):
        if hasattr(self, "conv"):
            return self

        def fuse(branch):
            running_var = branch.norm.running_var
            running_mean = branch.norm.running_mean
            weight = branch.norm.weight
            bias = branch.norm.bias
            if any(
                value is None for value in (running_var, running_mean, weight, bias)
            ):
                raise ValueError("deploy fusion requires affine BatchNorm state")
            assert running_var is not None
            assert running_mean is not None
            assert weight is not None
            assert bias is not None
            std = (running_var + branch.norm.eps).sqrt()
            scale = (weight / std).reshape(-1, 1, 1, 1)
            return (
                branch.conv.weight * scale,
                bias - running_mean * weight / std,
            )

        kernel3, bias3 = fuse(self.conv1)
        kernel1, bias1 = fuse(self.conv2)
        self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1).to(kernel3)
        if self.conv.bias is None:
            raise RuntimeError("deploy fusion requires a convolution bias")
        with torch.no_grad():
            self.conv.weight.copy_(kernel3 + F.pad(kernel1, [1, 1, 1, 1]))
            self.conv.bias.copy_(bias3 + bias1)
        del self.conv1
        del self.conv2
        return self


class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=3,
        expansion=1.0,
        bias=False,
        act="silu",
        bottletype=VGGBlock,
    ):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(
            *[
                bottletype(hidden, hidden, act=_activation(act))
                for _ in range(num_blocks)
            ]
        )
        self.conv3 = (
            ConvNormLayer_fuse(hidden, out_channels, 1, 1, bias=bias, act=act)
            if hidden != out_channels
            else nn.Identity()
        )

    def forward(self, value):
        return self.conv3(self.bottlenecks(self.conv1(value)) + self.conv2(value))


class RepNCSPELAN4(nn.Module):
    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu", csp_type="csp"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        csp_block = CSPLayer2 if csp_type == "csp2" else CSPLayer
        self.cv2 = nn.Sequential(
            csp_block(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            csp_block(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayer_fuse(c3 + 2 * c4, c2, 1, 1, bias=bias, act=act)

    def forward(self, value):
        outputs = list(self.cv1(value).split((self.c, self.c), 1))
        outputs.extend(module(outputs[-1]) for module in (self.cv2, self.cv3))
        return self.cv4(torch.cat(outputs, 1))


class CSPLayer2(nn.Module):
    """RepC3-style CSP block with a single chunked projection."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=3,
        expansion=1.0,
        bias=False,
        act="silu",
        bottletype=VGGBlock,
    ):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(
            in_channels, hidden * 2, 1, 1, bias=bias, act=act
        )
        self.bottlenecks = nn.Sequential(
            *[
                bottletype(hidden, hidden, act=_activation(act))
                for _ in range(num_blocks)
            ]
        )
        self.conv3 = (
            ConvNormLayer_fuse(hidden, out_channels, 1, 1, bias=bias, act=act)
            if hidden != out_channels
            else nn.Identity()
        )

    def forward(self, value):
        first, second = self.conv1(value).chunk(2, 1)
        return self.conv3(first + self.bottlenecks(second))


class RepNCSPELAN5(nn.Module):
    """DEIM encoder fuse block: RepNCSPELAN4 with plain CSPLayer2 branches."""

    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        # Upstream wraps each branch in a single-element Sequential; keep the
        # wrapper so checkpoint keys keep their .0 index.
        self.cv2 = nn.Sequential(
            CSPLayer2(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock)
        )
        self.cv3 = nn.Sequential(
            CSPLayer2(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock)
        )
        self.cv4 = ConvNormLayer_fuse(c3 + 2 * c4, c2, 1, 1, bias=bias, act=act)

    def forward(self, value):
        outputs = list(self.cv1(value).split((self.c, self.c), 1))
        outputs.extend(module(outputs[-1]) for module in (self.cv2, self.cv3))
        return self.cv4(torch.cat(outputs, 1))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _activation(activation)

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        query = src if pos_embed is None else src + pos_embed
        src = self.self_attn(query, query, src, attn_mask=src_mask)[0]
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        return src if self.normalize_before else self.norm2(src)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None):
        for layer in self.layers:
            src = layer(src, src_mask=src_mask, pos_embed=pos_embed)
        return self.norm(src) if self.norm is not None else src


@register
@serializable
class DFINEHybridEncoder(nn.Module):
    def __init__(
        self,
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=None,
        version="dfine",
        fuse_op="cat",
        csp_type="csp",
        distill_teacher_dim=None,
        project_f5=False,
    ):
        super().__init__()
        if len(in_channels) != len(feat_strides):
            raise ValueError("in_channels and feat_strides level count must match")
        if not use_encoder_idx or any(
            index >= len(in_channels) or index < 0 for index in use_encoder_idx
        ):
            raise ValueError(
                "use_encoder_idx must select at least one valid feature level"
            )
        if project_f5 and (distill_teacher_dim is None or distill_teacher_dim <= 0):
            raise ValueError(
                "distill_teacher_dim must be positive when project_f5 is enabled"
            )
        if distill_teacher_dim is not None and distill_teacher_dim <= 0:
            raise ValueError("distill_teacher_dim must be positive when provided")
        if distill_teacher_dim is not None and not project_f5:
            raise ValueError(
                "project_f5 must be enabled when distill_teacher_dim is provided"
            )
        if version not in ("dfine", "deim", "rt_detrv2"):
            raise ValueError("unsupported D-FINE family encoder version")
        if fuse_op not in ("cat", "sum"):
            raise ValueError("fuse_op must be 'cat' or 'sum'")
        # csp_type only parameterizes the dfine fuse blocks; the deim and
        # rt_detrv2 versions have fixed block layouts and ignore it.
        self.fuse_op = fuse_op
        self.csp_type = csp_type
        self.in_channels = list(in_channels)
        self.feat_strides = list(feat_strides)
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = list(use_encoder_idx)
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim] * len(in_channels)
        self.out_strides = list(feat_strides)
        self.encoder_idx_for_distillation = use_encoder_idx[-1]
        self.input_proj = nn.ModuleList(
            # The DEIMv2-era upstream encoder keeps nn.Identity when the
            # backbone already outputs hidden_dim channels; the pinned
            # D-FINE/RT-DETRv2 graphs always project (their official
            # checkpoints contain these weights).
            nn.Identity()
            if version == "deim" and channels == hidden_dim
            else nn.Sequential(
                OrderedDict(
                    [
                        ("conv", nn.Conv2d(channels, hidden_dim, 1, bias=False)),
                        ("norm", nn.BatchNorm2d(hidden_dim)),
                    ]
                )
            )
            for channels in in_channels
        )
        layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, enc_act
        )
        self.encoder = nn.ModuleList(
            TransformerEncoder(copy.deepcopy(layer), num_encoder_layers)
            for _ in use_encoder_idx
        )
        self.feature_projector = (
            nn.Sequential(nn.Linear(hidden_dim, distill_teacher_dim))
            if project_f5
            else None
        )
        # DEIM fuses by summation, keeping the fuse-block input at hidden_dim.
        input_dim = hidden_dim if fuse_op == "sum" else hidden_dim * 2

        def _fuse_block():
            if version == "rt_detrv2":
                return CSPLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    expansion=expansion,
                    act=act,
                )
            if version == "deim":
                return RepNCSPELAN5(
                    input_dim,
                    hidden_dim,
                    hidden_dim * 2,
                    round(expansion * hidden_dim // 2),
                    round(3 * depth_mult),
                )
            return RepNCSPELAN4(
                input_dim,
                hidden_dim,
                hidden_dim * 2,
                round(expansion * hidden_dim // 2),
                round(3 * depth_mult),
                csp_type=csp_type,
            )

        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                ConvNormLayer_fuse(
                    hidden_dim,
                    hidden_dim,
                    1,
                    1,
                    act=act if version == "rt_detrv2" else None,
                )
            )
            self.fpn_blocks.append(_fuse_block())
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 2, act=act)
                if version == "rt_detrv2"
                else nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2))
            )
            self.pan_blocks.append(_fuse_block())
        if eval_spatial_size:
            for index in use_encoder_idx:
                stride = feat_strides[index]
                self.register_buffer(
                    f"pos_embed{index}",
                    self.build_2d_sincos_position_embedding(
                        eval_spatial_size[1] // stride,
                        eval_spatial_size[0] // stride,
                        hidden_dim,
                        pe_temperature,
                    ),
                    persistent=False,
                )

    @staticmethod
    def build_2d_sincos_position_embedding(
        width, height, embed_dim=256, temperature=10000.0, device=None
    ):
        if embed_dim % 4:
            raise ValueError("embed_dim must be divisible by 4")
        grid_width = torch.arange(int(width), dtype=torch.float32, device=device)
        grid_height = torch.arange(int(height), dtype=torch.float32, device=device)
        grid_width, grid_height = torch.meshgrid(grid_width, grid_height, indexing="ij")
        omega = torch.arange(embed_dim // 4, dtype=torch.float32, device=device) / (
            embed_dim // 4
        )
        omega = 1 / temperature**omega
        width_embedding = grid_width.flatten()[:, None] @ omega[None]
        height_embedding = grid_height.flatten()[:, None] @ omega[None]
        return torch.cat(
            [
                width_embedding.sin(),
                width_embedding.cos(),
                height_embedding.sin(),
                height_embedding.cos(),
            ],
            1,
        )[None]

    def forward(self, feats):
        if len(feats) != len(self.in_channels):
            raise ValueError(
                f"expected {len(self.in_channels)} feature levels, got {len(feats)}"
            )
        projected = [module(feat) for module, feat in zip(self.input_proj, feats)]
        projected_f5 = None
        if self.num_encoder_layers > 0:
            for encoder, index in zip(self.encoder, self.use_encoder_idx):
                height, width = projected[index].shape[2:]
                flattened = projected[index].flatten(2).permute(0, 2, 1)
                position = getattr(self, f"pos_embed{index}", None)
                if self.training or position is None:
                    position = self.build_2d_sincos_position_embedding(
                        width,
                        height,
                        self.hidden_dim,
                        self.pe_temperature,
                        flattened.device,
                    )
                    if not self.training:
                        # Cache eval grids as buffers: torch.jit.trace freezes
                        # the device argument built here at trace time, while a
                        # buffer follows the model across devices.
                        self.register_buffer(
                            f"pos_embed{index}", position, persistent=False
                        )
                memory = encoder(flattened, pos_embed=position)
                projected[index] = (
                    memory.permute(0, 2, 1)
                    .reshape(-1, self.hidden_dim, height, width)
                    .contiguous()
                )
                if (
                    self.training
                    and self.feature_projector is not None
                    and index == self.encoder_idx_for_distillation
                ):
                    projected_f5 = self.feature_projector(
                        projected[index].permute(0, 2, 3, 1)
                    ).permute(0, 3, 1, 2)
        inner = [projected[-1]]
        for index in range(len(self.in_channels) - 1, 0, -1):
            high = self.lateral_convs[len(self.in_channels) - 1 - index](inner[0])
            inner[0] = high
            upsampled = F.interpolate(high, scale_factor=2, mode="nearest")
            fused_input = (
                upsampled + projected[index - 1]
                if self.fuse_op == "sum"
                else torch.cat([upsampled, projected[index - 1]], 1)
            )
            inner.insert(
                0, self.fpn_blocks[len(self.in_channels) - 1 - index](fused_input)
            )
        outputs = [inner[0]]
        for index in range(len(self.in_channels) - 1):
            downsampled = self.downsample_convs[index](outputs[-1])
            fused_input = (
                downsampled + inner[index + 1]
                if self.fuse_op == "sum"
                else torch.cat([downsampled, inner[index + 1]], 1)
            )
            outputs.append(self.pan_blocks[index](fused_input))
        return (
            (outputs, projected_f5)
            if self.training and projected_f5 is not None
            else outputs
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "in_channels": [shape.channels for shape in input_shape],
            "feat_strides": [shape.stride for shape in input_shape],
        }

    @property
    def out_shape(self):
        return [
            ShapeSpec(channels=self.hidden_dim, stride=stride)
            for stride in self.feat_strides
        ]


@register
@serializable
class RTDETRV2HybridEncoder(DFINEHybridEncoder):
    def __init__(
        self,
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=None,
        version="rt_detrv2",
    ):
        if version != "rt_detrv2":
            raise ValueError("RTDETRV2HybridEncoder requires version rt_detrv2")
        super().__init__(
            in_channels=in_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enc_act=enc_act,
            use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers,
            pe_temperature=pe_temperature,
            expansion=expansion,
            depth_mult=depth_mult,
            act=act,
            eval_spatial_size=eval_spatial_size,
            version=version,
        )


__all__ = ["DFINEHybridEncoder", "RTDETRV2HybridEncoder"]
