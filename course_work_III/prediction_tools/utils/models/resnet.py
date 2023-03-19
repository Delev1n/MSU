from typing import Collection, Optional

import torch
import torch.nn as nn
from fastai.layers import AdaptiveConcatPool1d, LinBnDrop
from fastcore.basics import listify


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, kernel_size=[3, 3], downsample=None):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]  # //2+1]

        self.conv1 = nn.Conv1d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=kernel_size[0],
            stride=stride,
            padding=(kernel_size[0] - 1) // 2,  # excluded custom `conv` function
            bias=False,
        )

        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=kernel_size[1],
            stride=1,
            padding=(kernel_size[1] - 1) // 2,  # excluded custom `conv` function
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.out_features = planes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck1d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, kernel_size=3, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(
            planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_features = planes * 4

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet1d(nn.Module):
    """1d adaptation of the torchvision resnet (not original implementation, class was adapted to work with metadata)"""

    def __init__(
        self,
        block,
        layers,
        kernel_size=3,
        num_classes=2,
        input_channels=3,
        inplanes=64,
        fix_feature_dim=True,
        kernel_size_stem=None,
        stride_stem=2,
        pooling_stem=True,
        stride=2,
        lin_ftrs_head=None,
        ps_head=0.5,
        bn_final_head=False,
        bn_head=True,
        act_head="relu",
        concat_pooling=True,
    ):

        super(ResNet1d, self).__init__()
        self.stem = None
        self.backbone = None
        self.pooling_adapter = None
        self.head = None
        self.inplanes = inplanes

        if kernel_size_stem is None:
            kernel_size_stem = (
                kernel_size[0] if isinstance(kernel_size, list) else kernel_size
            )

        # stem
        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                inplanes,
                kernel_size=kernel_size_stem,
                stride=stride_stem,
                padding=(kernel_size_stem - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1) if pooling_stem else None,
        )

        # backbone
        self.backbone = nn.Sequential()

        for i, _ in enumerate(layers):

            if not self.backbone:
                out_feats = inplanes
            else:
                out_feats = inplanes if fix_feature_dim else (2**i) * inplanes

            blks = layers[i]

            self.backbone.add_module(
                "hidden_block{}".format(i),
                self.__make_layer(
                    block=block,
                    out_features=out_feats,
                    blocks=blks,
                    stride=stride,
                    kernel_size=kernel_size,
                ),
            )

            self.pooling_adapter = nn.Sequential(
                AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2),
                nn.Flatten(),
            )

        # head
        self.head = self.__make_head(
            n_features=(inplanes if fix_feature_dim else (2 ** len(layers) * inplanes))
            * block.expansion,
            n_classes=num_classes,
            lin_ftrs=lin_ftrs_head,
            ps=ps_head,
            bn_final=bn_final_head,
            bn=bn_head,
            act=act_head,
            concat_pooling=concat_pooling,
        )

    def get_cnn(self):
        return (
            nn.Sequential(self.stem, self.backbone),
            self.backbone[-1][-1].out_features,
        )

    def __make_layer(self, block, out_features, blocks, stride=1, kernel_size=3):
        downsample = None

        if stride != 1 or self.inplanes != out_features * block.expansion:
            downsample = nn.Sequential()
            downsample.add_module(
                f"{block.__name__}_downsampler",
                nn.Conv1d(
                    self.inplanes,
                    out_features * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
            )

            downsample.add_module(
                f"{block.__name__}_normalizer",
                nn.BatchNorm1d(out_features * block.expansion),
            )

        layers = nn.Sequential()
        layers.add_module(
            f"{block.__name__}_layer0",
            block(self.inplanes, out_features, stride, kernel_size, downsample),
        )
        self.inplanes = out_features * block.expansion
        for i in range(1, blocks):
            layers.add_module(
                f"{block.__name__}_layer{i}", block(self.inplanes, out_features)
            )

        return layers

    def __make_head(
        self,
        n_features: int,
        n_classes: int,
        lin_ftrs: Optional[Collection[int]] = None,
        ps: float = 0.5,
        bn_final: bool = False,
        bn: bool = True,
        act="relu",
        concat_pooling=True,
    ):
        "Model head that takes `n_features` features, runs through `lin_ftrs`, and about `n_classes` classes; added bn and act here"

        lin_ftrs = (
            [2 * n_features if concat_pooling else n_features, n_classes]
            if lin_ftrs is None
            else [2 * n_features if concat_pooling else n_features]
            + lin_ftrs
            + [n_classes]
        )  # was [nf, 512,nc]

        ps = listify(ps)
        if len(ps) == 1:
            ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps
        actns = [nn.ReLU(inplace=True) if act == "relu" else nn.ELU(inplace=True)] * (
            len(lin_ftrs) - 2
        ) + [None]

        layers = nn.Sequential()

        layers.add_module("pooling_adapter", self.pooling_adapter)

        for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
            layers.add_module("lin_bn_drop", LinBnDrop(ni, no, bn, p, actn))

        if bn_final:
            layers.add_module("bn_final", nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))

        return layers

    def forward(self, x):
        y = self.stem(x)
        y = self.backbone(y)
        y = self.head(y)
        return y


def resnet1d18(**kwargs):
    kwargs["block"] = BasicBlock1d
    kwargs["layers"] = [2, 2, 2, 2]
    return ResNet1d(**kwargs)