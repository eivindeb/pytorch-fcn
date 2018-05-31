import torch
import torch.nn.functional as F
from torch import nn
from torchfcn.models import resnet as resnet_models

from utils import initialize_weights
from utils.misc import Conv2dDeformable
from torchfcn.utils import CFARModule, MetadataModule


# TODO: instead of removing Batchnorm from pyramid size 1, do .view(1, 2048) before and (1, 2048, 1, 1) after

class _PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting, group_norm=True):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            if group_norm:
                self.features.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(s),
                    nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(32, reduction_dim),
                    nn.ReLU(inplace=True)
                ))
            else:
                if s == 1:
                    self.features.append(nn.Sequential(
                        nn.AdaptiveAvgPool2d(s),
                        nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    self.features.append(nn.Sequential(
                        nn.AdaptiveAvgPool2d(s),
                        nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                        nn.BatchNorm2d(reduction_dim, momentum=.95),
                        nn.ReLU(inplace=True)
                    ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)
        return out


class PSPNet(nn.Module):
    def __init__(self, num_classes, pretrained=True, use_aux=True, metadata_channels=0, in_channels=1, use_cfar_filters=False, freeze=None, group_norm=True):
        super(PSPNet, self).__init__()
        self.use_aux = use_aux
        self.metadata = metadata_channels > 0
        self.cfar = use_cfar_filters

        if self.metadata:
            #self.nn_meta = MetadataModule(in_features=metadata_channels, out_features=1)
            self.nn_meta = nn.Sequential(
                nn.Linear(metadata_channels, metadata_channels),
                nn.ReLU(inplace=True),
                nn.Linear(metadata_channels, metadata_channels),
                nn.ReLU(inplace=True),
                nn.Linear(metadata_channels, metadata_channels),
                nn.ReLU(inplace=True),
                nn.Linear(metadata_channels, metadata_channels),
                nn.ReLU(inplace=True),
                nn.Linear(metadata_channels, metadata_channels),
                nn.ReLU(inplace=True),
                nn.Linear(metadata_channels, metadata_channels),
                nn.ReLU(inplace=True),
                nn.Linear(metadata_channels, 1),
                nn.ReLU(inplace=True)
            )

        if self.cfar:
            self.nn_cfar = nn.ModuleList([
                CFARModule(10, 2, 1),
                CFARModule(15, 2, 1),
                CFARModule(20, 3, 1),
            ])

        resnet = resnet_models.resnet101(in_channels=in_channels + (len(self.nn_cfar) if self.cfar else 0), pretrained=pretrained, group_norm=group_norm)

        if freeze is not None:
            for name, param in resnet.named_parameters():
                if "layer0" in freeze and "layer" not in name:
                    param.requires_grad = False
                if name.split(".")[0] in freeze:
                    param.requires_grad = False

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6), group_norm=group_norm)
        self.final = nn.Sequential(
            nn.Conv2d(4096 + (1 if self.metadata else 0), 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95) if not group_norm else nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        if use_aux:
            self.aux_logits = nn.Conv2d(1024, num_classes, kernel_size=1)
            initialize_weights(self.aux_logits)

        initialize_weights(self.ppm, self.final)

    def forward(self, x, m=None):
        x_size = x.size()
        x_in = x

        if self.cfar:
            for f in self.nn_cfar:
                x_cfar = f(x_in)
                x_cfar = (x_cfar / x_cfar.max()) * 200  # should share approximately same scale as input
                x = x.cat([x, x_cfar], dim=1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.training and self.use_aux:
            aux = self.aux_logits(x)
        x = self.layer4(x)
        x = self.ppm(x)

        if self.metadata and m is not None:
            x_m = self.nn_meta(m)
            x_m = x_m.repeat(x.shape[2], x.shape[3])
            x_m = x_m.unsqueeze(0)
            x_m = x_m.unsqueeze(0)
            x = x.cat([x, x_m], dim=1)

        x = self.final(x)
        if self.training and self.use_aux:
            return F.upsample(x, x_size[2:], mode='bilinear'), F.upsample(aux, x_size[2:], mode='bilinear')
        return F.upsample(x, x_size[2:], mode='bilinear')