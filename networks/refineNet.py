import torch.nn as nn
import torch


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):  # (B, inplanes, h, w)->(B, planes * 2, h, w)
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * 2,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * 2),
        )

        self.stride = stride

    def forward(self, x):
        residual = x  # (B, inplanes, h, w)

        out = self.conv1(x)  # (B, planes, h, w)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # (B, planes, h, w)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)  # (B, planes*2, h, w)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)  # (B, planes*2, h, w)

        out += residual
        out = self.relu(out)

        return out  # (B, planes*2, h, w)


class refineNet(nn.Module):
    def __init__(self, lateral_channel, out_shape, num_class):
        super(refineNet, self).__init__()
        cascade = []
        num_cascade = 4
        for i in range(num_cascade):
            cascade.append(self._make_layer(lateral_channel, num_cascade - i - 1, out_shape))
        self.cascade = nn.ModuleList(cascade)
        self.final_predict = self._predict(4 * lateral_channel, num_class)

    def _make_layer(self, input_channel, num, output_shape):
        layers = [Bottleneck(input_channel, 128) for _ in range(num)]
        # layers = []
        # for i in range(num):
        #     layers.append(Bottleneck(input_channel, 128))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        return nn.Sequential(*layers)

    def _predict(self, input_channel, num_class):
        layers = [
            Bottleneck(input_channel, 128),
            nn.Conv2d(256, num_class, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_class)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        refine_fms = []
        for i in range(4):
            refine_fms.append(self.cascade[i](x[i]))
        # [(B, 256, 64, 48)]*4
        out = torch.cat(refine_fms, dim=1)  # (B, 1024, 64, 48)
        out = self.final_predict(out)
        return out
