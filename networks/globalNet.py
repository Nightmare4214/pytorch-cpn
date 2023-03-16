import torch.nn as nn
import torch
import math


class globalNet(nn.Module):
    def __init__(self, channel_settings, output_shape, num_class):
        super(globalNet, self).__init__()
        self.channel_settings = channel_settings
        laterals, upsamples, predict = [], [], []
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i]))
            predict.append(self._predict(output_shape, num_class))
            if i != len(channel_settings) - 1:
                upsamples.append(self._upsample())
        self.laterals = nn.ModuleList(laterals)
        self.upsamples = nn.ModuleList(upsamples)
        self.predict = nn.ModuleList(predict)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _lateral(self, input_size):
        layers = [
            nn.Conv2d(input_size, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        ]

        return nn.Sequential(*layers)

    def _upsample(self):
        layers = [
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256)
        ]

        return nn.Sequential(*layers)

    def _predict(self, output_shape, num_class):
        layers = [
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, num_class, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(size=output_shape, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(num_class)
        ]

        return nn.Sequential(*layers)

    def forward(self, x):  # [(B, 2048, 8, 6), (B, 1024, 16, 12),(B, 512, 32, 24), (B, 256, 64, 48)]
        global_fms, global_outs = [], []
        for i in range(len(self.channel_settings)):
            if i == 0:
                feature = self.laterals[i](x[i])
            else:
                feature = self.laterals[i](x[i]) + up
            global_fms.append(feature)
            if i != len(self.channel_settings) - 1:
                up = self.upsamples[i](feature)
            feature = self.predict[i](feature)
            global_outs.append(feature)
        # [(B, 256, 8, 6), (B, 256, 16, 12), (B, 256, 32, 24), (B, 256, 64, 48)]
        # [(B, 17, 64, 48), (B, 17, 64, 48), (B, 17, 64, 48), (B, 17, 64, 48)]
        return global_fms, global_outs
