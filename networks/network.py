import torch.nn as nn
import torch
from networks.globalNet import globalNet
from networks.refineNet import refineNet

__all__ = ['CPN50', 'CPN101']

from networks.resnet import resnet101, resnet50


class CPN(nn.Module):
    def __init__(self, resnet, output_shape, num_class, pretrained=True):
        super(CPN, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        self.resnet = resnet
        self.global_net = globalNet(channel_settings, output_shape, num_class)
        self.refine_net = refineNet(channel_settings[-1], output_shape, num_class)

    def forward(self, x):
        res_out = self.resnet(x)
        global_fms, global_outs = self.global_net(res_out)
        refine_out = self.refine_net(global_fms)

        return global_outs, refine_out


def CPN50(out_size, num_class, pretrained=True):
    res50 = resnet50(pretrained=pretrained)
    model = CPN(res50, output_shape=out_size, num_class=num_class, pretrained=pretrained)
    return model


def CPN101(out_size, num_class, pretrained=True):
    res101 = resnet101(pretrained=pretrained)
    model = CPN(res101, output_shape=out_size, num_class=num_class, pretrained=pretrained)
    return model


if __name__ == '__main__':
    data_shape = (256, 192)
    output_shape = (64, 48)
    num_class = 17
    model = CPN50(out_size=output_shape, num_class=num_class, pretrained=False)
    x = torch.randn((1, 3, *data_shape))
    print(model(x))
