from thop import profile
import torch
from Models.model import uncertainty_net
import argparse


if __name__ == '__main__':
    model = uncertainty_net(in_channels=4).cuda()
    input1 = torch.randn(1, 1, 1216, 256).cuda()
    input2 = torch.randn(1, 3, 1216, 256).cuda()
    input = [input1, input2]
    out, _, _, _ = model(input)
    print(out[-1].shape)
    flops, params = profile(model, inputs=(input1, input2,))
    print(flops)
    print(params)