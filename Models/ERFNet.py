#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation=nn.ReLU):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out  # + x
        return out


class Channel_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation=nn.ReLU):
        super(Channel_Attn, self).__init__()
        channel = in_dim
        reduction = 8
        self.fc = nn.Sequential(
            nn.Linear(channel * 2, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.lamb = nn.Parameter(torch.ones(1))

    def forward(self, x):
        size = x.size()
        new = x.view(size[0], size[1], -1)
        mean = torch.mean(new, dim=-1)
        std = torch.std(new, dim=-1)
        input = torch.cat((mean, std), 1)
        attention = self.fc(input)
        attention_new = attention.view(size[0], size[1], 1, 1)
        attention_new = attention_new.expand_as(x)
        out = self.lamb * attention_new * x

        return out


class SCModule(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation=nn.ReLU):
        super(SCModule, self).__init__()
        channel = in_dim
        self.spatial = Self_Attn(channel)
        self.channel = Channel_Attn(channel)

    def forward(self, x):
        out = self.spatial(x) + self.channel(x) + x
        return out


class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1*dilated), bias=True, dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output+input)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        chans = 32 if in_channels > 16 else 16
        self.initial_block = DownsamplerBlock(in_channels, chans)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(chans, 64))

        for x in range(0, 5):
            self.layers.append(non_bottleneck_1d(64, 0.03, 1)) 

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        #Only in encoder mode:
        self.scenhancer = SCModule(128)
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)
        
        output = self.scenhancer(output)
        
        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layer1 = UpsamplerBlock(128, 64)
        self.layer2 = non_bottleneck_1d(64, 0, 1)
        self.layer3 = non_bottleneck_1d(64, 0, 1) # 64x64x304

        self.layer4 = UpsamplerBlock(64, 32)
        self.layer5 = non_bottleneck_1d(32, 0, 1)
        self.layer6 = non_bottleneck_1d(32, 0, 1) # 32x128x608

        self.output_conv = nn.ConvTranspose2d(32, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        em2 = output
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.layer6(output)
        em1 = output

        output = self.output_conv(output)

        return output, em1, em2


class Net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):  #use encoder to pass pretrained encoder
        super().__init__()
        self.encoder = Encoder(in_channels, out_channels)
        self.decoder = Decoder(out_channels)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)
            return self.decoder.forward(output)