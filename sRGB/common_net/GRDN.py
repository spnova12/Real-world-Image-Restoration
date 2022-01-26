import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from sRGB.common_net.cbam import *


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class GRDB(nn.Module):
    def __init__(self, numofkernels, nDenselayer, growthRate, numforrg):
        super(GRDB, self).__init__()

        modules = []
        for i in range(numforrg):
            modules.append(RDB(numofkernels, nDenselayer=nDenselayer, growthRate=growthRate))
        self.rdbs = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(numofkernels * numforrg, numofkernels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = x
        outputlist = []
        for rdb in self.rdbs:
            output = rdb(out)
            outputlist.append(output)
            out = output
        concat = torch.cat(outputlist, 1)
        out = x + self.conv_1x1(concat)
        return out


class GRDN(nn.Module):
    def __init__(self, input_channel, numforrg=1, numofrdb=16, numofconv=8, numoffilters=64, t=1, cbam=True):
        super(GRDN, self).__init__()

        self.numforrg = numforrg
        self.numofrdb = numofrdb
        self.nDenselayer = numofconv
        self.numofkernels = numoffilters
        self.t = t
        self.CBAM = cbam

        self.layer1 = nn.Conv2d(input_channel, self.numofkernels, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(self.numofkernels, self.numofkernels, kernel_size=4, stride=2, padding=1)
        self.layerCat = nn.Conv2d(self.numofkernels*2, self.numofkernels, kernel_size=3, stride=1, padding=1)

        modules = []
        for i in range(self.numofrdb // self.numforrg):
            modules.append(GRDB(self.numofkernels, self.nDenselayer, self.numofkernels, self.numforrg))
        self.rglayer = nn.Sequential(*modules)

        self.layer7 = nn.ConvTranspose2d(self.numofkernels, self.numofkernels, kernel_size=4, stride=2, padding=1)

        self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(self.numofkernels, input_channel, kernel_size=3, stride=1, padding=1)

        if self.CBAM:
            self.cbam = CBAM(self.numofkernels, 16)

    def forward(self, x, y):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        outy = self.layer1(y)
        outy = self.layer2(outy)
        outy = self.layer3(outy)

        out = torch.cat((out, outy), dim=1)
        out = self.layerCat(out)

        for grdb in self.rglayer:
            for i in range(self.t):
                out = grdb(out)

        out = self.layer7(out)
        if self.cbam:
            out = self.cbam(out)

        out = self.layer8(out)
        out = self.layer9(out)

        return out + x