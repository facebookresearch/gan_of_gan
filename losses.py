"""Modified VGG16 to compute perceptual loss.

This class is mostly copied from pytorch/examples.
See, fast_neural_style in https://github.com/pytorch/examples.
"""

import torch
import torch.nn as nn
from torchvision import models


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, which_layer=16):
        """Init function, compute the VGG feature of an input at certain layer.
        A simple illustration of VGG network is conv->conv->maxpool->conv->conv
        ->maxpool->conv->conv->conv->maxpool (layer 16)->conv->conv->conv->
        maxpool->conv->conv->conv->maxpool.

        Parameters
        ----------
        requires_grad : bool
        which_layer : int
        """
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 31):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        """Initialize VGG16 network and the MSE loss to compute the feature
        differences."""
        super(VGGLoss, self).__init__()
        self.vgg = Vgg16().cuda()
        self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()
        self.weights = [1.0, 1.0, 1.0, 0.0, 0.0]
        self.weights = [1.0, 1.0, 1.0, 1.0, 0.0]
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    def forward(self, x, y):
        """Compute the beginning-intermediate feature of x, y and their MSE loss."""
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
