import torch
from util import count_parameters
from warp import warp_torch
import torch.nn as nn
import os


def initialize_modules(modules):
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


class two_branch_LAPNet(nn.Module):
    def __init__(self):
        super(two_branch_LAPNet, self).__init__()
        self.LAPNet1 = LAPNet()
        self.LAPNet2 = LAPNet()
        self.weight1 = torch.nn.Parameter(data=torch.Tensor(1, ), requires_grad=True)
        self.weight1.data.uniform_(0, 1)
        self.weight2 = torch.nn.Parameter(data=torch.Tensor(1, ), requires_grad=True)
        self.weight2.data.uniform_(0, 1)

    def forward(self, k_coronal, k_sagittal):
        coronal_flow = self.LAPNet1(k_coronal)
        sagittal_flow = self.LAPNet2(k_sagittal)
        weight1 = torch.clamp(self.weight1, 0, 1)
        weight2 = torch.clamp(self.weight2, 0, 1)
        u_1 = weight1 * coronal_flow[:, 0] + (1 - weight1) * sagittal_flow[:, 0]
        u_2 = coronal_flow[:, 1]
        u_3 = weight2 * coronal_flow[:, 0] + (1 - weight2) * sagittal_flow[:, 0]
        u_4 = sagittal_flow[:, 1]
        flow_cor = torch.stack((u_1, u_2), dim=-1)
        flow_sag = torch.stack((u_3, u_4), dim=-1)
        return flow_cor, flow_sag


class LAPNet(nn.Module):
    def __init__(self):
        super(LAPNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1)
        self.act1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.act2 = nn.LeakyReLU(0.1)
        self.conv2_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.act2_1 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.LeakyReLU(0.1)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.act4 = nn.LeakyReLU(0.1)
        self.conv4_1 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.act4_1 = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1)
        self.fc2 = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)

        initialize_modules(self.modules())

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv2_1(x)
        x = self.act2_1(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.conv4_1(x)
        x = self.act4_1(x)
        x = self.pool(x)
        x = self.fc2(x)
        return torch.squeeze(x)


class LAPNet_21(nn.Module):
    def __init__(self):
        super(LAPNet_21, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1)
        self.act1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.act2 = nn.LeakyReLU(0.1)
        self.conv2_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.act2_1 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.LeakyReLU(0.1)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.act4 = nn.LeakyReLU(0.1)
        self.conv4_1 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.act4_1 = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.fc2 = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)

        initialize_modules(self.modules())

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv2_1(x)
        x = self.act2_1(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.conv4_1(x)
        x = self.act4_1(x)
        x = self.pool(x)
        x = self.fc2(x)
        return torch.squeeze(x)


if __name__ == '__main__':
    model = LAPNet_21().cuda()
    print(model)
    random_k = torch.rand(64, 4, 21, 21).cuda()
    random_mov = torch.rand(64, 4, 21, 21).cuda()
    x = model(random_k)
    print(x.shape)
    print('number of parameters ', count_parameters(model))
