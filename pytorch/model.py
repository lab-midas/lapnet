import torch
import torch.nn.functional as F
import torch.nn as nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def warp_torch(x,flo):
    B, C, H, W = x.size()
    theta_sample = torch.tensor([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=torch.float)

    theta_batch = theta_sample.unsqueeze(0).repeat(B, 1, 1).cuda()

    flo_scaled = 2*flo / W

    theta_batch[:,:,-1] = flo_scaled

    size = torch.Size((B, C, H, W))
    grid = F.affine_grid(theta_batch, size, align_corners=True).cuda()
    output = F.grid_sample(x.float(), grid, align_corners=True)
    return output


class WarpLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, flo):
        return warp_torch(x, flo)



class LAPNet_SS(nn.Module):
    def __init__(self):
        super(LAPNet_SS, self).__init__()
        self.warping = WarpLayer()
        self.LAPNet = LAPNet()

    def forward(self, x, mov):
        x = self.LAPNet(x)
        x = self.warping(mov, x)
        return x


def initialize_modules(modules):
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


class LAPNet(nn.Module):
    def __init__(self):
        super(LAPNet, self).__init__()
        self.conv1   = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1)
        self.act1    =    nn.LeakyReLU(0.1)
        self.conv2   =     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.act2    =    nn.LeakyReLU(0.1)
        self.conv2_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.act2_1  = nn.LeakyReLU(0.1)
        self.conv3   = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.act3    = nn.LeakyReLU(0.1)
        self.conv4   = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.act4    =  nn.LeakyReLU(0.1)
        self.conv4_1 =  nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.act4_1  =  nn.LeakyReLU(0.1)
        self.pool    =   nn.MaxPool2d(kernel_size=5, stride=1)
        self.fc2     =   nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)

        initialize_modules(self.modules())

    def forward(self, x):
        x = self.conv1  (x)
        x =  self.act1   (x)
        x =  self.conv2  (x)
        x =  self.act2   (x)
        x =  self.conv2_1(x)
        x =  self.act2_1 (x)
        x =  self.conv3  (x)
        x =  self.act3   (x)
        x =  self.conv4  (x)
        x =  self.act4   (x)
        x =  self.conv4_1(x)
        x =  self.act4_1 (x)
        x =  self.pool   (x)
        x =  self.fc2    (x)
        return torch.squeeze(x)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_per_process_memory_fraction(0.33, device=None)
    print(device)
    model = LAPNet_SS().to(device)
    print(model)
    random_k = torch.rand(64, 4, 33, 33).to(device)
    random_mov = torch.rand(64, 2, 33, 33).to(device)
    x = model(random_k, random_mov)
    print(x.shape)
    print('number of parameters ', count_parameters(model))


