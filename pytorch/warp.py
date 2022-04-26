import torch
import torch.nn.functional as F


def warp_torch(x, flo):
    B, C, H, W = x.size()
    theta_sample = torch.tensor([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=torch.float)

    theta_batch = theta_sample.unsqueeze(0).repeat(B, 1, 1).cuda()

    flo = torch.flip(flo, dims=[1])
    flo_scaled = 2 * flo / W

    theta_batch[:, :, -1] = flo_scaled

    size = torch.Size((B, C, H, W))
    grid = F.affine_grid(theta_batch, size, align_corners=True).cuda()
    output = F.grid_sample(x.float(), grid, align_corners=True)

    mask = torch.ones(x.size(), dtype=x.dtype)
    if x.is_cuda:
        grid = grid.cuda()
        mask = mask.cuda()
    mask = torch.nn.functional.grid_sample(mask.float(), grid, align_corners=True)
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    output = output * mask

    return output
