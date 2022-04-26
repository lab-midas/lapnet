import torch
import torch.nn as nn
from warp import warp_torch


class CriterionBase(torch.nn.Module):
    def __init__(self, config):
        super(CriterionBase, self).__init__()
        self.loss_names = config.which
        self.loss_weights = config.loss_weights
        self.loss_list = []
        for loss_name in config.which:
            loss_args = eval(f'config.{loss_name}').__dict__
            loss_item = self.get_loss(loss_name=loss_name, args_dict=loss_args)
            self.loss_list.append(loss_item)

    def get_loss(self, loss_name, args_dict):
        if loss_name == 'photometric':
            return PhotometricLoss(**args_dict)
        elif loss_name == 'smooth':
            return SmoothLoss(**args_dict)
        elif loss_name == 'EPE':
            return EPE(**args_dict)
        elif loss_name == 'EAE':
            return EAE(**args_dict)
        else:
            raise NotImplementedError


class LAPNetLoss_2D(CriterionBase, torch.nn.Module):
    def __init__(self, config):
        super().__init__(config=config)

    def forward(self, flow_pred, ref=None, mov=None, flow_gt=None):
        loss_dict = {}
        total_loss = 0
        for loss_name, loss_weight, loss_term in zip(self.loss_names, self.loss_weights, self.loss_list):
            if loss_name == 'photometric':
                warped = warp_torch(ref, flow_pred)
                i_loss = loss_term(mov, warped)
            elif loss_name == 'smooth':
                i_loss = loss_term(flow_pred, ref)
            elif loss_name == 'EPE':
                i_loss = loss_term(flow_pred, flow_gt)
            elif loss_name == 'EAE':
                i_loss = loss_term(flow_pred, flow_gt)
            else:
                raise KeyError('loss_name not registered')
            loss_dict[loss_name] = i_loss
            total_loss += loss_weight * i_loss
        loss_dict['total_loss'] = total_loss
        return loss_dict


class LAPNetLoss_3D(CriterionBase, torch.nn.Module):
    def __init__(self, config):
        super().__init__(config=config)

    def forward(self, flow_pred, ref=None, mov=None, flow_gt=None):
        loss_dict = {}
        total_loss = 0
        num_branches = 2
        flow_gt_list = [flow_gt[:, :2], torch.stack((flow_gt[:, 0], flow_gt[:, 2]), -1)]
        for loss_name, loss_weight, loss_term in zip(self.loss_names, self.loss_weights, self.loss_list):
            i_loss = 0
            for i in range(num_branches):
                if loss_name == 'photometric':
                    warped = warp_torch(ref[i], flow_pred[i])
                    i_loss += loss_term(mov[i], warped)
                elif loss_name == 'smooth':
                    i_loss += loss_term(flow_pred[i], ref[i])
                elif loss_name == 'EPE':
                    i_loss += loss_term(flow_pred[i], flow_gt_list[i])
                elif loss_name == 'EAE':
                    i_loss += loss_term(flow_pred[i], flow_gt_list[i])
                else:
                    raise KeyError('loss_name not registered')
            loss_dict[loss_name] = i_loss
            total_loss += loss_weight * i_loss
        loss_dict['total_loss'] = total_loss
        return loss_dict


class EAE(nn.Module):
    def __init__(self, mode=1):
        super(EAE, self).__init__()

    def forward(self, inputs, targets):
        EAE_loss = torch.acos((1 + torch.sum(targets * inputs)) /
                              (torch.sqrt(1 + torch.sum(torch.pow(inputs, 2))) *
                               torch.sqrt(1 + torch.sum(torch.pow(targets, 2)))))

        return EAE_loss


class EPE(nn.Module):
    def __init__(self, mode=1):
        super(EPE, self).__init__()

    def forward(self, inputs, targets):
        EPE_loss = torch.mean(torch.square(targets - inputs))
        return EPE_loss


class SmoothLoss(nn.Module):
    def __init__(self, mode=1, boundary_awareness=True, alpha=10):
        super(SmoothLoss, self).__init__()
        self.boundary_awareness = boundary_awareness
        self.alpha = alpha
        if mode == 2:
            self.func_smooth = self.smooth_grad_2nd
        elif mode == 1:
            self.func_smooth = self.smooth_grad_1st

    def smooth_grad_2nd(self, flo, image):
        img_dx, img_dy = self.gradient(image)
        dx, dy = self.gradient(flo)
        dx2, dxdy = self.gradient(dx)
        dydx, dy2 = self.gradient(dy)
        # dx2, dy2 = dx2.abs(), dy2.abs()
        eps = 1e-6
        dx2, dy2 = torch.sqrt(dx2 ** 2 + eps), torch.sqrt(dy2 ** 2 + eps)
        if self.boundary_awareness:
            weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * self.alpha)
            weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * self.alpha)
            loss_x = weights_x[:, :, :, 1:] * dx2
            loss_y = weights_y[:, :, 1:, :] * dy2
        else:
            loss_x = dx2
            loss_y = dy2
        return loss_x.mean() / 2. + loss_y.mean() / 2.

    def smooth_grad_1st(self, flow, image):
        img_dx, img_dy = self.gradient(image)
        dx, dy = self.gradient(flow)
        eps = 1e-6
        dx, dy = torch.sqrt(dx ** 2 + eps), torch.sqrt(dy ** 2 + eps)
        dx, dy = dx.abs(), dy.abs()
        if self.boundary_awareness:
            weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * self.alpha)
            weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * self.alpha)
            loss_x = weights_x * dx / 2.
            loss_y = weights_y * dy / 2.
        else:
            loss_x = dx / 2.  # todo: why here need to be divided by 2? in return it's already divided by 2
            loss_y = dy / 2.

        return loss_x.mean() / 2. + loss_y.mean() / 2.

    def gradient(self, data):
        D_dy = data[:, :, 1:] - data[:, :, :-1]
        D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
        return D_dx, D_dy

    def forward(self, flow_vec, image):
        B, C, H, W = image.size()
        flow = flow_vec[:, :, None, None].tile((1, 1, H, W))
        return self.func_smooth(flow, image).mean()


class PhotometricLoss(torch.nn.Module):
    def __init__(self, mode):
        super(PhotometricLoss, self).__init__()
        assert mode in ('L1', 'L2')
        if mode == 'L1':
            self.loss = torch.nn.L1Loss(reduction='mean')
        elif mode == 'L2':
            self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, inputs, outputs):
        return self.loss(inputs, outputs)
