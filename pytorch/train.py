from model import LAPNet, count_parameters, warp_torch
import numpy as np
import os
from input_pipeline import fetch_dataloader
import torch
import torch.nn as nn
import yaml

class LAPNetLoss(nn.Module):
    def __init__(self):
        super(LAPNetLoss, self).__init__()

    def forward(self, inputs, targets):
        w_1 = 0.8
        w_2 = 0.2

        squared_difference = torch.square(targets - inputs)
        EPE_loss = torch.mean(squared_difference)

        EAE_loss = torch.acos((1 + torch.sum(targets * inputs)) /
                                (torch.sqrt(1 + torch.sum(torch.pow(inputs, 2))) *
                                 torch.sqrt(1 + torch.sum(torch.pow(targets, 2)))))

        return torch.add(w_1 * EPE_loss, w_2 * EAE_loss)


def get_mask(flo, shape):
    mask = torch.ones(shape)
    flow = torch.round(flo)
    for i in range(shape[0]):
        if flow[i, 1] > 0:
            mask[i, :, :int(flow[i, 1]), :] = 0
        else:
            mask[i, :, int(flow[i, 1]):, :] = 0
        if flow[i, 0] > 0:
            mask[i, :, :, :int(flow[i, 0])] = 0
        else:
            mask[i, :,:, int(flow[i, 0]):] = 0
    return mask.cuda()


def photometric_loss(im1, im2, flo, mask=1, mode='L1'):
    warped = warp_torch(im1, flo)
    im_diff = im2 - warped
    if mode == 'L1':
        return torch.abs(im_diff * mask).mean()
    elif mode == 'L2':
        return torch.square(im_diff).mean()


def train(args):
    # define the model
    model = LAPNet()
    print(model)
    model.cuda()
    print("Parameter Count: %d" % count_parameters(model))

    # fetch the data
    train_loader = fetch_dataloader(args)

    # hyperparams
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wdecay'], eps=1e-8)
    total_step = len(train_loader)
    if args['training_mode'] is 'self_supervised':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    args['lr'],
                                                    epochs=args['epochs'],
                                                    steps_per_epoch=total_step + 1,
                                                    pct_start=0.05,
                                                    cycle_momentum=False,
                                                    anneal_strategy='linear')
    elif args['training_mode'] is 'supervised':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args['epochs']):
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            # load data
            if args['training_mode'] is 'self_supervised':
                k_space, mov, ref = data_blob['k_space'], data_blob['img_mov'], data_blob['img_ref']
                k_space, mov, ref = k_space.cuda(), mov.cuda(), ref.cuda()
            elif args['training_mode'] is 'supervised':
                k_space, flow_gt = data_blob['k_space'], data_blob['flow']
                k_space, flow_gt = k_space.cuda(), flow_gt.cuda()

            # loss
            with torch.cuda.amp.autocast():
                flow = model(k_space)
                if args['training_mode'] is 'self_supervised':
                    loss = photometric_loss(ref, mov, flow, mode='L1')
                elif args['training_mode'] is 'supervised':
                    loss = LAPNetLoss()(flow_gt, flow)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip'])
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            if (i_batch + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, args['epochs'], i_batch + 1, total_step, loss.item()))

        # checkpoint
        PATH = os.path.join(args['save_dir'], args['experiment_name'])
        torch.save({
            'epoch': args['epochs'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),},
            f'{PATH}_{epoch}.pt')


if __name__ == '__main__':
    with open("/home/studghoul1/lapnet/lapnet/pytorch/config.yaml", 'r') as stream:
        args = yaml.safe_load(stream)
    # device configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = args['Setup']['gpu_num']

    torch.manual_seed(1234)
    np.random.seed(1234)
    os.environ['CUDA_VISIBLE_DEVICES'] = args['Setup']['gpu_num']
    train(args['Train'])