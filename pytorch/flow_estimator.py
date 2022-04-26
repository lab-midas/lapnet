import glob
import os
import numpy as np
from util import read_test_data_2D, obj
from core import arrange_predicted_flow
import torch
from model import LAPNet
import yaml
import json


def flow_torch2np(pred, args, height, width):
    flow_pixel = pred.cpu().detach().numpy()
    flow_pixel = np.squeeze(flow_pixel)
    flow = arrange_predicted_flow(flow_pixel, args, height, width, 'coronal')
    if not args.supervised:
        flow = np.flip(flow, -1)
    return flow


def save_LAPNet_estimation_as_npy(args):

    ## read model checkpoint
    model = LAPNet()
    checkpoint = torch.load(args.checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    print(model)
    model.cuda()
    model.eval()

    IDs_path = glob.glob('/mnt/qdata/projects/MoCo/LAPNet/coronal_test_data/1/*.npz')
    IDs = [p.split('/')[-1] for p in IDs_path]

    acc_list = np.arange(0, 31, 2)
    acc_list[0] = 1

    for i, ID in enumerate(IDs[104:105]):
        print(args.experiment, i, 'from', len(IDs))
        ID_name = ID.split('.')[0]
        save_dir = f'{args.save_dir}/{args.experiment}/{ID_name}'
        os.makedirs(save_dir, exist_ok=True)
        for acc in acc_list[:1]:
            im1, im2, flow_orig, data_forward = read_test_data_2D(ID, acc, args)
            height, width = im1.shape
            ref_patches = data_forward[..., :2]
            mov_patches = data_forward[..., 2:]

            data_ref = np.concatenate((ref_patches, ref_patches), axis=-1)
            data_mov = np.concatenate((mov_patches, mov_patches), axis=-1)
            data_backward = np.concatenate((mov_patches, ref_patches), axis=-1)

            input_forward = torch.from_numpy(np.transpose(data_forward, (0, 3, 2, 1)))
            input_ref = torch.from_numpy(np.transpose(data_ref, (0, 3, 2, 1)))
            input_mov = torch.from_numpy(np.transpose(data_mov, (0, 3, 2, 1)))
            input_backward = torch.from_numpy(np.transpose(data_backward, (0, 3, 2, 1)))

            with torch.no_grad():
                flow_forward = model(input_forward.cuda())
                # flow_ref = model(input_ref.cuda())
                # flow_mov = model(input_mov.cuda())
                # flow_backward = model(input_backward.cuda())

            flow_forward_arranged = flow_torch2np(flow_forward, args, height, width)
            # flow_ref_arranged = flow_torch2np(flow_ref, args, height, width)
            # flow_mov_arranged = flow_torch2np(flow_mov, args, height, width)
            # flow_backward_arranged = flow_torch2np(flow_backward, args, height, width)

            flow_field = np.zeros((2, 2, height, width, 2), dtype=np.float32)

            flow_field[0, 1, ...] = flow_forward_arranged
            flow_field[1, 0, ...] = -flow_forward_arranged #flow_backward_arranged

            # flow_field[0, 0, ...] = flow_ref_arranged
            # flow_field[1, 1, ...] = flow_mov_arranged

            np.save(f'/home/studghoul1/lapnet/lapnet/test_flow.npy', flow_field)
            # print(ID, acc, 'is saved in', save_dir)


if __name__ == '__main__':
    with open('/home/studghoul1/lapnet/lapnet/pytorch/config.yaml', 'r') as stream:
        args = yaml.safe_load(stream)
    # device configuration
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # args['Setup']['gpu_num']
    args = json.loads(json.dumps(args), object_hook=obj)


    # args.Evaluate.experiment = 'consistency_loss'
    # args.Evaluate.checkpoint_path = '/home/studghoul1/lapnet/logs/LAPNet/semi_supervised/ckpt_num_17.pt'
    # args.Evaluate.supervised = False
    # save_LAPNet_estimation_as_npy(args.Evaluate)

    args.Evaluate.experiment = 'self_supervised'
    args.Evaluate.checkpoint_path = '/home/studghoul1/lapnet/logs/LAPNet/photometric_different_lr/ckpt_num_11.pt'
    args.Evaluate.supervised = False
    save_LAPNet_estimation_as_npy(args.Evaluate)

    # args.Evaluate.experiment = 'supervised_than_self_supervised'
    # args.Evaluate.checkpoint_path = '/home/studghoul1/lapnet/logs/LAPNet/ss_after8_epochs/ckpt_num_11.pt'
    # args.Evaluate.supervised = True
    # save_LAPNet_estimation_as_npy(args.Evaluate)

