from model import two_branch_LAPNet, LAPNet
from util import count_parameters, eval_dict, eval_imgs_sag, obj
import numpy as np
import os
from input_pipeline import fetch_dataloader
import torch
import json
import yaml
import wandb
from losses import LAPNetLoss_2D, LAPNetLoss_3D


def train(args):
    args_experiment = args.Train

    # define the model
    if args_experiment.branches:
        model = two_branch_LAPNet()
    else:
        model = LAPNet()
    print(model)
    model.cuda()
    print("Parameter Count: %d" % count_parameters(model))

    if args_experiment.checkpoint:
        model.load_state_dict(
            torch.load(args_experiment.checkpoint, map_location='cuda:0')['model_state_dict'])
        print('checkpoint is loaded', args_experiment.checkpoint)

    # fetch the data
    train_loader = fetch_dataloader(args_experiment)

    # hyperparams
    optimizer = torch.optim.Adam(model.parameters(), lr=args_experiment.lr, weight_decay=args_experiment.wdecay)
    total_step = len(train_loader)
    print(total_step)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(total_step + 1) * 3, gamma=0.85)

    if not args_experiment.debug:
        ckpt_dir = os.path.join(args_experiment.save_dir, args_experiment.experiment_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        wandb.init(project=args_experiment.project, entity="ayaghoul", name=args_experiment.experiment_name)

    scaler = torch.cuda.amp.GradScaler()
    if args_experiment.branches:
        criterion = LAPNetLoss_3D(args_experiment.train_loss)
    else:
        criterion = LAPNetLoss_2D(args_experiment.train_loss)

    for epoch in range(args_experiment.epochs):
        train_log_dic = {}
        model.train()
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            # load data
            # 3D training
            if args_experiment.branches:
                if args_experiment.training_mode == 'supervised':
                    k_cor, k_sag, flow_gt = data_blob['k_coronal'], data_blob['k_sagittal'], data_blob['flow']
                    k_cor, k_sag, flow_gt = k_cor.cuda(), k_sag.cuda(), flow_gt.cuda()
                    ref, mov = None, None
                    with torch.cuda.amp.autocast():
                        flow = model(k_cor, k_sag)

                elif args_experiment.training_mode == 'self_supervised':
                    k_cor, k_sag, ref_c, mov_c, ref_s, mov_s, flow_gt = data_blob['k_coronal'], data_blob['k_sagittal'], \
                                                                        data_blob['ref_c'], data_blob['mov_c'], \
                                                                        data_blob['ref_s'], data_blob['mov_s'], \
                                                                        data_blob['flow']
                    k_cor, k_sag, ref_c, mov_c, ref_s, mov_s, flow_gt = k_cor.cuda(), k_sag.cuda(), \
                                                                        ref_c.cuda(), mov_c.cuda(), \
                                                                        ref_s.cuda(), mov_s.cuda(), \
                                                                        flow_gt.cuda()
                    ref = [ref_c, ref_s]
                    mov = [mov_c, mov_s]
                    with torch.cuda.amp.autocast():
                        flow = model(k_cor, k_sag)

            # 2D training
            else:
                if args_experiment.training_mode == 'supervised':
                    k_space, flow_gt = data_blob['k_space'], data_blob['flow']
                    k_space, flow_gt = k_space.cuda(), flow_gt.cuda()
                    ref, mov = None, None
                elif args_experiment.training_mode == 'self_supervised':
                    k_space, mov, ref, flow_gt = data_blob['k_space'], data_blob['img_mov'], data_blob['img_ref'], \
                                                 data_blob['flow']
                    k_space, mov, ref, flow_gt = k_space.cuda(), mov.cuda(), ref.cuda(), flow_gt.cuda()

                # loss
                with torch.cuda.amp.autocast():
                    flow = model(k_space)

            loss_dic = criterion(flow_gt=flow_gt, flow_pred=flow, ref=ref, mov=mov)

            loss = loss_dic['total_loss']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            for key in loss_dic:
                if key not in train_log_dic.keys():
                    train_log_dic[key] = 0
                train_log_dic[key] += loss_dic[key].item()

            # print('loss', f'{i_batch}/{total_step}', train_log_dic)

        train_log_dic = {k: v / total_step for k, v in train_log_dic.items()}
        # train_log_dic['lr'] = scheduler.get_last_lr()
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        print('train logs:\n', train_log_dic)

        if not args_experiment.debug:
            # log training loss
            wandb.log(train_log_dic)

            # checkpoint
            current_ckpt_path = os.path.join(ckpt_dir, f'ckpt_num_{epoch}.pt')
            ckpt_dict = {
                'epoch': args_experiment.epochs,
                'lr': scheduler.get_last_lr(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), }
            torch.save(ckpt_dict, current_ckpt_path)

            # log evaluation results
            if args_experiment.eval_model:
                model.eval()
                vis_im, losses = eval_dict(model, current_ckpt_path, args.Evaluate)
                wandb.log({'evaluation_coronal_imgs': [wandb.Image(vis_im[key], caption=key) for key in vis_im.keys()]})
                wandb.log(losses)
                print('evaluation logs:\n', losses)
                # wandb.log({"eval_img": eval_image})
                if args_experiment.branches:
                    vis_im_sag = eval_imgs_sag(model, current_ckpt_path, args.Evaluate)
                    wandb.log({'eval_sagittal_imgs': [wandb.Image(vis_im_sag[key], caption=key) for key in
                                                      vis_im_sag.keys()]})


if __name__ == '__main__':

    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.cuda.empty_cache()

    with open("/home/studghoul1/lapnet/lapnet/pytorch/configs/train.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    args = json.loads(json.dumps(config), object_hook=obj)

    # device configuration
    if args.Train.debug:
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    os.environ["WANDB_API_KEY"] = 'x'
    train(args)
