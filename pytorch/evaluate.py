import torch
from model import LAPNet, two_branch_LAPNet
import yaml
from eval_plots import eval_plot
from util import get_sample_results, obj
import json


@torch.no_grad()
def evaluate_checkpoint(args):
    checkpoint = args.checkpoint_path
    if args.model == '2D':
        model = LAPNet()
    elif args.model == '3D':
        model = two_branch_LAPNet()

    model.load_state_dict(
        torch.load(checkpoint, map_location='cuda:0')['model_state_dict'])
    print(model)
    model.cuda()
    model.eval()
    model.load_state_dict(
        torch.load(checkpoint, map_location='cuda:0')['model_state_dict'])

    with torch.no_grad():
        list_predictions = []
        IDs = args.IDs
        for name in IDs:
            for acc in args.acc_list:
                results = get_sample_results(model, name, acc, args)
                list_predictions.append(results)

        eval_plot(list_predictions, args.acc_list, save_path=args.save_path)

    return list_predictions


if __name__ == '__main__':
    with open('/home/studghoul1/lapnet/lapnet/pytorch/configs/evaluate_resp.yaml', 'r') as stream:
        args = yaml.safe_load(stream)
    args = json.loads(json.dumps(args), object_hook=obj)
    evaluate_checkpoint(args.Evaluate)
