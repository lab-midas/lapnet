import torch
from model import LAPNet
import os
import yaml
from ..core.eval_utils import *

def get_sample_results(model, name, acc, args):
    im1, im2, flow_orig, batches_cp = read_test_data(name, acc, args)

    data = torch.from_numpy(np.transpose(batches_cp, (0, 3, 2, 1)))
    flow_pixel = model(data.cuda())
    flow_pixel = flow_pixel.cpu().detach().numpy()
    flow_pixel = np.squeeze(flow_pixel)

    flow_final = arrange_predicted_flow(flow_pixel, args)
    results = get_dic_results(im1, im2, flow_orig, flow_final)
    print('EPE: ', '{:.4f}'.format(results['loss_pred']))
    print('EAE: ', '{:.4f}'.format(results['loss_ang_pred']))
    show_results(results, name, args)

    return results

@torch.no_grad()
def evaluate(model, args):
    list_predictions = []
    for name in args['training_IDs']:
        for acc in args['acc_list']:
            results = get_sample_results(model, name, acc, args)
            list_predictions.append(results)
    eval_img(list_predictions, show=True, save_path=args['save_path'])
    return list_predictions


if __name__ == '__main__':
    with open("/home/studghoul1/lapnet/lapnet/pytorch/config.yaml", 'r') as stream:
        args = yaml.safe_load(stream)
    # device configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = args['Setup']['gpu_num']

    model = LAPNet()
    model.load_state_dict(torch.load(args['Evaluate']['checkpoint_path'], map_location=args['Setup']['gpu'])['model_state_dict'])
    print(model)

    model.cuda()
    model.eval()

    with torch.no_grad():
        list_predictions = evaluate(model, args['Evaluate'])
