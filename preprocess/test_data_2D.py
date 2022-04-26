import os
import yaml
import numpy as np

import multiprocessing as mp
from functools import partial

from util import get_slice_info_from_ods_file, Map, read_fnames
from processing import subsample_img_drus, select_2d_data, respiratoryDataset, load_4d_img, cineDataset, \
    sliding_tapering, sliding_tapering_one_frame


def create_moco_data(args):
    data_paths = read_fnames(args.fnames_path)
    slice_info = get_slice_info_from_ods_file(args.info_file)
    for f_name in data_paths:
        img = load_4d_img(f_name, args.img_path)
        slice_info_fname = slice_info[f_name]
        frames = range(img.shape[-1])
        list_var = [(us, slice_num, frame) for us in args.list_us for slice_num in slice_info_fname for frame in frames]
        part_pool_func = partial(create_MoCo_data,
                                 ID=f_name,
                                 img=img,
                                 save_dir=args.saving_dir)
        pool = mp.Pool(args.workers)
        pool.starmap(part_pool_func, list_var)
        pool.close()
        pool.join()


def create_respiratory_test_data(config):
    data_paths = read_fnames(config.fnames_path)
    slice_info = get_slice_info_from_ods_file(config.info_file)
    print('start test data creation')
    dataset = respiratoryDataset(config)
    for f_name in data_paths:
        print(f'create {f_name} test data')
        for acc in config.list_us:
            _, _, ref, mov, u, us_rate = dataset.read_respiratory_data(f_name, 'real', acc)
            slice_info_fname = slice_info[f_name]
            part_pool_func = partial(create_resp_test_data,
                                     ID=f_name,
                                     ref=ref,
                                     mov=mov,
                                     flow=u,
                                     args=config)
            list_var = [(us, slice_num) for us in config.list_us for slice_num in slice_info_fname]
            print(len(list_var))
            pool = mp.Pool(config.workers)
            pool.starmap(part_pool_func, list_var)
            pool.close()
            pool.join()


def create_cardiac_test_data(config):
    data_paths = read_fnames(config.fnames_path)
    slice_info = get_slice_info_from_ods_file(config.info_file)

    list_var = [(ID, us) for ID in data_paths for us in config.list_us]
    part_pool_func = partial(create_cine_test_data,
                             slice_info=slice_info,
                             args=config)
    pool = mp.Pool(config.workers)
    pool.starmap(part_pool_func, list_var)
    pool.close()
    pool.join()

    print('end test data creation')


def create_cine_test_data(f_name, acc, slice_info, args):
    ID_slice_info = slice_info[f_name]
    dataset = cineDataset(args)
    for s in ID_slice_info:
        npz_saving_path = f'{args.save_dir}/{acc}/{f_name}_slice{s}.npz'
        os.makedirs(f'{args.save_dir}/{acc}', exist_ok=True)
        if not os.path.exists(npz_saving_path):
            k_ref, k_mov, k_u = dataset.read_cardiac_data(f_name, 'real', acc, s)

            # fig, ax = plt.subplots(2)
            # ax[0].imshow(np.abs(k_ref))
            # ax[1].imshow(np.abs(k_mov))
            # ax[0].axis('off')
            # ax[1].axis('off')
            # plt.show()

            sliding_tapering(k_ref, k_mov, k_u, npz_saving_path, args.tapering_size, args.crop_stride)


def create_resp_test_data(us_acc, slice_num, f_name, ref, mov, flow, args):
    saving_dir_tapered = f'{args.save_dir}/{us_acc}'
    print(f_name, us_acc)
    os.makedirs(saving_dir_tapered, exist_ok=True)
    npz_saving_path = f'{saving_dir_tapered}/{f_name}_slice{slice_num}.npz'
    if not os.path.exists(npz_saving_path):
        k_ref, k_mov, k_u = select_2d_data(ref, mov, flow, slice_num, 'coronal')
        print(f'{f_name}: create slice {slice_num} for acceleration {us_acc}')
        sliding_tapering(k_ref, k_mov, k_u, npz_saving_path, args.tapering_size, args.crop_stride)


def create_MoCo_data(us_acc, slice_num, frame, ID, img, save_dir, tapering_size=33, crop_stride=2):
    img_3d = img[..., frame]
    if us_acc != 1:
        img_3d = subsample_img_drus(img_3d, us_acc)
    img_2d = img_3d[..., slice_num]
    radius = int((tapering_size - 1) / 2)
    img_2d = np.pad(img_2d, (radius, radius), constant_values=0)
    os.makedirs(f'{save_dir}/{us_acc}/{ID}', exist_ok=True)
    save_path = f'{save_dir}/{us_acc}/{ID}/slice_{slice_num}frame_{frame}.npy'
    sliding_tapering_one_frame(img_2d, save_path, tapering_size, crop_stride)


if __name__ == '__main__':
    with open("/home/studghoul1/lapnet/lapnet/TF2/config.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    data_setup = data_loaded['Test']['test_data']
    args = Map(data_setup)
    create_respiratory_test_data(args)
