import torch
import os
from torch.utils.data import DataLoader
from random import shuffle
import numpy as np
from torch.utils.data import DataLoader
from core import ifft_along_dim, fftnshift, np_fftconvolve, flowCrop
from scipy.interpolate import interpn
import random
import json
import glob


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        dict = {}
        for key in sample:
            tmp_tensor = torch.from_numpy(sample[key]).float()
            dict[key] = tmp_tensor
        return dict


def fetch_dataloader(args):
    data_num = 128 if args.debug else 1500000
    train_dataset = RespDataset(args.data_path, transform=ToTensor(), training_mode=args.training_mode,
                                branches=args.branches, data_num=data_num)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              pin_memory=True, shuffle=True, num_workers=args.num_workers, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader


class RespDatasetPreprocessing(torch.utils.data.Dataset):
    def __init__(self, dataset_path, patch_size=33, transform=None, training_mode='self_supervised', branches=False):
        self.dataset_path = dataset_path
        self.patch_size = patch_size
        self.patch_ray = int((self.patch_size - 1) / 2)
        self.z_dim_dict = self.read_z_dim()
        self.list_IDs = self.get_paths_list()
        self.k_win, self.padded_arr = self.create_window_and_arr()
        self.training_mode = training_mode
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def read_z_dim(self):
        path_z_dim_file = '/home/studghoul1/dataset/Resp/self_supervised/z_dim_file.txt'
        with open(path_z_dim_file) as f:
            d = json.load(f)
        return d

    def create_window_and_arr(self):
        square = 256
        dim = 2 * square - self.patch_size
        padded_arr = np.zeros((2, dim, dim), dtype=np.complex128)
        window_img = np.zeros((dim, dim), dtype=np.float32)
        pos_win = int(dim / 2) - self.patch_ray
        window_img[pos_win:pos_win + self.patch_size, pos_win:pos_win + self.patch_size] = 1
        k_win = fftnshift(window_img)
        return k_win, padded_arr

    def regrid_reshuffle(self, arr1, arr2):
        # regrid
        rows, cols = arr1.shape
        grid_x, grid_y = np.mgrid[0:rows], np.mgrid[0:cols]
        x = np.linspace(0, rows - 1, num=self.patch_size)
        y = np.linspace(0, cols - 1, num=self.patch_size)
        grid_xq, grid_yq = np.meshgrid(x, y)
        arr1 = interpn((grid_x, grid_y), arr1, (grid_yq, grid_xq), method='linear')
        arr2 = interpn((grid_x, grid_y), arr2, (grid_yq, grid_xq), method='linear')

        # shuffle
        arr1[::2, :] = -arr1[::2, :]
        arr1[:, ::2] = -arr1[:, ::2]
        arr2[::2, :] = -arr2[::2, :]
        arr2[:, ::2] = -arr2[:, ::2]
        return arr1, arr2

    def taper(self, arr, u=None):
        x_dim, y_dim = arr.shape[1:]
        posx = random.randint(0, x_dim - self.patch_size)
        posy = random.randint(0, y_dim - self.patch_size)

        if x_dim == 256 and y_dim == 256:
            k_win = self.k_win
            padded_arr = self.padded_arr
            dim = 2 * 256 - self.patch_size
        else:
            square = x_dim if y_dim < x_dim else y_dim
            dim = 2 * square - self.patch_size
            padded_arr = np.zeros((2, dim, dim), dtype=np.complex128)
            window_img = np.zeros((dim, dim), dtype=np.float32)
            pos_win = int(dim / 2) - self.patch_ray
            window_img[pos_win:pos_win + self.patch_size, pos_win:pos_win + self.patch_size] = 1
            k_win = fftnshift(window_img)

        ray = int(dim / 2)
        new_x_pos = ray - posx - self.patch_ray
        new_y_pos = ray - posy - self.patch_ray
        padded_arr[:, new_x_pos:new_x_pos + x_dim, new_y_pos:new_y_pos + y_dim] = arr

        k_ref_full = fftnshift(padded_arr[0, ...])
        k_mov_full = fftnshift(padded_arr[1, ...])

        k_ref_win = np_fftconvolve(k_ref_full, k_win)
        k_mov_win = np_fftconvolve(k_mov_full, k_win)
        k_ref, k_mov = self.regrid_reshuffle(k_ref_win, k_mov_win)
        k_space = np.stack((np.real(k_ref), np.imag(k_ref), np.real(k_mov), np.imag(k_mov)), axis=0)
        if u:
            flow = flowCrop(u, posx, posy, self.patch_size)
            return {'k_space': k_space, 'flow': flow}
        else:
            img_ref = ifft_along_dim(k_ref)
            img_mov = ifft_along_dim(k_mov)
            img_ref_stack = np.stack((np.real(img_ref), np.imag(img_ref)), axis=0)
            img_mov_stack = np.stack((np.real(img_mov), np.imag(img_mov)), axis=0)

            return {'k_space': k_space, 'img_mov': img_mov_stack, 'img_ref': img_ref_stack}

    def get_paths_list(self):
        IDS_list = []
        list_dir = [x[0] for x in os.walk(self.dataset_path)][1:]
        print(list_dir)

        for i, mypath in enumerate(list_dir):
            IDs = [(f'{mypath}/{item}', int(z), patch) for item in os.listdir(mypath) for z in
                   range(self.z_dim_dict[item]) for patch in range(200)]
            IDS_list += IDs
            shuffle(IDS_list)

        shuffle(IDS_list)

        return IDS_list

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_path = self.list_IDs[idx][0]
        z_dim = self.list_IDs[idx][1]
        sample_data = np.load(sample_path, mmap_mode='r')

        if self.training_mode == 'supervised':
            # t = time.time()
            full_img = sample_data[''][..., z_dim]
            flow_2D = sample_data['flow']
            sample = self.taper(full_img, flow_2D)
            # print(time.time()-t)

        elif self.training_mode == 'self_supervised':
            # t = time.time()
            full_img = sample_data[..., z_dim]
            sample = self.taper(full_img)
            # print(time.time() - t)

        if self.transform:
            sample = self.transform(sample)
        return sample


class RespDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, box_num=200, branches=False, transform=None, training_mode='supervised',
                 data_num=200):
        """
        Args:
            dataset_path (string): Directory to the images
            transform (callable, optional): Optional transform to be applied
                on a sample.
            training_mode (string): supervised or self_supervised
        """
        self.dataset_path = dataset_path
        self.box_num = box_num
        self.branches = branches
        self.data_num = data_num
        self.training_mode = training_mode
        self.transform = transform
        self.list_IDs = self.get_paths_list()

    def get_paths_list(self):
        num_real, num_realxsmooth, num_smooth = 300000, 560000, 560000
        num_list = [num_real, num_smooth, num_realxsmooth]
        res = []
        list_dir = [x[0] for x in os.walk(self.dataset_path)][1:]
        print(list_dir)
        for i, mypath in enumerate(list_dir):
            list_tmp = glob.glob(f'{mypath}/*.npz')
            shuffle(list_tmp)
            res += list_tmp #[:num_list[i]]
            shuffle(res)
        return res[:self.data_num]

    def __len__(self):
        return len(self.list_IDs)

    def normalize_img(self, img):
        if (np.max(img) - np.min(img)) == 0:
            return img
        else:
            return (img - np.min(img)) / (np.max(img) - np.min(img))

    def normalize_k_space(self, k):
        img = ifft_along_dim(k[..., 0] + 1j * k[..., 1])
        max = np.max(np.abs(img))
        if max == 0:
            max = 1
        return k / max

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        try:
            data = np.load(self.list_IDs[idx], mmap_mode='r')
        except:
            print(self.list_IDs[idx], 'cannot be loaded')
            data = {'k_space': np.zeros((33, 33, 4, 3), dtype=np.float32),
                    'k_full': np.zeros((33, 33, 4, 3), dtype=np.float32),
                    'flow': np.zeros((3,), dtype=np.float32)}

        flow = data['flow']

        if self.training_mode == 'self_supervised':
            k_full = data['k_full']
            k_cor_full = k_full  # k_full[..., 0]
            ref_coronal = np.abs(ifft_along_dim(k_cor_full[..., 0] + 1j * k_cor_full[..., 1]))[None, ...]
            mov_coronal = np.abs(ifft_along_dim(k_cor_full[..., 2] + 1j * k_cor_full[..., 3]))[None, ...]
            ref_coronal = self.normalize_img(ref_coronal)
            mov_coronal = self.normalize_img(mov_coronal)

        if self.branches:
            if self.training_mode == 'self_supervised':
                k_cor = self.normalize_k_space(data['k_space'][..., 0])
                k_sag = self.normalize_k_space(data['k_space'][..., 1])
                k_cor_us = np.transpose(k_cor, (2, 1, 0))
                k_sag_us = np.transpose(k_sag, (2, 1, 0))

                k_sag_full = k_full[..., 1]
                ref_sagittal = np.abs(ifft_along_dim(k_sag_full[..., 0] + 1j * k_sag_full[..., 1]))[None, ...]
                mov_sagittal = np.abs(ifft_along_dim(k_sag_full[..., 2] + 1j * k_sag_full[..., 3]))[None, ...]
                ref_sagittal = self.normalize_img(ref_sagittal)
                mov_sagittal = self.normalize_img(mov_sagittal)

                sample = {'k_coronal': k_cor_us, 'k_sagittal': k_sag_us,
                          'mov_c': mov_coronal, 'ref_c': ref_coronal,
                          'mov_s': mov_sagittal, 'ref_s': ref_sagittal,
                          'flow': flow}

            elif self.training_mode == 'supervised':
                k_out = data['k_space']
                k_cor = k_out[..., 0]  # coronal
                k_sag = k_out[..., 1]  # sagittal
                k_cor = np.transpose(k_cor, (2, 1, 0))
                k_sag = np.transpose(k_sag, (2, 1, 0))
                flow = data['flow']
                sample = {'k_coronal': k_cor, 'k_sagittal': k_sag, 'flow': flow}

        else:
            k_out = self.normalize_k_space(data['k_space'])  # [..., 0])
            k_out = np.transpose(k_out, (2, 1, 0))
            flow = flow[:2]
            if self.training_mode == 'self_supervised':
                # flow = np.flip(flow, axis=0).copy()
                sample = {'k_space': k_out, 'img_mov': mov_coronal, 'img_ref': ref_coronal, 'flow': flow}

            elif self.training_mode == 'supervised':
                sample = {'k_space': k_out, 'flow': flow}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    print(1)
    BATCH_SIZE = 1
    DATA_PATH = '/home/studghoul1/dataset/Resp/self_supervised/fully_sampled'
    transformed_dataset = RespDatasetPreprocessing(DATA_PATH)  # , transform=ToTensor())
    train_dataloader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1,
                                  pin_memory=True)
    x = next(iter(train_dataloader))
    for e in x:
        print(x[e].shape)

    # save z dims in txt file with names
    # run training

    # BATCH_SIZE = 64
    # DATA_PATH = '/home/studghoul1/Documents/research_thesis_data_results/data/3D_dataset'
    # transformed_dataset = RespDatasetPreprocessing(DATA_PATH, transform=ToTensor(),  branches=True)
    # train_dataloader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
    # x = next(iter(train_dataloader))
