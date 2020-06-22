import os
import sys
import math
import time
import numpy as np
import tensorflow as tf
import random
import matplotlib
# matplotlib.use('pdf')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab
from ..core.input import read_png_image, Input, load_mat_file
from ..core.augment import random_crop
from ..core.flow_util import flow_to_color
from ..core.image_warp import np_warp_3D
from ..core.sampling import generate_mask
from ..core.sampling_center import sampleCenter
from ..core.util import pos_generation_2D, _u_generation_3D, arr2kspace
from e2eflow.core.flow_util import flow_to_color_np


def _read_flow(filenames, num_epochs=None):
    """Given a list of filenames, constructs a reader op for ground truth."""
    filename_queue = tf.train.string_input_producer(filenames,
        shuffle=False, capacity=len(filenames), num_epochs=num_epochs)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    gt_uint16 = tf.image.decode_png(value, dtype=tf.uint16)
    gt = tf.cast(gt_uint16, tf.float32)
    flow = (gt[:, :, 0:2] - 2 ** 15) / 64.0
    mask = gt[:, :, 2:3]
    return flow, mask


class MRI_Resp_2D(Input):
    def __init__(self, data, batch_size, dims, *,
                 num_threads=1, normalize=True,
                 skipped_frames=False):
        super().__init__(data, batch_size, dims, num_threads=num_threads,
                         normalize=normalize, skipped_frames=skipped_frames)

    def load_aug_data(self,
                      fn_im_paths,
                      slice_info,
                      aug_type,
                      amp=5,
                      mask_type='drUS',
                      US_rate='random',
                      num_to_take=1500):
        output = np.zeros((0, self.dims[0], self.dims[1], 4), dtype=np.float32)
        if num_to_take == 0:
            return output
        i = 0
        flag = 0
        if num_to_take == 'all':
            num_to_take = 10000000
            flag = 1
        while len(output) <= num_to_take:
            fn_im_path = fn_im_paths[i]
            # fn_im_path = '/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/train/volunteer_01_cw.npz'
            try:
                f = load_mat_file(fn_im_path)
            except:
                try:
                    f = np.load(fn_im_path)
                except ImportError:
                    print("Wrong Data Format")

            name = fn_im_path.split('/')[-1].split('.')[0]

            ref = np.asarray(f['dFixed'], dtype=np.float32)
            ux = np.asarray(f['ux'], dtype=np.float32)  # ux for warp
            uy = np.asarray(f['uy'], dtype=np.float32)
            uz = np.zeros(np.shape(ux), dtype=np.float32)
            u = np.stack((ux, uy, uz), axis=-1)

            if aug_type == 'real_x_smooth':
                u_syn = _u_generation_3D(np.shape(ux), amp, motion_type=1)
                u = np.multiply(u, u_syn)
            elif aug_type == 'smooth':
                u = _u_generation_3D(np.shape(ux), amp, motion_type=1)
            elif aug_type == 'constant':
                u = _u_generation_3D(np.shape(ux), amp, motion_type=0)
            elif aug_type == 'real':
                pass
            else:
                raise ImportError('wrong augmentation type is given')
            mov = np_warp_3D(ref, u)

            # # for showing of arrows
            # im1 = ref[..., 35]
            # im2 = mov[..., 35]
            # u = u[..., :2][..., 35, :]
            # save_imgs_with_arrow(im1, im2, u)

            if US_rate:
                if US_rate == 'random':
                    acc = np.random.choice(np.arange(1, 32, 6))
                else:
                    try:
                        acc = US_rate
                    except ImportError:
                        print("Wrong undersampling rate is given")
                        continue

                if mask_type == 'drUS':
                    mask = np.transpose(generate_mask(acc=acc, size_y=256, nRep=4), (2, 1, 0))
                elif mask_type == 'crUS':
                    mask = sampleCenter(1 / acc * 100, 256, 72)
                    mask = np.array([mask, ] * 4, dtype=np.float32)
                k_ref = np.multiply(np.fft.fftn(ref), np.fft.ifftshift(mask[0, ...]))
                k_mov = np.multiply(np.fft.fftn(mov), np.fft.ifftshift(mask[3, ...]))
                ref = (np.fft.ifftn(k_ref)).real
                mov = (np.fft.ifftn(k_mov)).real

            # save_img(mov[..., 40], '/home/jpa19/PycharmProjects/MA/UnFlow/output/temp/'+'us_mov30')
            # save_img(mask[3, ...], '/home/jpa19/PycharmProjects/MA/UnFlow/output/temp/' + 'mask_mov30')

            data_3D = np.stack((ref, mov, u[..., 0], u[..., 1]), axis=-1)
            data_3D = np.moveaxis(data_3D, 2, 0)
            slice2take = slice_info[name]
            Imgs = data_3D[slice2take, ...]

            # fig, ax = plt.subplots(1, 2, figsize=(12, 8))
            # plt.axis('off')
            # ax[0].imshow(Imgs[10,...,0], cmap='gray')
            # ax[1].imshow(Imgs[10,...,1], cmap='gray')
            # plt.show()


            output = np.concatenate((output, Imgs), axis=0)
            i += 1
            if i == len(fn_im_paths):
                if flag == 0:
                    i = 0
                else:
                    break
        print("{} real {} data are generated".format(num_to_take, aug_type))

        return np.asarray(output[:num_to_take, ...], dtype=np.float32)

    def input_train_data(self, img_dirs, slice_info, params, case='train'):
        # strategy 1: fixed total number
        if case == 'train':
            total_data_num = params.get('total_data_num')
        elif case == 'validation':
            total_data_num = 128
        num_constant = math.floor(total_data_num * params.get('augment_type_percent')[0])
        num_smooth = math.floor(total_data_num * params.get('augment_type_percent')[1])
        num_real = math.floor(total_data_num * params.get('augment_type_percent')[2])
        num_real_x_smooth = math.floor(total_data_num * params.get('augment_type_percent')[3])
        assert (num_real <= 1585 and case == 'train') or (num_real <= 136 and case == 'validation')

        # # strategy 2: use all real_simulated data
        # num_real = 1721
        # num_constant = int(params.get('augment_type_percent')[0]/params.get('augment_type_percent')[2]*num_real)
        # num_smooth = int(params.get('augment_type_percent')[1] / params.get('augment_type_percent')[2] * num_real)
        # num_real_x_smooth = int(params.get('augment_type_percent')[3] / params.get('augment_type_percent')[2] * num_real)


        batches = np.zeros((0, self.dims[0], self.dims[1], 4), dtype=np.float32)
        fn_im_paths = self.get_data_paths(img_dirs)
        aug_data_constant = self.load_aug_data(fn_im_paths,
                                               slice_info,
                                               aug_type='constant',
                                               amp=params.get('flow_amplitude'),
                                               mask_type=params.get('mask_type'),
                                               US_rate=(params.get('us_rate')),
                                               num_to_take=num_constant)

        np.random.shuffle(fn_im_paths)
        aug_data_smooth = self.load_aug_data(fn_im_paths,
                                             slice_info,
                                             aug_type='smooth',
                                             amp=params.get('flow_amplitude'),
                                             mask_type=params.get('mask_type'),
                                             US_rate=(params.get('us_rate')),
                                             num_to_take=num_smooth)
        # np.save('/home/jpa19/PycharmProjects/MA/UnFlow/hardcoded_data_USfalse', aug_data_smooth)
        np.random.shuffle(fn_im_paths)
        aug_data_real = self.load_aug_data(fn_im_paths,
                                           slice_info,
                                           aug_type='real',
                                           amp=params.get('flow_amplitude'),
                                           mask_type=params.get('mask_type'),
                                           US_rate=(params.get('us_rate')),
                                           num_to_take=num_real)

        np.random.shuffle(fn_im_paths)
        aug_data_real_x_smooth = self.load_aug_data(fn_im_paths,
                                                    slice_info,
                                                    aug_type='real_x_smooth',
                                                    amp=5,
                                                    mask_type=params.get('mask_type'),
                                                    US_rate=(params.get('us_rate')),
                                                    num_to_take=num_real_x_smooth)

        batches = np.concatenate((batches, aug_data_real, aug_data_constant, aug_data_smooth, aug_data_real_x_smooth), axis=0)
        np.random.shuffle(batches)
        if params.get('network') == 'ftflownet':
            radius = int((params.get('crop_size') - 1) / 2)
            if params.get('padding'):
                batches = np.pad(batches, ((0, 0), (radius, radius), (radius, radius), (0, 0)), constant_values=0)
            if params.get('random_crop'):
                batches = self.crop2D(batches, crop_size=params.get('crop_size'), box_num=params.get('crop_box_num'), cut_margin=20)
            else:
                x_dim, y_dim = np.shape(batches)[1:3]
                pos = pos_generation_2D(intervall=[[0, x_dim - params.get('crop_size') + 1],
                                                   [0, y_dim - params.get('crop_size') + 1]], stride=4)
                batches = self.crop2D_FixPts(batches, crop_size=params.get('crop_size'), box_num=np.shape(pos)[1], pos=pos)
            batches = np.concatenate((arr2kspace(batches[..., :2]), batches[..., 2:]), axis=-1)
            im1 = batches[..., :2]
            im2 = batches[..., 2:4]
            flow = batches[:, radius, radius, 4:6]
        elif params.get('network') == 'flownet':
            im1 = batches[..., 0]
            im2 = batches[..., 1]
            flow = batches[..., 2:]
        else:
            raise ImportError('Wrong Network name is given')
        return [im1, im2, flow]


# for original Kitti data
class KITTIInput(Input):
    def __init__(self, data, batch_size, dims, *,
                 num_threads=1, normalize=True,
                 skipped_frames=False):
        super().__init__(data, batch_size, dims, num_threads=num_threads,
                         normalize=normalize, skipped_frames=skipped_frames)

    def _preprocess_flow(self, gt):
        flow, mask = gt
        height, width = self.dims
        # Reshape to tell tensorflow we know the size statically
        flow = tf.reshape(self._resize_crop_or_pad(flow), [height, width, 2])
        mask = tf.reshape(self._resize_crop_or_pad(mask), [height, width, 1])
        return flow, mask

    def _input_flow(self, flow_dir, hold_out_inv):
        flow_dir_occ = os.path.join(self.data.current_dir, flow_dir, 'flow_occ')
        flow_dir_noc = os.path.join(self.data.current_dir, flow_dir, 'flow_noc')
        filenames_gt_occ = []
        filenames_gt_noc = []
        flow_files_occ = os.listdir(flow_dir_occ)
        flow_files_occ.sort()
        flow_files_noc = os.listdir(flow_dir_noc)
        flow_files_noc.sort()

        if hold_out_inv is not None:
            random.seed(0)
            random.shuffle(flow_files_noc)
            flow_files_noc = flow_files_noc[:hold_out_inv]

            random.seed(0)
            random.shuffle(flow_files_occ)
            flow_files_occ = flow_files_occ[:hold_out_inv]

        assert len(flow_files_noc) == len(flow_files_occ)

        for i in range(len(flow_files_occ)):
            filenames_gt_occ.append(os.path.join(flow_dir_occ,
                                                 flow_files_occ[i]))
            filenames_gt_noc.append(os.path.join(flow_dir_noc,
                                                 flow_files_noc[i]))

        flow_occ, mask_occ = self._preprocess_flow(
            _read_flow(filenames_gt_occ, 1))
        flow_noc, mask_noc = self._preprocess_flow(
            _read_flow(filenames_gt_noc, 1))
        return flow_occ, mask_occ, flow_noc, mask_noc

    def _input_train(self, image_dir, flow_dir, hold_out_inv=None):
        input_shape, im1, im2 = self._input_images(image_dir, hold_out_inv)
        flow_occ, mask_occ, flow_noc, mask_noc = self._input_flow(flow_dir, hold_out_inv)
        return tf.train.batch(
            [im1, im2, input_shape, flow_occ, mask_occ, flow_noc, mask_noc],
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            allow_smaller_final_batch=True)

    def input_train_gt(self, hold_out):
        img_dirs = ['data_scene_flow/training/image_2',
                    'data_stereo_flow/training/colored_0']
        gt_dirs = ['data_scene_flow/training/flow_occ',
                   'data_stereo_flow/training/flow_occ']

        height, width = self.dims

        filenames = []
        for img_dir, gt_dir in zip(img_dirs, gt_dirs):
            dataset_filenames = []
            img_dir = os.path.join(self.data.current_dir, img_dir)
            gt_dir = os.path.join(self.data.current_dir, gt_dir)
            img_files = os.listdir(img_dir)
            gt_files = os.listdir(gt_dir)
            img_files.sort()
            gt_files.sort()
            assert len(img_files) % 2 == 0 and len(img_files) / 2 == len(gt_files)

            for i in range(len(gt_files)):
                fn_im1 = os.path.join(img_dir, img_files[2 * i])
                fn_im2 = os.path.join(img_dir, img_files[2 * i + 1])
                fn_gt = os.path.join(gt_dir, gt_files[i])
                dataset_filenames.append((fn_im1, fn_im2, fn_gt))

            random.seed(0)
            random.shuffle(dataset_filenames)
            dataset_filenames = dataset_filenames[hold_out:]
            filenames.extend(dataset_filenames)

        random.seed(0)
        random.shuffle(filenames)

        #shift = shift % len(filenames)
        #filenames_ = list(np.roll(filenames, shift))

        fns_im1, fns_im2, fns_gt = zip(*filenames)
        fns_im1 = list(fns_im1)
        fns_im2 = list(fns_im2)
        fns_gt = list(fns_gt)

        im1 = read_png_image(fns_im1)
        im2 = read_png_image(fns_im2)
        flow_gt, mask_gt = _read_flow(fns_gt)

        gt_queue = tf.train.string_input_producer(fns_gt,
            shuffle=False, capacity=len(fns_gt), num_epochs=None)
        reader = tf.WholeFileReader()
        _, gt_value = reader.read(gt_queue)
        gt_uint16 = tf.image.decode_png(gt_value, dtype=tf.uint16)
        gt = tf.cast(gt_uint16, tf.float32)

        im1, im2, gt = random_crop([im1, im2, gt],
                                   [height, width, 3])
        flow_gt = (gt[:, :, 0:2] - 2 ** 15) / 64.0
        mask_gt = gt[:, :, 2:3]

        if self.normalize:
            im1 = self._normalize_image(im1)
            im2 = self._normalize_image(im2)

        return tf.train.batch(
            [im1, im2, flow_gt, mask_gt],
            batch_size=self.batch_size,
            num_threads=self.num_threads)

    def input_train_2015(self, hold_out_inv=None):
        return self._input_train('data_scene_flow/training/image_2',
                                 'data_scene_flow/training',
                                 hold_out_inv)

    def input_test_2015(self, hold_out_inv=None):
        return self._input_test('data_scene_flow/testing/image_2', hold_out_inv)

    def input_train_2012(self, hold_out_inv=None):
        return self._input_train('data_stereo_flow/training/colored_0',
                                 'data_stereo_flow/training',
                                 hold_out_inv)

    def input_test_2012(self, hold_out_inv=None):
        return self._input_test('data_stereo_flow/testing/colored_0', hold_out_inv)
