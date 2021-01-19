import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.util.shape import view_as_windows
from ..core.util import pos_generation_2D, arr2kspace, load_mat_file, fftnshift, rectangulartapering2d, flowCrop
from ..core.image_warp import np_warp_3D
from e2eflow.core.resp_US.sampling_center import sampleCenter
from ..core.card_US.retrospective_radial import subsample_radial
from ..core.card_US.pad_crop import post_crop
from .augment import random_crop


def resize_input(t, height, width, resized_h, resized_w):
    # Undo old resizing and apply bilinear
    t = tf.reshape(t, [resized_h, resized_w, 3])
    t = tf.expand_dims(tf.image.resize_image_with_crop_or_pad(t, height, width), 0)
    return tf.image.resize_bilinear(t, [resized_h, resized_w])


def resize_output_crop(t, height, width, channels):
    _, oldh, oldw, c = tf.unstack(tf.shape(t))
    t = tf.reshape(t, [oldh, oldw, c])
    t = tf.image.resize_image_with_crop_or_pad(t, height, width)
    return tf.reshape(t, [1, height, width, channels])


def resize_output(t, height, width, channels):
    return tf.image.resize_bilinear(t, [height, width])


def resize_output_flow(t, height, width, channels):
    batch, old_height, old_width, _ = tf.unstack(tf.shape(t), num=4)
    t = tf.image.resize_bilinear(t, [height, width])
    u, v = tf.unstack(t, axis=3)
    u *= tf.cast(width, tf.float32) / tf.cast(old_width, tf.float32)
    v *= tf.cast(height, tf.float32) / tf.cast(old_height, tf.float32)
    return tf.reshape(tf.stack([u, v], axis=3), [batch, height, width, 2])


def frame_name_to_num(name):
    stripped = name.split('.')[0].lstrip('0')
    if stripped == '':
        return 0
    return int(stripped)


class Input:

    def __init__(self, data, batch_size, dims, *,
                 num_threads=1, normalize=True,
                 skipped_frames=False):
        assert len(dims) == 2
        self.data = data
        self.dims = dims
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.normalize = normalize
        self.skipped_frames = skipped_frames

    def get_data_paths(self, img_dirs):
        fn_im_paths = []
        if isinstance(img_dirs, str):
            img_dirs = [img_dirs]
        for img_dir in img_dirs:
            if os.path.isfile(img_dir):
                fn_im_paths.append(img_dir)
                continue
            img_dir = os.path.join(self.data.current_dir, img_dir)
            img_files = os.listdir(img_dir)
            for img_file in img_files:
                if ('.mat' in img_file) or ('.npz' in img_file):
                    im_path = os.path.join(img_dir, img_file)
                    fn_im_paths.append(im_path)
                else:
                    if not img_file.startswith('.'):
                        try:
                            img_mat = os.listdir(os.path.join(img_dir, img_file))[0]
                        except Exception:
                            print("File {} is empty!".format(img_file))
                            continue
                        fn_im_paths.append(os.path.join(img_dir, img_file, img_mat))
        return fn_im_paths

    def crop2D_FixPts(self, arr, crop_size, box_num, pos):
        """
        :param arr: shape:[batch_size, img_size, img_size, n]
        :param crop_size:
        :param box_num:
        :param pos:
        :return:
        """
        arr_cropped_augmented = np.zeros((np.shape(arr)[0] * box_num, crop_size, crop_size, np.shape(arr)[-1]),
                                         dtype=np.float32)
        # if len(arr.shape()) is 4:
        #     arr = arr[0, ...]
        # elif len(arr.shape()) is 2:
        #     arr = arr[..., np.newaxis]
        # elif len(arr.shape()) is 3:
        #     pass
        # else:
        #     raise ImportError
        for batch in range(np.shape(arr)[0]):
            for i in range(box_num):
                arr_cropped_augmented[batch * box_num + i, ...] = arr[batch,
                                                                      pos[0][i]:pos[0][i] + crop_size,
                                                                      pos[1][i]:pos[1][i] + crop_size,
                                                                      :]

        return arr_cropped_augmented

    def taper2D_FixPts(self, arr, crop_size, box_num, pos):
        """
        :param arr: shape:[batch_size, img_size, img_size, n]
        :param crop_size:
        :param box_num:
        :param pos:
        :return:
        """
        radius = int((crop_size - 1) / 2)
        arr_tappered_augmented = np.zeros((arr.shape[0] * box_num, crop_size, crop_size, 4), dtype=np.float32)
        flow_cropped_augmented = np.zeros((arr.shape[0] * box_num, crop_size, crop_size, 2), dtype=np.float32)
        img = arr[..., :2]
        flow = arr[..., 2:]
        k_arr = fftnshift(img)
        for i in range(box_num):
            x_pos = np.arange(pos[0][i], pos[0][i] + crop_size)
            y_pos = np.arange(pos[1][i], pos[1][i] + crop_size)
            for k in range(2):
                outk = rectangulartapering2d(k_arr[:, :, :, k], x_pos, y_pos, crop_size)
                arr_tappered_augmented[i * arr.shape[0]:(i + 1) * arr.shape[0], :, :, 2 * k:2 + 2 * k] = outk

        for batch in range(np.shape(arr)[0]):
            for i in range(box_num):
                flow_cropped_augmented[batch * box_num + i, ...] = flow[batch, pos[0][i]:pos[0][i] + crop_size, pos[1][i]:pos[1][i] + crop_size, :]
        # flow = flow_cropped_augmented[:, radius, radius, 4:6]
        res = np.concatenate((arr_tappered_augmented, flow_cropped_augmented), axis=-1)
        # k_ref = arr_tappered_augmented[..., :2]
        # k_mov = arr_tappered_augmented[..., 2:4]
        # return (k_ref, k_mov, flow)
        return res

    def crop2D(self, arr, crop_size, box_num, pos=None, cut_margin=0):
        """
        :param arr:
        :param crop_size:
        :param box_num: crops per slices
        :param pos: shape (2, n), pos_x and pos_y, n must be same as the box number. if pos=None, select random pos
        :param cut_margin: margin that don't take into account
        :return:
        """
        arr_cropped_augmented = np.zeros((arr.shape[0] * box_num, crop_size, crop_size, arr.shape[-1]), dtype=np.float32)
        x_dim, y_dim = np.shape(arr)[1:3]
        for i in range(box_num):
            if pos is None:
                x_pos = np.random.randint(0 + cut_margin, x_dim - crop_size + 1 - cut_margin, arr.shape[0])
                y_pos = np.random.randint(0 + cut_margin, y_dim - crop_size + 1 - cut_margin, arr.shape[0])
            else:
                x_pos = pos[0]
                y_pos = pos[1]

            w = view_as_windows(arr, (1, crop_size, crop_size, 1))[..., 0, :, :, 0]
            out = w[np.arange(arr.shape[0]), x_pos, y_pos]
            out = out.transpose(0, 2, 3, 1)
            arr_cropped_augmented[i * arr.shape[0]:(i + 1) * arr.shape[0], ...] = out
        return arr_cropped_augmented

    def taper2D(self, arr, crop_size, box_num, pos=None, cut_margin=0):
        arr_tappered_augmented = np.zeros((arr.shape[0] * box_num, crop_size, crop_size, 4), dtype=np.float32)
        flow_cropped_augmented = np.zeros((arr.shape[0] * box_num, crop_size, crop_size, 2), dtype=np.float32)
        # flow = np.zeros((arr.shape[0] * box_num, 2), dtype=np.float32)
        img = arr[..., :2]
        flow_arr = arr[..., 2:]
        x_dim, y_dim = np.shape(arr)[1:3]
        k_arr = fftnshift(img)
        for i in range(box_num):
            if pos is None:
                x_pos = np.random.randint(0 + cut_margin, x_dim - crop_size + 1 - cut_margin, arr.shape[0])
                y_pos = np.random.randint(0 + cut_margin, y_dim - crop_size + 1 - cut_margin, arr.shape[0])
            else:
                x_pos = pos[0]
                y_pos = pos[1]
            for k in range(2):
                outk = rectangulartapering2d(k_arr[:, :, :, k], x_pos, y_pos, crop_size)
                arr_tappered_augmented[i * arr.shape[0]:(i + 1) * arr.shape[0], :, :, 2 * k:2 + 2 * k] = outk
            # p = flowCrop(flow_arr, x_pos, y_pos, crop_size)
            # flow[i * arr.shape[0]:(i + 1) * arr.shape[0], :] = p
            win = view_as_windows(flow_arr, (1, crop_size, crop_size, 1))[..., 0, :, :, 0]
            outflow = win[np.arange(arr.shape[0]), x_pos, y_pos]
            outflow = outflow.transpose(0, 2, 3, 1)
            flow_cropped_augmented[i * arr.shape[0]:(i + 1) * arr.shape[0], ...] = outflow

        # x = arr_tappered_augmented[0, :, :, 0] + 1j * arr_tappered_augmented[0, :, :, 1]
        # plottaperedkspace(x)
        res = np.concatenate((arr_tappered_augmented, flow_cropped_augmented), axis=-1)
        # k_ref = arr_tappered_augmented[..., :2]
        # k_mov = arr_tappered_augmented[..., 2:4]
        # return (k_ref, k_mov, flow)
        return res

    def test_flown(self, config):
        batches = self.test_set_generation_flownet(config)
        if len(batches.shape) == 3:
            batches = batches[np.newaxis, ...]
        im1_queue = tf.train.slice_input_producer([batches[..., 0]], shuffle=False,
                                                  capacity=len(list(batches[..., 0])), num_epochs=None)
        im2_queue = tf.train.slice_input_producer([batches[..., 1]], shuffle=False,
                                                  capacity=len(list(batches[..., 1])), num_epochs=None)
        flow_queue = tf.train.slice_input_producer([batches[..., 2:]], shuffle=False,
                                                   capacity=len(list(batches[..., 2:4])), num_epochs=None)
        # num_queue = tf.train.slice_input_producer([patient_num], shuffle=False,
        #                                            capacity=len(list(patient_num)), num_epochs=None)

        im1 = batches[..., 0]
        im2 = batches[..., 1]
        flow_orig = batches[..., 2:4]
        test_batches = tf.train.batch([im1_queue, im2_queue, flow_queue],
                                      batch_size=min(self.batch_size, np.shape(batches)[1]),
                                      num_threads=self.num_threads,
                                      allow_smaller_final_batch=True)

        return test_batches, im1, im2, flow_orig

    def test_lapnet(self, config):

        # batch = self.old_test_set_generation(config)
        batch = self.test_set_generation(config)
        batch = batch[np.newaxis, ...]

        radius = int((config['crop_size'] - 1) / 2)
        if config['padding']:
            batch = np.pad(batch, ((0, 0), (radius, radius), (radius, radius), (0, 0)), constant_values=0)
        x_dim, y_dim = np.shape(batch)[1:3]
        pos = pos_generation_2D(intervall=[[0, x_dim - config['crop_size'] + 1],
                                           [0, y_dim - config['crop_size'] + 1]], stride=config['crop_stride'])

        batches_cp = self.crop2D_FixPts(batch, crop_size=config['crop_size'], box_num=np.shape(pos)[1], pos=pos)
        batches_cp = np.concatenate((arr2kspace(batches_cp[..., :2]), batches_cp[..., 2:4]), axis=-1)
        flow = batches_cp[:, radius, radius, 4:6]

        im1_patch_k_queue = tf.train.slice_input_producer([batches_cp[..., :2]], shuffle=False,
                                                          capacity=len(list(batches_cp[..., 0])), num_epochs=None)
        im2_patch_k_queue = tf.train.slice_input_producer([batches_cp[..., 2:4]], shuffle=False,
                                                          capacity=len(list(batches_cp[..., 1])), num_epochs=None)
        flow_patch_gt = tf.train.slice_input_producer([flow], shuffle=False,
                                                      capacity=len(flow), num_epochs=None)

        im1 = batch[..., 0]
        im2 = batch[..., 1]
        flow_orig = batch[..., 2:4]

        test_batch = tf.train.batch([im1_patch_k_queue, im2_patch_k_queue, flow_patch_gt],
                                    batch_size=self.batch_size,
                                    num_threads=self.num_threads)

        return test_batch, im1, im2, flow_orig, np.transpose(pos)

    def test_set_generation(self, config):
        path = config['path']
        slice = config['slice']
        u_type = config['u_type']
        US_acc = config['US_acc']
        # data_type = config['data']
        mask_type = config['mask_type']
        if 'size' in config:
            cropped_size = config['size']
        else:
            cropped_size = False

        try:
            f = load_mat_file(path)
        except:
            try:
                f = np.load(path)
            except ImportError:
                print("Wrong Data Format")

        ref = np.asarray(f['dFixed'], dtype=np.float32)
        ux = np.asarray(f['ux'], dtype=np.float32)  # ux for warp
        uy = np.asarray(f['uy'], dtype=np.float32)
        uz = np.zeros(np.shape(ux), dtype=np.float32)
        if (self.dims[1] != np.shape(ref)[1]) or (self.dims[0] != np.shape(ref)[0]):
            pad_size_x = int((self.dims[0] - np.shape(ref)[0]) / 2)
            pad_size_y = int((self.dims[1] - np.shape(ref)[1]) / 2)

        u = np.stack((ux, uy, uz), axis=-1)

        if u_type == 3:
            u_syn = np.load('/home/jpa19/PycharmProjects/MA/UnFlow/data/u_3D/u_smooth_apt10_3D.npy')
            u = np.multiply(u, u_syn)
        elif u_type == 1:
            u = np.load('/home/jpa19/PycharmProjects/MA/UnFlow/data/u_3D/u_smooth_apt10_3D.npy')
        elif u_type == 0:
            u = np.load('/home/jpa19/PycharmProjects/MA/UnFlow/data/u_3D/u_constant_amp10_3D.npy')
        elif u_type == 2:
            pass
        else:
            raise ImportError('wrong augmentation type is given')
        try:
            mov = np_warp_3D(ref, u)
        except:
            u = u[::np.shape(ref)[0], ::np.shape(ref)[1], :np.shape(ref)[-1]]
            mov = np_warp_3D(ref, u)
        if US_acc > 1:
            if mask_type == 'drUS':
                mask = np.load('/home/jpa19/PycharmProjects/MA/UnFlow/data/mask/mask_acc{}.npy'.format(US_acc))
            elif mask_type == 'crUS':
                mask = sampleCenter(1/US_acc*100, 256, 72)
                mask = np.array([mask, ] * 4, dtype=np.float32)
            elif mask_type == 'radial':
                im_pair = np.stack((ref, mov), axis=-1)[..., slice, :]
                u = u[..., slice, :][..., :2]
                im_pair_US = np.squeeze(subsample_radial(im_pair[..., np.newaxis, :], acc=US_acc))
                im_pair = np.absolute(post_crop(im_pair_US, np.shape(im_pair)))
                # im_pair = np.absolute(im_pair_US)
                data = np.concatenate((im_pair, u), axis=-1)
                data = np.pad(data, ((pad_size_x, pad_size_x), (pad_size_y, pad_size_y), (0, 0)), constant_values=0)
                if cropped_size:
                    data = post_crop(data, (cropped_size[0], cropped_size[1], 4))
                return np.asarray(data, dtype=np.float32)

            k_dset = np.multiply(np.fft.fftn(ref), np.fft.ifftshift(mask[0, ...]))
            k_warped_dset = np.multiply(np.fft.fftn(mov), np.fft.ifftshift(mask[3, ...]))
            ref = (np.fft.ifftn(k_dset)).real
            mov = (np.fft.ifftn(k_warped_dset)).real

        data_3D = np.stack((ref, mov, u[..., 0], u[..., 1]), axis=-1)
        data_3D = np.moveaxis(data_3D, 2, 0)

        Imgs = data_3D[slice, ...]
        if list(np.shape(Imgs)[:2]) != cropped_size and cropped_size:
            Imgs = post_crop(Imgs, (cropped_size[0], cropped_size[1], 4))
        return np.asarray(Imgs, dtype=np.float32)

    def test_set_generation_flownet(self, config):
        path = config['path']
        slice = config['slice']
        u_type = config['u_type']
        US_acc = config['US_acc']
        data_type = config['data']
        mask_type = config['mask_type']

        try:
            f = load_mat_file(path)
        except:
            try:
                f = np.load(path)
            except ImportError:
                print("Wrong Data Format")

        ref = np.asarray(f['dFixed'], dtype=np.float32)
        ux = np.asarray(f['ux'], dtype=np.float32)  # ux for warp
        uy = np.asarray(f['uy'], dtype=np.float32)
        uz = np.zeros(np.shape(ux), dtype=np.float32)
        if (self.dims[1] != np.shape(ref)[1]) or (self.dims[0] != np.shape(ref)[0]):
            pad_size_x = int((self.dims[0] - np.shape(ref)[0]) / 2)
            pad_size_y = int((self.dims[1] - np.shape(ref)[1]) / 2)

        u = np.stack((ux, uy, uz), axis=-1)

        if u_type == 3:
            u_syn = np.load('/home/jpa19/PycharmProjects/MA/UnFlow/data/u_3D/u_smooth_apt10_3D.npy')
            u = np.multiply(u, u_syn)
        elif u_type == 1:
            u = np.load('/home/jpa19/PycharmProjects/MA/UnFlow/data/u_3D/u_smooth_apt10_3D.npy')
        elif u_type == 0:
            u = np.load('/home/jpa19/PycharmProjects/MA/UnFlow/data/u_3D/u_constant_amp10_3D.npy')
        elif u_type == 2:
            pass
        else:
            raise ImportError('wrong augmentation type is given')
        try:
            mov = np_warp_3D(ref, u)
        except:
            u = u[:192, :192, :np.shape(ref)[-1]]
            mov = np_warp_3D(ref, u)
        if US_acc > 1:
            if mask_type == 'drUS':
                mask = np.load('/home/jpa19/PycharmProjects/MA/UnFlow/data/mask/mask_acc{}.npy'.format(US_acc))
            elif mask_type == 'crUS':
                mask = sampleCenter(1/US_acc*100, 256, 72)
                mask = np.array([mask, ] * 4, dtype=np.float32)
            elif mask_type == 'radial':
                im_pair = np.stack((ref, mov), axis=-1)[..., slice, :]
                u = u[..., slice, :][..., :2]
                output = np.zeros((0, self.dims[0], self.dims[1], 4), dtype=np.float32)
                for i in range(len(slice)):
                    im_pair_single = im_pair[..., i, :]
                    u_single = u[..., i, :]
                    im_pair_US = np.squeeze(subsample_radial(im_pair_single[..., np.newaxis, :], acc=US_acc))
                    im_pair_single = np.absolute(post_crop(im_pair_US, np.shape(im_pair_single)))

                    data = np.concatenate((im_pair_single, u_single), axis=-1)
                    data = np.pad(data, ((pad_size_x, pad_size_x), (pad_size_y, pad_size_y), (0, 0)), constant_values=0)
                    output = np.concatenate((output, data[np.newaxis, ...]), axis=0)
                return np.asarray(output, dtype=np.float32)

            k_dset = np.multiply(np.fft.fftn(ref), np.fft.ifftshift(mask[0, ...]))
            k_warped_dset = np.multiply(np.fft.fftn(mov), np.fft.ifftshift(mask[3, ...]))
            ref = (np.fft.ifftn(k_dset)).real
            mov = (np.fft.ifftn(k_warped_dset)).real

        data_3D = np.stack((ref, mov, u[..., 0], u[..., 1]), axis=-1)
        data_3D = np.moveaxis(data_3D, 2, 0)

        Imgs = data_3D[slice, ...]
        if list(np.shape(Imgs)[1:3]) != [self.dims[0], self.dims[1]]:
            Imgs = np.pad(Imgs, ((0, 0), (pad_size_x, pad_size_x), (pad_size_y, pad_size_y), (0, 0)), constant_values=0)
        return np.asarray(Imgs, dtype=np.float32)

    def _resize_crop_or_pad(self, tensor):
        height, width = self.dims
        # return tf.image.resize_bilinear(tf.expand_dims(tensor, 0), [height, width])
        return tf.image.resize_image_with_crop_or_pad(tensor, height, width)

    def _resize_image_fixed(self, image):
        height, width = self.dims
        return tf.reshape(self._resize_crop_or_pad(image), [height, width, 3])

    def _normalize_image(self, image):
        return (image - self.mean) / self.stddev

    def _preprocess_image(self, image):
        image = self._resize_image_fixed(image)
        if self.normalize:
            image = self._normalize_image(image)
        return image

    def _input_images(self, image_dir, hold_out_inv=None):
        """Assumes that paired images are next to each other after ordering the
        files.
        """
        image_dir = os.path.join(self.data.current_dir, image_dir)

        filenames_1 = []
        filenames_2 = []
        image_files = os.listdir(image_dir)
        image_files.sort()

        assert len(image_files) % 2 == 0, 'expected pairs of images'

        for i in range(len(image_files) // 2):
            filenames_1.append(os.path.join(image_dir, image_files[i * 2]))
            filenames_2.append(os.path.join(image_dir, image_files[i * 2 + 1]))

        if hold_out_inv is not None:
            filenames = list(zip(filenames_1, filenames_2))
            random.seed(0)
            random.shuffle(filenames)
            filenames = filenames[:hold_out_inv]

            filenames_1, filenames_2 = zip(*filenames)
            filenames_1 = list(filenames_1)
            filenames_2 = list(filenames_2)

        input_1 = read_png_image(filenames_1, 1)
        input_2 = read_png_image(filenames_2, 1)
        image_1 = self._preprocess_image(input_1)
        image_2 = self._preprocess_image(input_2)
        return tf.shape(input_1), image_1, image_2

    def _input_test(self, image_dir, hold_out_inv=None):
        input_shape, im1, im2 = self._input_images(image_dir, hold_out_inv)
        return tf.train.batch(
            [im1, im2, input_shape],
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            allow_smaller_final_batch=True)

    # def get_normalization(self):
    #     return self.mean, self.stddev

    def input_raw(self, swap_images=True, sequence=True,
                  needs_crop=True, shift=0, seed=0,
                  center_crop=False, skip=0):
        """Constructs input of raw data.
        Args:
            sequence: Assumes that image file order in data_dirs corresponds to
                temporal order, if True. Otherwise, assumes uncorrelated pairs of
                images in lexicographical ordering.
            shift: number of examples to shift the input queue by.
                Useful to resume training.
            swap_images: for each pair (im1, im2), also include (im2, im1)
            seed: seed for filename shuffling.
        Returns:
            image_1: batch of first images
            image_2: batch of second images
        """
        if not isinstance(skip, list):
            skip = [skip]

        data_dirs = self.data.get_raw_dirs()
        height, width = self.dims
        #assert batch_size % 2 == 0

        filenames = []
        for dir_path in data_dirs:
            files = os.listdir(dir_path)
            files.sort()
            if sequence:
                steps = [1 + s for s in skip]
                stops = [len(files) - s for s in steps]
            else:
                steps = [2]
                stops = [len(files)]
                assert len(files) % 2 == 0
            for step, stop in zip(steps, stops):
                for i in range(0, stop, step):
                    if self.skipped_frames and sequence:
                        assert step == 1
                        num_first = frame_name_to_num(files[i])
                        num_second = frame_name_to_num(files[i+1])
                        if num_first + 1 != num_second:
                            continue
                    fn1 = os.path.join(dir_path, files[i])
                    fn2 = os.path.join(dir_path, files[i + 1])
                    filenames.append((fn1, fn2))

        random.seed(seed)
        random.shuffle(filenames)
        print("Training on {} frame pairs.".format(len(filenames)))

        filenames_extended = []
        for fn1, fn2 in filenames:
            filenames_extended.append((fn1, fn2))
            if swap_images:
                filenames_extended.append((fn2, fn1))

        shift = shift % len(filenames_extended)
        filenames_extended = list(np.roll(filenames_extended, shift))


        filenames_1, filenames_2 = zip(*filenames_extended)
        filenames_1 = list(filenames_1)
        filenames_2 = list(filenames_2)

        with tf.variable_scope('train_inputs'):
            image_1 = read_png_image(filenames_1)
            image_2 = read_png_image(filenames_2)

            if needs_crop:
                #if center_crop:
                #    image_1 = tf.image.resize_image_with_crop_or_pad(image_1, height, width)
                #    image_2 = tf.image.resize_image_with_crop_or_pad(image_1, height, width)
                #else:
                image_1, image_2 = random_crop([image_1, image_2], [height, width, 3])
            else:
                image_1 = tf.reshape(image_1, [height, width, 3])
                image_2 = tf.reshape(image_2, [height, width, 3])

            if self.normalize:
                image_1 = self._normalize_image(image_1)
                image_2 = self._normalize_image(image_2)

            return tf.train.batch(
                [image_1, image_2],
                batch_size=self.batch_size,
                num_threads=self.num_threads)


def read_png_image(filenames, num_epochs=None):
    """Given a list of filenames, constructs a reader op for images."""
    filename_queue = tf.train.string_input_producer(filenames,
        shuffle=False, capacity=len(filenames))
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_uint8 = tf.image.decode_png(value, channels=3)
    image = tf.cast(image_uint8, tf.float32)
    return image