import math
import numpy as np
from core.image_warp import np_warp_2D
from core.undersample.retrospective_radial import subsample_radial
from core.undersample.pad_crop import post_crop
from core.util import load_mat_file
from processing import pos_generation_2D, _u_generation_2D
from core.cropping import arr2kspace


class MRI_Card_2D():

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
                      mask_type='radial',
                      US_rate='random',
                      num_to_take=1500):
        """

        :param fn_im_paths: list, the subject list for training
        :param slice_info: list, which slices to take
        :param aug_type: synthetic motion augmentation type
        :param amp: amplitude of augmented motion
        :param mask_type: currently only 2D radial is available
        :param US_rate:
        :param num_to_take:
        :return:
        """

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
            try:
                f = load_mat_file(fn_im_path)
            except:
                try:
                    f = np.load(fn_im_path)
                except ImportError:
                    print("Wrong Data Format")

            name = fn_im_path.split('/')[-1].split('.')[0]

            ref = np.asarray(f['dFixed'], dtype=np.float32)
            pad_size_x = int((self.dims[0] - np.shape(ref)[0]) / 2)
            pad_size_y = int((self.dims[1] - np.shape(ref)[1]) / 2)
            ux = np.asarray(f['ux'], dtype=np.float32)  # ux for warp
            uy = np.asarray(f['uy'], dtype=np.float32)

            uxy = np.stack((ux, uy), axis=-1)
            slice2take = slice_info[name]

            for s in slice2take:
                ref_2D = ref[..., s]
                if aug_type == 'real_x_smooth':
                    u_syn = _u_generation_2D(np.shape(ux)[:2], amp, motion_type=1)
                    u = np.multiply(uxy[:, :, s], u_syn)
                elif aug_type == 'smooth':
                    u = _u_generation_2D(np.shape(ux)[:2], amp, motion_type=1)
                elif aug_type == 'constant':
                    u = _u_generation_2D(np.shape(ux)[:2], amp, motion_type=0)
                elif aug_type == 'real':
                    u = uxy[:, :, s]
                else:
                    raise ImportError('wrong augmentation type is given')
                mov_2D = np_warp_2D(ref_2D, u)
                im_pair = np.stack((ref_2D, mov_2D), axis=-1)

                if US_rate:
                    if US_rate == 'random':
                        # acc1 = 1
                        # acc2 = np.random.choice(np.arange(2, 21, 6))
                        # acc = np.random.choice([acc1, acc2])
                        acc = np.random.choice(np.arange(1, 18, 4))  # TODO the US-rate can be modified here
                    else:
                        try:
                            acc = US_rate
                        except ImportError:
                            print("Wrong undersampling rate is given")
                            continue
                    if acc != 1:
                        im_pair_US = np.squeeze(subsample_radial(im_pair[..., np.newaxis, :], acc=acc))
                        im_pair = np.absolute(post_crop(im_pair_US, np.shape(im_pair)))
                        # fig = plt.figure(figsize=(5, 5), dpi=100)
                        # plt.imshow(im_pair[..., 0])
                        # fig.savefig('/home/jpa19/PycharmProjects/MA/UnFlow/US_without_pad.png')

                data = np.concatenate((im_pair, u), axis=-1)
                data = np.pad(data, ((pad_size_x, pad_size_x), (pad_size_y, pad_size_y), (0, 0)), constant_values=0)
                output = np.concatenate((output, data[np.newaxis, ...]), axis=0)

            i += 1
            if i == len(fn_im_paths):
                if flag == 0:
                    i = 0
                else:
                    break
        print("{} real {} data are generated".format(num_to_take, aug_type))
        return np.asarray(output[:num_to_take, ...], dtype=np.float32)

    def input_train_data(self, img_dirs, slice_info, params, case='train_supervised'):

        if case == 'train_supervised':
            total_data_num = params.get('total_data_num')
        elif case == 'validation':
            total_data_num = 72
        num_constant = math.floor(total_data_num * params.get('augment_type_percent')[0])
        num_smooth = math.floor(total_data_num * params.get('augment_type_percent')[1])
        num_real = math.floor(total_data_num * params.get('augment_type_percent')[2])
        num_real_x_smooth = math.floor(total_data_num * params.get('augment_type_percent')[3])
        assert (num_real <= 475 and case == 'train_supervised') or (num_real <= 72 and case == 'validation')

        batches = np.zeros((0, self.dims[0], self.dims[1], 4), dtype=np.float32)
        # batches = np.zeros((0, 176, 132, 4), dtype=np.float32)
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

        batches = np.concatenate((batches, aug_data_real, aug_data_constant, aug_data_smooth, aug_data_real_x_smooth),
                                 axis=0)
        np.random.shuffle(batches)
        if params.get('network') == 'lapnet':
            radius = int((params.get('crop_size') - 1) / 2)
            if params.get('padding'):
                # here only x-axis will be padded cause y-axis has been padded much in load_aug_data()
                batches = np.pad(batches, ((0, 0), (radius, radius), (0, 0), (0, 0)), constant_values=0)
            if params.get('random_crop'):
                batches = self.crop2D(batches, crop_size=params.get('crop_size'), box_num=params.get('crop_box_num'),
                                      cut_margin=20)
            else:
                x_dim, y_dim = np.shape(batches)[1:3]
                pos = pos_generation_2D(intervall=[[0, x_dim - params.get('crop_size') + 1],
                                                   [0, y_dim - params.get('crop_size') + 1]], stride=4)
                batches = self.crop2D_FixPts(batches, crop_size=params.get('crop_size'), box_num=np.shape(pos)[1],
                                             pos=pos)
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
