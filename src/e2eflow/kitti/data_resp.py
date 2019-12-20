class MRI_Resp_2D(Input):
    def __init__(self, data, batch_size, dims, *,
                 num_threads=1, normalize=True,
                 skipped_frames=False):
        super().__init__(data, batch_size, dims, num_threads=num_threads,
                         normalize=normalize, skipped_frames=skipped_frames)

    def _u_generation_2D(self, img_size, amplitude, motion_type='constant'):
        M, N = self.dims
        if motion_type == 'constant':
            u_C = np.random.rand(2)
            amplitude = amplitude / np.linalg.norm(u_C, 2)
            u = amplitude * np.ones((M, N, 2))
            u[..., 0] = u_C[0] * u[..., 0]
            u[..., 1] = u_C[1] * u[..., 1]
        elif motion_type == 'smooth':
            u = np.random.rand(M, N, 2)
            cut_off = 0.01
            w_x_cut = math.floor(cut_off / (1 / M) + (M + 1) / 2)
            w_y_cut = math.floor(cut_off / (1 / N) + (N + 1) / 2)

            LowPass_win = np.zeros((M, N))
            LowPass_win[(M - w_x_cut): w_x_cut, (N - w_y_cut): w_y_cut] = 1

            u[..., 0] = (np.fft.ifft2(np.fft.fft2(u[..., 0]) * np.fft.ifftshift(LowPass_win))).real
            u[..., 1] = (np.fft.ifft2(np.fft.fft2(u[..., 1]) * np.fft.ifftshift(LowPass_win))).real
        elif motion_type == 'realistic':
            pass

        return u

    def _np_warp_2D(self, img, flow):
        img = img.astype('float32')
        flow = flow.astype('float32')
        height, width = self.dims
        posx, posy = np.mgrid[:height, :width]
        # flow=np.reshape(flow, [-1, 3])
        vx = flow[:, :, 0]
        vy = flow[:, :, 1]

        coord_x = posx + vx
        coord_y = posy + vy
        coords = np.array([coord_x, coord_y])
        warped = warp(img, coords, order=1)  # order=1 for bi-linear

        return warped

    def input_train_gt(self, hold_out):
        img_dirs = ['resp/patient',
                    'resp/volunteer']
        selected_frames = [0, 3]
        selected_slices = list(range(15, 55))
        amplitude = 30

        height, width = self.dims
        #batches = {'img1': [], 'img2': [], 'flow_gt': []}
        batches = []
        mask = np.zeros((height, width, 1))
        fn_im_paths = []
        for img_dir in img_dirs:
            img_dir = os.path.join(self.data.current_dir, img_dir)
            img_files = os.listdir(img_dir)
            for img_file in img_files:
                if not img_file.startswith('.'):
                    try:
                        img_mat = os.listdir(os.path.join(img_dir, img_file))[0]
                    except Exception:
                        print("File {} is empty!".format(img_file))
                        continue
                    fn_im_paths.append(os.path.join(img_dir, img_file, img_mat))

        augment how many times
        n_training = len(fn_impaths) * n_augment

        for idata in range(n_training):
            list_data = [fn_im_paths[idata], np.randn(0,3), ...]


        random.seed(0)
        random.shuffle(fn_im_paths)
        for fn_im_path in fn_im_paths:

        def dotheactualloading(fn_im_path, intype=0, iaugment): # img1=0/img2=1/flow=2/p=3):
            with h5py.File(fn_im_path, 'r') as f:
                # fn_im_raw = sio.loadmat(fn_im_path)
                dset = f['dImg']
                try:
                    dset = np.array(dset, dtype=np.float32)
                    # dset = tf.constant(dset, dtype=tf.float32)
                except Exception:
                    print("File {} is defective and cannot be read!".format(img_file))
                    continue
                dset = np.transpose(dset, (2, 3, 1, 0))
                dset = dset[..., selected_frames]
                dset = dset[..., selected_slices, :]
            for frame in range(np.shape(dset)[3]):
                for slice in range(np.shape(dset)[2]):
                    img = dset[..., slice, :][..., frame]
                    img_size = np.shape(img)
                    u = self._u_generation_2D(img_size, amplitude, motion_type='smooth')
                    warped_img = self._np_warp_2D(img, u)
                    img, warped_img, = img[..., np.newaxis], warped_img[..., np.newaxis]
                    try:
                        batch = np.concatenate([img, warped_img, u, mask], 2)
                        #  batch = tf.convert_to_tensor(batch, dtype=tf.float32)
                        batches.append(batch)
                    except Exception:
                        pass
                        print('the size of {} is {}, does not match {}. '
                              'It cannot be loaded!'.format(fn_im_path, np.shape(img)[:2], self.dims))
                        break
                break
            if len(batches) > 500:
                break

        return [img, warped_img, flow, mask]



        random.shuffle(batches)
        batches = np.array(batches, dtype=np.float32)


        batchdata = tf.data.Dataset (list_data)

        out_data = tf.map(lamba x: dotheactualloading(x[0], iaugment=x[1] )


        # im1 = tf.data.Dataset.from_tensor_slices(batches[..., 0])
        # im2 = tf.data.Dataset.from_tensor_slices(batches[..., 1])
        # flow_gt = tf.data.Dataset.from_tensor_slices(batches[..., 2:4])
        # mask_gt = tf.data.Dataset.from_tensor_slices(batches[..., 4])
        # batches = tf.data.Dataset.zip((im1, im2, flow_gt, mask_gt))
        # batches = tf.convert_to_tensor(batches)
        # batches = tf.data.Dataset.from_tensor_slices(batches)

        # if self.normalize:
        #     batches[..., 1] = self._normalize_image(batches[..., 1])
        #     im2 = self._normalize_image(im2)

        # return next(iter(batches.batch(self.batch_size)))

        return tf.train.batch(
            out_data,
            batch_size=self.batch_size,
            num_threads=self.num_threads)