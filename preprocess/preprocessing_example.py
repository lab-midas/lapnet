from skimage import data
import matplotlib.pyplot as plt
from math import pi
import numpy as np
from core.tapering import taper2D, ifft_along_dim
from processing import flow_variation_3d, subsample_respiratory_data
from core.image_warp import np_warp_3D

# read complex image
image = data.brain()
phase = np.random.rand(*image.shape)
image = image * np.exp(1j * 2 * pi * 0.1 * phase)

# normalize image
max_amp = np.max(np.abs(image))
image = (image.real / max_amp) + 1j * (image.imag / max_amp)

# generate 3D random flow
random_flow = np.random.rand(*image.shape, 3)

# choose the flow augmentation type
## 'real': keep the random generated flow
## 'smooth': gnerate smooth flow
## 'constant': shift flow with the specified pixels in the amp variable
## 'real_x_smooth': the real flow x smooth flow
flow_augmentation_type = 'constant'
constant_flow = flow_variation_3d(random_flow, flow_augmentation_type, amp=30)

# perform the warping on the 3D image
moving_image = np_warp_3D(image, constant_flow)

# undersampling
us_rate = 8  # specify the undersampling rate
ref_us, mov_us = subsample_respiratory_data(image, moving_image,
                                            acc=us_rate,
                                            mask_type='drUs')

# choose layer to display and taper
image = image[5, :, :]
moving_image = moving_image[5, :, :]
ref_us = ref_us[5, :, :]
mov_us = mov_us[5, :, :]

# tapering
k_ref, k_mov = taper2D(ref_us, mov_us, x_pos=70, y_pos=50, crop_size=100)

# show the results
fig, ax = plt.subplots(2, 3, figsize=(14, 14))
ax[0][0].imshow(np.abs(image))
ax[0][0].set_title('reference image')
ax[0][0].axis('off')
ax[1][0].imshow(np.abs(moving_image))
ax[1][0].set_title('moving image')
ax[1][0].axis('off')
ax[0][1].imshow(np.abs(ref_us))
ax[0][1].set_title('reference image undersampled')
ax[0][1].axis('off')
ax[1][1].imshow(np.abs(mov_us))
ax[1][1].set_title('moving image undersampled')
ax[1][1].axis('off')
ax[0][2].imshow(np.abs(ifft_along_dim(k_ref)))
ax[0][2].set_title('reference image tapered')
ax[0][2].axis('off')
ax[1][2].imshow(np.abs(ifft_along_dim(k_mov)))
ax[1][2].set_title('moving image tapered')
ax[1][2].axis('off')
plt.show()
