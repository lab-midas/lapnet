import tensorflow as tf
import numpy as np
# import tf_fft_helper
import e2eflow.core.card_US.tf_fft_helper
# input data is (rank==4): batch, height(x), width(y), channels(mag/pha)
# complex dimension is always last!
# default set to match this input

#############################
# real <-> complex conversion
def to_complex(x, axis_expand=3, real_imag=False):  # real/imag (True) or mag/pha (False)
    if not real_imag:
        x = mag_pha_to_real_imag(x, axis_expand)
    if axis_expand == 3:  # assume axis_expand+1 = rank
        return tf.complex(x[:, :, :, 0], x[:, :, :, 1])  # returns complex64
    elif axis_expand == 4:
        return tf.complex(x[:, :, :, :, 0], x[:, :, :, :, 1])
    elif axis_expand == 5:
        return tf.complex(x[:, :, :, :, :, 0], x[:, :, :, :, :, 1])
    elif axis_expand == 6:
        return tf.complex(x[:, :, :, :, :, :, 0], x[:, :, :, :, :, :, 1])

def mag_pha_to_real_imag(x, axis_expand=3):
    if axis_expand == 3:
        return tf.concat([tf.expand_dims(tf.cast(tf.multiply(x[:, :, :, 0], tf.math.cos(x[:, :, :, 1])), dtype=tf.float32), axis=axis_expand),
                          tf.expand_dims(tf.cast(tf.multiply(x[:, :, :, 0], tf.math.sin(x[:, :, :, 1])), dtype=tf.float32), axis=axis_expand)], axis=axis_expand)
    elif axis_expand == 4:
        return tf.concat([tf.expand_dims(tf.cast(tf.multiply(x[:, :, :, :, 0], tf.math.cos(x[:, :, :, :, 1])), dtype=tf.float32), axis=axis_expand),
                          tf.expand_dims(tf.cast(tf.multiply(x[:, :, :, :, 0], tf.math.sin(x[:, :, :, :, 1])), dtype=tf.float32), axis=axis_expand)], axis=axis_expand)
    elif axis_expand == 5:
        return tf.concat([tf.expand_dims(tf.cast(tf.multiply(x[:, :, :, :, :, 0], tf.math.cos(x[:, :, :, :, :, 1])), dtype=tf.float32), axis=axis_expand),
                          tf.expand_dims(tf.cast(tf.multiply(x[:, :, :, :, :, 0], tf.math.sin(x[:, :, :, :, :, 1])), dtype=tf.float32), axis=axis_expand)], axis=axis_expand)
    elif axis_expand == 6:
        return tf.concat([tf.expand_dims(tf.cast(tf.multiply(x[:, :, :, :, :, :, 0], tf.math.cos(x[:, :, :, :, :, :, 1])), dtype=tf.float32), axis=axis_expand),
                          tf.expand_dims(tf.cast(tf.multiply(x[:, :, :, :, :, :, 0], tf.math.sin(x[:, :, :, :, :, :, 1])), dtype=tf.float32), axis=axis_expand)], axis=axis_expand)

def to_real(x, axis_expand=3, real_imag=False):
    if real_imag:
        return tf.concat([tf.expand_dims(tf.cast(tf.math.real(x), dtype=tf.float32), axis=axis_expand), tf.expand_dims(tf.cast(tf.math.imag(x), dtype=tf.float32), axis=axis_expand)], axis=axis_expand)
    else:  # mag/pha
        return tf.concat([tf.expand_dims(tf.cast(tf.math.abs(x), dtype=tf.float32), axis=axis_expand), tf.expand_dims(tf.cast(tf.math.angle(x), dtype=tf.float32), axis=axis_expand)], axis=axis_expand)
#############################

#########################
# pre/post FFT operations
def pre_fft(x, axes=(1, 2), perm=np.asarray([0,3,1,2]), axis_expand=3, real_imag=False):  # ifftshift -> complex -> permute to be transformed dimensions to end (order important!)
    return tf.transpose(to_complex(tf_fft_helper.ifftshift(x, axes=axes), axis_expand=axis_expand, real_imag=real_imag), perm=perm)

def post_fft(x, axes=(1, 2), perm=np.asarray([0,3,1,2]), axis_expand=3, real_imag=False):
    iperm = np.empty(perm.size, dtype=np.int32)  # get back transpose
    for i in np.arange(perm.size):
        iperm[perm[i]] = i
    return tf_fft_helper.fftshift(tf.transpose(to_real(x, axis_expand=1, real_imag=real_imag), perm=iperm), axes=axes)  # put complex channel in 2nd dim again
#########################

##########
# FFT/IFFT
def fftnshift(x, axes=(1, 2), real_imag=False):  # always along the most inner dimension (last dim)
    # always return real-valued tensor
    # use python variables instead of evaluation tensorflow graph conditions -> makes code more readable
    inshape = x.get_shape().as_list()
    rank = len(inshape)  # can be invoked at creation
    perm = np.array(np.arange(0, rank))
    mask = np.in1d(perm, np.asarray(axes), invert=True)
    perm = np.concatenate([perm[mask], np.asarray(axes)])
    if rank==4:  # batch, height(x), width(y), channels/complex(real_imag/mag_pha)
        axis_expand_pre = 3
    elif rank==5:  # batch, height(x), width(y), depth(z), channels/complex(real_imag/mag_pha)
        axis_expand_pre = 4
    elif rank==6:  # batch, height(x), width(y), depth(z), channels, complex(real_imag/mag_pha)
        axis_expand_pre = 5
    elif rank==7:  # batch, height(x), width(y), depth(z), time, channels, complex(real_imag/mag_pha)
        axis_expand_pre = 6
    axis_expand_post = int(np.where(perm == rank - 1)[0])
    perm_post = perm
    mask = np.in1d(perm, np.asarray(rank-1), invert=True)
    perm_pre = perm[mask]  # after taking the complex dimension out (by to_complex())
    scalefft = tf.math.divide(tf.constant(1, dtype=tf.float32), tf.sqrt(tf.constant(np.prod(inshape[np.asarray(axes)]), dtype=tf.float32)))

    if len(axes) == 1:
        return post_fft(tf.multiply(scalefft, tf.signal.fft(pre_fft(x, axes, perm_pre, axis_expand_pre, real_imag))), axes, perm_post, axis_expand_post, real_imag)
    elif len(axes) == 2:
        return post_fft(tf.multiply(scalefft, tf.signal.fft2d(pre_fft(x, axes, perm_pre, axis_expand_pre, real_imag))), axes, perm_post, axis_expand_post, real_imag)
    elif len(axes) == 3:
        return post_fft(tf.multiply(scalefft, tf.signal.fft3d(pre_fft(x, axes, perm_pre, axis_expand_pre, real_imag))), axes, perm_post, axis_expand_post, real_imag)
    else:
        for ax in axes:
            perm = np.array(np.arange(0, rank))
            mask = np.in1d(perm, np.asarray(ax), invert=True)
            perm = np.concatenate([perm[mask], np.asarray(ax)])
            perm_post = perm
            mask = np.in1d(perm, np.asarray(rank - 1), invert=True)
            perm_pre = perm[mask]  # after taking the complex dimension out (by to_complex())
            scalefftax = tf.math.divide(tf.constant(1, dtype=tf.float32), tf.sqrt(tf.constant(inshape[ax], dtype=tf.float32)))
            x = post_fft(tf.multiply(scalefftax, tf.signal.fft(pre_fft(x, ax, perm_pre, axis_expand_pre, real_imag))), ax, perm_post, axis_expand_post, real_imag)
        return x

def ifftnshift(x, axes=(1, 2), real_imag=False):  # always along the most inner dimension (last dim)
    # always return real-valued tensor
    # use python variables instead of evaluation tensorflow graph conditions -> makes code more readable
    inshape = x.get_shape().as_list()
    rank = len(inshape)  # can be invoked at creation
    perm = np.array(np.arange(0, rank))
    mask = np.in1d(perm, np.asarray(axes), invert=True)
    perm = np.concatenate([perm[mask], np.asarray(axes)])
    if rank == 4:  # batch, height(x), width(y), channels/complex(real_imag/mag_pha)
        axis_expand_pre = 3
    elif rank == 5:  # batch, height(x), width(y), depth(z), channels/complex(real_imag/mag_pha)
        axis_expand_pre = 4
    elif rank == 6:  # batch, height(x), width(y), depth(z), channels, complex(real_imag/mag_pha)
        axis_expand_pre = 5
    elif rank == 7:  # batch, height(x), width(y), depth(z), time, channels, complex(real_imag/mag_pha)
        axis_expand_pre = 6
    axis_expand_post = int(np.where(perm == rank - 1)[0])
    perm_post = perm
    mask = np.in1d(perm, np.asarray(rank - 1), invert=True)
    perm_pre = perm[mask]  # after taking the complex dimension out (by to_complex())
    scalefft = tf.sqrt(tf.constant(np.prod(inshape[np.asarray(axes)]), dtype=tf.float32))

    if len(axes) == 1:
        return post_fft(tf.multiply(scalefft, tf.signal.ifft(pre_fft(x, axes, perm_pre, axis_expand_pre, real_imag))), axes, perm_post, axis_expand_post,
                        real_imag)
    elif len(axes) == 2:
        return post_fft(tf.multiply(scalefft, tf.signal.ifft2d(pre_fft(x, axes, perm_pre, axis_expand_pre, real_imag))), axes, perm_post, axis_expand_post,
                        real_imag)
    elif len(axes) == 3:
        return post_fft(tf.multiply(scalefft, tf.signal.ifft3d(pre_fft(x, axes, perm_pre, axis_expand_pre, real_imag))), axes, perm_post, axis_expand_post,
                        real_imag)
    else:
        for ax in axes:
            perm = np.array(np.arange(0, rank))
            mask = np.in1d(perm, np.asarray(ax), invert=True)
            perm = np.concatenate([perm[mask], np.asarray(ax)])
            perm_post = perm
            mask = np.in1d(perm, np.asarray(rank - 1), invert=True)
            perm_pre = perm[mask]  # after taking the complex dimension out (by to_complex())
            scalefftax = tf.sqrt(tf.constant(inshape[ax], dtype=tf.float32))
            x = post_fft(tf.multiply(scalefftax, tf.signal.ifft(pre_fft(x, ax, perm_pre, axis_expand_pre, real_imag))), ax, perm_post, axis_expand_post, real_imag)
        return x
##########

def expand_dims_axis(x, axis=-1):
    # length of axis gives amount of expands, e.g. [-1, -1] expands 2 ending dimensions
    # ATTENTION: order of appended dimensions is crucial, but usually just end-appending required
    for ax in axis:
        x = tf.expand_dims(x, axis=ax)
    return x


def fftnshift_np(x, axes=(0, 1)):
    for ax in axes:
        x = 1 / np.sqrt(x.shape[ax]) * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=ax), axis=ax), axes=ax)
    return x


def ifftnshift_np(x, axes=(0, 1)):
    for ax in axes:
        x = np.sqrt(x.shape[ax]) * np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=ax), axis=ax), axes=ax)
    return x