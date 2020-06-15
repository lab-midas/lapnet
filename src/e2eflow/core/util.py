import tensorflow as tf
import os
from ..ops import downsample as downsample_ops
from ..core.input import load_mat_file
import numpy as np
from pathlib import Path


def summarized_placeholder(name, prefix=None, key=tf.GraphKeys.SUMMARIES):
    prefix = '' if not prefix else prefix + '/'
    p = tf.placeholder(tf.float32, name=name)
    tf.summary.scalar(prefix + name, p, collections=[key])
    return p


def resize_area(tensor, like):
    _, h, w, _ = tf.unstack(tf.shape(like))
    return tf.stop_gradient(tf.image.resize_area(tensor, [h, w]))


def resize_bilinear(tensor, like):
    _, h, w, _ = tf.unstack(tf.shape(like))
    return tf.stop_gradient(tf.image.resize_bilinear(tensor, [h, w]))


def downsample(tensor, num):
    _,height, width,_ = tensor.shape.as_list()
    if height%2==0 and width%2==0:
        return downsample_ops(tensor, num)
    else:
        return tf.image.resize_area(tensor, tf.constant([int(height/num), int(width/num)]))


def mat2npz(mat_dirs, output_dirs, save_type=np.float32):
    if not os.path.exists(output_dirs):
        os.mkdir(output_dirs)
    mat_files = os.listdir(mat_dirs)
    for mat_file in mat_files:
        f = load_mat_file(os.path.join(mat_dirs, mat_file))
        keys2save = [i for i in f.keys() if isinstance(f[i], np.ndarray)]
        values2save = [np.asarray(f[i], dtype=save_type) if f[i].dtype == np.float64 else f[i] for i in keys2save]
        save_path = os.path.join(output_dirs, mat_file.split('.')[0])
        np.savez(save_path, **{name: value for name, value in zip(keys2save, values2save)})
