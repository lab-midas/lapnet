import os
import sys

import numpy as np
import matplotlib.image as mpimg

from . import raw_records
from ..core.data import Data
from ..util import tryremove
from ..core.input import frame_name_to_num


def exclude_test_and_train_images(kitti_dir, exclude_lists_dir, exclude_target_dir,
                                  remove=False):
    to_move = []

    def exclude_from_seq(day_name, seq_str, image, view, distance=10):
        # image is the first frame of each frame pair to exclude
        seq_dir_rel = os.path.join(day_name, seq_str, view, 'data')
        seq_dir_abs = os.path.join(kitti_dir, seq_dir_rel)
        target_dir_abs = os.path.join(exclude_target_dir, seq_dir_rel)
        if not os.path.isdir(seq_dir_abs):
            print("Not found: {}".format(seq_dir_abs))
            return
        try:
            os.makedirs(target_dir_abs)
        except:
            pass
        seq_files = sorted(os.listdir(seq_dir_abs))
        image_num = frame_name_to_num(image)
        try:
            image_index = seq_files.index(image)
        except ValueError:
            return
        # assume that some in-between files may be missing
        start = max(0, image_index - distance)
        stop = min(len(seq_files), image_index + distance + 2)
        start_num = image_num - distance
        stop_num = image_num + distance + 2
        for i in range(start, stop):
            filename = seq_files[i]
            num = frame_name_to_num(filename)
            if num < start_num or num >= stop_num:
                continue
            to_move.append((os.path.join(seq_dir_abs, filename),
                            os.path.join(target_dir_abs, filename)))

    for filename in os.listdir(exclude_lists_dir):
        exclude_list_path = os.path.join(exclude_lists_dir, filename)
        with open(exclude_list_path) as f:
            for line in f:
                line = line.rstrip('\n')
                if line.split(' ')[0].endswith('_10'):
                    splits = line.split(' ')[-1].split('\\')
                    image = splits[-1]
                    seq_str = splits[0]
                    day_name, seq_name = seq_str.split('_drive_')
                    seq_name = seq_name.split('_')[0] + '_extract'
                    seq_str = day_name + '_drive_' + seq_name
                    exclude_from_seq(day_name, seq_str, image, 'image_02')
                    exclude_from_seq(day_name, seq_str, image, 'image_03')
    if remove:
        print("Collected {} files. Deleting...".format(len(to_move)))
    else:
        print("Collected {} files. Moving...".format(len(to_move)))

    for i, data in enumerate(to_move):
        try:
            src, dst = data
            print("{} / {}: {}".format(i, len(to_move) - 1, src))
            if remove:
                os.remove(src)
            else:
                os.rename(src, dst)
        except: # Some ranges may overlap
            pass

    return len(to_move)

