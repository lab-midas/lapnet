import tensorflow as tf
from random import shuffle
import os
import numpy as np
from processing import get_maxmin_info_from_ods_file, load_data_3D

def generate_us_data(list_IDS, img_path, flow_path, slice_info, numrecords, aug, showfunc, mask_type='drUS'):
    num_data = 0
    list_us = np.arange(0, 31, 2)
    list_us[0] = 1
    for ind, dataID in enumerate(list_IDS):
        print ('start dataID .... ', dataID)
        ID_slice_info = slice_info[dataID]
        np.random.seed()
        us_rate_list = np.random.RandomState().choice(list_us, size=numrecords, replace=False)

        # make sure that fully sampled data is also included in the training data
        if not(1 in us_rate_list) and ind < int(len(list_IDS)/8):
            us_rate_list[0] = 1

        for undersampling_rate in us_rate_list:
           print('undersampling_rate ', undersampling_rate)
           img_ref, img_mov, flow, _ = load_data_3D(dataID=dataID,
                                                    img_path=img_path,
                                                    flow_path=flow_path,
                                                    aug_type=aug,
                                                    us_rate=undersampling_rate,
                                                    mask_type=mask_type,
                                                    normalized=False,
                                                    masking=False)


           for z_dim in range(int(ID_slice_info[0]), int(ID_slice_info[1])):
               print('slice ', z_dim, ' is saved')
               showfunc(img_ref[:,:,z_dim], img_mov[:,:,z_dim], flow[:,:,z_dim,:2])
               num_data +=1

        print(f'the generated training data has {num_data} samples')



class LAPNet_TFRec():
    def __init__(self, filename):
        self.fname = filename
        self.tfwriter = tf.io.TFRecordWriter(self.fname)

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, nparr):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[nparr.tobytes()]))

    def _float_feature(self, nparr):
        return tf.train.Feature(float_list=tf.train.FloatList(value=nparr))

    def write_record(self, img_ref, img_mov, flow):
        feature = {
            'image_ref_real': self._float_feature(img_ref.real.ravel()),
            'image_ref_imag': self._float_feature(img_ref.imag.ravel()),
            'image_mov_real': self._float_feature(img_mov.real.ravel()),
            'image_mov_imag': self._float_feature(img_mov.imag.ravel()),
            'flow': self._float_feature(flow.ravel()),
            'height': self._int64_feature(img_ref.shape[0]),
            'width': self._int64_feature(img_ref.shape[1]),
        }

        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.tfwriter.write(tf_example.SerializeToString())

    def close_record(self):
        self.tfwriter.flush()
        self.tfwriter.close()


class TFRsaver():
    def __init__(self, qtfr):
        self.qtfr = qtfr

    def savedata(self, img_ref, img_mov, flow):
        self.qtfr.write_record(
            img_ref,
            img_mov,
            flow
        )


def create_tfrecord(record_path, list_IDs, img_path, flow_path, slice_info, num_ID_usage, aug):
    qtfr = LAPNet_TFRec(record_path)
    tfrsaver = TFRsaver( qtfr)
    generate_us_data(list_IDs, img_path, flow_path, slice_info, num_ID_usage, aug, tfrsaver.savedata)
    qtfr.close_record()

if __name__ == '__main__':

    img_path = '/mnt/qdata/rawdata/MoCo/LAPNet/resp/motion_data'
    flow_path = '/mnt/qdata/rawdata/MoCo/LAPNet/resp/LAP'
    txt_training_IDs = '/home/students/studghoul1/Documents/research_thesis_data_results/create_data/training_subjects_names.txt'
    slice_info_coronal = '/home/students/studghoul1/Documents/research_thesis_data_results/create_data/slice_info_resp_coronal.ods'
    save_path = '/home/students/studghoul1/lapnet/tfrecords'
    TRAINSET_FNAME_real = 'training2Ddata_LAP.tfrecord'

    infile = open(txt_training_IDs, 'r')
    contents = infile.read().strip().split()
    training_IDs = [f for f in contents]
    infile.close()
    shuffle(training_IDs)

    slice_info = get_maxmin_info_from_ods_file(slice_info_coronal)
    fname_real = os.path.join(save_path, TRAINSET_FNAME_real)

    create_tfrecord(record_path= fname_real, list_IDs= training_IDs,
                    img_path= img_path, flow_path=flow_path,
                    slice_info=slice_info, num_ID_usage=1, aug='real')






