from __future__ import print_function

import numpy as np
import tensorflow as tf
from model import MODEL
from scipy.misc import imread, imsave
import scipy.io as scio
import time
from tqdm import tqdm
import warnings
import h5py
import cv2
import random
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# path1 = 'GF_test/reduced/PAN/'
# path2 = 'GF_test/reduced/MS/'

# path1 = 'GF_test/full/PAN/'
# path2 = 'GF_test/full/MS/'

# path1 = 'QB_test/full/PAN/'
# path2 = 'QB_test/full/MS/'

path1 = 'QB_test/reduced/PAN/'
path2 = 'QB_test/reduced/MS/'

output_path = 'results/'
BATCH_SIZE=1
EPOCHES=100

patch=264
def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('GFLOPs: {};    Trainable params: {}'.format(flops.total_float_ops/1000000000.0, params.total_parameters))

def main():
    print('\nBegin to generate pictures ...\n')
    t = []

    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            MS = tf.placeholder(tf.float32, shape=(1, patch//4, patch//4, 4), name='MS')
            PAN = tf.placeholder(tf.float32, shape=(1, patch, patch, 1), name='PAN')
            # MS = tf.placeholder(tf.float32, shape=(1, patch, patch, 4), name='MS')
            # PAN = tf.placeholder(tf.float32, shape=(1, patch*4, patch*4, 1), name='PAN')
            GT = tf.placeholder(tf.float32, shape=(1, patch*4, patch*4, 4), name='GT')
            model = MODEL(PAN, MS, GT, BATCH_SIZE)
            result = model.HRMS
            # result2 = model.HRMS2
            # result4 = model.HRMS4

            t_list = tf.trainable_variables()
            saver = tf.train.Saver(var_list=t_list)
            # stats_graph(graph)

            sess.run(tf.global_variables_initializer())

            for epoch in range(100):
                e=epoch+1
                if e==90:
                    print('epoch: ', str(e))
                    MODEL_SAVE_PATH = 'models_QB/epoch' + str(epoch+1) + '/model.model'
                    # MODEL_SAVE_PATH = 'models_GF/epoch' + str(epoch+1) + '/model.model'
                    saver.restore(sess, MODEL_SAVE_PATH)

                    for i in tqdm(range(10)):
                        begin = time.time()
                        file_name1 = path1 + str(i + 1) + '.tif'
                        file_name2 = path2 + str(i + 1) + '.tif'

                        pan = imread(file_name1) / 255.0
                        ms = imread(file_name2) / 255.0
                        pan = np.expand_dims(pan, axis=0)
                        pan = np.expand_dims(pan, axis=3)
                        ms = np.expand_dims(ms, axis=0)
                        ms = ms.astype('float32')
                        pan = pan.astype('float32')

                        output = sess.run(result, feed_dict={PAN: pan, MS: ms})
                        ##reduced_save
                        scio.savemat(output_path + str(i + 1) + '.mat', {'i': output[0, :, :, :]})

                        ##full_save
                        # scio.savemat(output_path + str(i + 1) + '.mat',
                        #              {'i': (output[0, :, :, :] + 1) / 2})
                        end = time.time()
                        if i > 0:
                            t.append(end - begin)
                            print("Time: mean: %s, std: %s" % (np.mean(t), np.std(t)))


def count():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        if 'p_fuse' in variable.name:
            print(variable.name)
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
    return total_parameters


def count_flops(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOPs: {}'.format(flops.total_float_ops))



if __name__ == '__main__':
    main()
