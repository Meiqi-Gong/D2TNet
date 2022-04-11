from __future__ import print_function
import time
import os
import h5py
import numpy as np
import scipy.ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
# from scipy.misc import imread
import scipy.io as scio
from datetime import datetime
import cv2
from model import MODEL

pan_path = 'QB_train/pan.h5'
gt_path = 'QB_train/gt.h5'
# pan_path = 'GF_train/pan.h5'
# gt_path = 'GF_train/gt.h5'

EPOCHES = 100
BATCH_SIZE = 16
patch_size = 264
logging_period = 5
LEARNING_RATE = 0.0005
#0.0005
DECAY_RATE = 0.95

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def main():
    start_time = datetime.now()
    print('Epoches: %d, Batch_size: %d' % (EPOCHES, BATCH_SIZE))

    num_imgs = 19999
    mod = num_imgs % BATCH_SIZE
    n_batches = int(num_imgs // BATCH_SIZE)
    print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))

    with tf.Graph().as_default(), tf.Session() as sess:
        PAN = tf.placeholder(tf.float32, shape=(BATCH_SIZE, patch_size, patch_size, 1), name='PAN')
        ms = tf.placeholder(tf.float32, shape=(BATCH_SIZE, patch_size / 4, patch_size / 4, 4), name='ms')
        GT = tf.placeholder(tf.float32, shape=(BATCH_SIZE, patch_size, patch_size, 4), name='GT')

        model = MODEL(PAN, ms, GT, BATCH_SIZE)

        GT2 = tf.nn.max_pool(GT, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        GT4 = tf.nn.max_pool(GT2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        current_iter = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE, global_step=current_iter,
                                                   decay_steps=int(n_batches), decay_rate=DECAY_RATE,
                                                   staircase=False)

        # loss_all, loss_1, loss_2, loss_4 = model.train_loss
        loss_all = model.train_loss
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_all, global_step=current_iter,
                                                                      var_list=model.variables)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=100)

        tf.summary.scalar('Loss_all', loss_all)

        tf.summary.image('PAN', PAN, max_outputs=2)
        tf.summary.image('ms', ms[:, :, :, 0:3], max_outputs=2)
        tf.summary.image('GT_low4', GT4[:, :, :, 0:3], max_outputs=3)
        tf.summary.image('GT', GT[:, :, :, 0:3], max_outputs=3)
        tf.summary.image('HRMS', model.HRMS[:, :, :, 0:3], max_outputs=3)
        tf.summary.image('HRMS_low4', model.HRMS4[:, :, :, 0:3], max_outputs=3)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/", sess.graph)

        # ** Start Training **
        step = 0
        for epoch in range(EPOCHES):
            # np.random.shuffle(data)
            for batch in range(n_batches):
                for i in range((batch+1)*BATCH_SIZE):
                    if i>=batch*BATCH_SIZE:
                        pan_data = h5py.File(pan_path, 'r')
                        pan_data = pan_data['pan' + str(i+1)]
                        pan_data = np.expand_dims(pan_data, axis=0)

                        gt_data = h5py.File(gt_path, 'r')
                        gt_data = gt_data['gt' + str(i+1)]
                        gt_data = np.expand_dims(gt_data, axis=0)

                        if i==batch*BATCH_SIZE:
                            pan_train=pan_data
                            gt_train=gt_data
                        else:
                            pre_pandata = np.concatenate([pan_train, pan_data], axis=0)
                            pre_gtdata = np.concatenate([gt_train, gt_data], axis=0)
                            pan_train = pre_pandata
                            gt_train = pre_gtdata

                step += 1
                current_iter = step

                ms_batch = np.zeros(shape=(BATCH_SIZE, int(patch_size / 4), int(patch_size / 4), 4), dtype=np.float32)
                PAN_batch = np.zeros(shape=(BATCH_SIZE, patch_size, patch_size, 1), dtype=np.float32)
                GT_batch = gt_train/255.0
                PAN_pre = np.expand_dims(pan_train, axis=-1)/255.0

                for b in range(BATCH_SIZE):
                    PAN_batch[b, :, :, 0] = cv2.resize(PAN_pre[b, :, :, 0], (patch_size, patch_size))
                    for c in range(4):
                        ms_batch[b, :, :, c] = cv2.resize(GT_batch[b, :, :, c],
                                                          (int(patch_size / 4), int(patch_size / 4)))

                FEED_DICT = {PAN: PAN_batch, ms: ms_batch, GT: GT_batch}
                id=0
                sess.run(train_op, feed_dict=FEED_DICT)

                loss = sess.run(loss_all, feed_dict=FEED_DICT)
                if batch%2 == 0:
                    while loss>30 and id <10:
                        sess.run(train_op, feed_dict=FEED_DICT)
                        id += 1

                result = sess.run(merged, feed_dict=FEED_DICT)
                # result = sess.run(model.HRMS, feed_dict=FEED_DICT)
                writer.add_summary(result, step)
                if step % 100 == 0:
                    saver.save(sess, 'models/' + str(step) + '.ckpt')

                is_last_step = (epoch == EPOCHES - 1) and (batch == n_batches - 1)
                if is_last_step or step % logging_period == 0:
                    elapsed_time = datetime.now() - start_time
                    loss = sess.run(loss_all, feed_dict=FEED_DICT)
                    lr = sess.run(learning_rate)
                    print('Epoch: %d/%d, Step: %d/%d, Loss: %s, Lr: %s, Time: %s\n' % (
                        epoch + 1, EPOCHES, step % n_batches, n_batches, loss, lr, elapsed_time))

            saver.save(sess, 'models_QB/epoch' +str(epoch+1)+ '/model.model')



def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)


if __name__ == '__main__':
    main()
