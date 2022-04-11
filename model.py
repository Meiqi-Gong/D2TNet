import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
from BasicConvLSTMCell import *
import warnings

warnings.filterwarnings("ignore")
weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.05)
weight_regularizer = None


class MODEL:
    def __init__(self, PAN, ms, gt, batch_size):
        self.batch_size = batch_size
        n, w, h, c = ms.get_shape().as_list()
        self.weight = w
        self.height = h
        # self.HRMS = self.model(PAN, ms, batch_size)
        self.HRMS, self.HRMS2, self.HRMS4 = self.model(PAN, ms, batch_size)
        self.train_loss = self.inference_losses(self.HRMS, self.HRMS2, self.HRMS4, gt, ms)

    def model(self, PAN, ms, batch_size):
        MS4 = ms
        PAN2 = tf.nn.max_pool(PAN, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        PAN4 = tf.nn.max_pool(PAN2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        MS2 = up_sample(MS4)
        MS = up_sample(MS2)

        I1 = tf.concat([PAN, MS], axis=-1)
        I2 = tf.concat([PAN2, MS2], axis=-1)
        I4 = tf.concat([PAN4, MS4], axis=-1)
        print('I2 Shape:', I2.shape)
        print('I4 Shape:', I4.shape)

        with tf.variable_scope('model'):
            ##U1
            with tf.variable_scope('beforeU1'):
                in4_init = self.conv_beforeU(I4, reuse=False)
                in2_init = self.conv_beforeU(I2, reuse=True)
                in1_init = self.conv_beforeU(I1, reuse=True)

            #LSTM1
            with tf.variable_scope('LSTM1'):
                with tf.variable_scope('unit1'):
                    cell = BasicConvLSTMCell([self.weight * 4, self.height * 4], [3, 3], 32)
                    state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
                with tf.variable_scope('unit2'):
                    cell2 = BasicConvLSTMCell([self.weight * 2, self.height * 2], [3, 3], 32)
                with tf.variable_scope('unit3'):
                    cell4 = BasicConvLSTMCell([self.weight, self.height], [3, 3], 32)

                with tf.variable_scope('lstm_1'):
                    y_1, state1 = cell(in1_init, state)
                with tf.variable_scope('lstm_2'):
                    down2_state1 = tf.nn.max_pool(state1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                    y_2, state2 = cell2(in2_init, down2_state1)
                with tf.variable_scope('lstm_3'):
                    down2_state2 = tf.nn.max_pool(state2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                    y_4, state4 = cell4(in4_init, down2_state2)
                with tf.variable_scope('lstm_4'):
                    y_4, state4 = cell4(in4_init, state4)
                with tf.variable_scope('lstm_5'):
                    up2_state4 = up_sample(state4)
                    y_2, state2 = cell2(in2_init, up2_state4 + state2)
                with tf.variable_scope('lstm_6'):
                    with tf.variable_scope('de1'):
                        up2_state2 = up_sample(state2)
                        y_1, state1 = cell(in1_init, up2_state2 + state1)

                y_4 = y_4 + in4_init
                y_2 = y_2 + in2_init
                y_1 = y_1 + in1_init


            ##U2
            with tf.variable_scope('beforeU2'):
                in4 = self.conv_beforeU(y_4, reuse=False)
                in2 = self.conv_beforeU(y_2, reuse=True)
                in1 = self.conv_beforeU(y_1, reuse=True)

            #LSTM2
            with tf.variable_scope('LSTM2'):
                with tf.variable_scope('unit1'):
                    cell = BasicConvLSTMCell([self.weight * 4, self.height * 4], [3, 3], 32)
                with tf.variable_scope('unit2'):
                    cell2 = BasicConvLSTMCell([self.weight * 2, self.height * 2], [3, 3], 32)
                with tf.variable_scope('unit3'):
                    cell4 = BasicConvLSTMCell([self.weight, self.height], [3, 3], 32)

                with tf.variable_scope('lstm_1'):
                    y_1, state1 = cell(in1, state1)
                with tf.variable_scope('lstm_2'):
                    down2_state1 = tf.nn.max_pool(state1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                    y_2, state2 = cell2(in2, down2_state1 + state2)
                with tf.variable_scope('lstm_3'):
                    down2_state2 = tf.nn.max_pool(state2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                    y_4, state4 = cell4(in4, down2_state2 + state4)
                with tf.variable_scope('lstm_4'):
                    y_4, state4 = cell4(in4, state4)
                with tf.variable_scope('lstm_5'):
                    up2_state4 = up_sample(state4)
                    y_2, state2 = cell2(in2, up2_state4 + state2)
                with tf.variable_scope('lstm_6'):
                    with tf.variable_scope('de1'):
                        up2_state2 = up_sample(state2)
                        y_1, state1 = cell(in1, up2_state2 + state1)

                y_1 = y_1 + in1
                y_2 = y_2 + in2
                y_4 = y_4 + in4

            with tf.variable_scope('Conv'):
                y1 = y_1
                y2 = y_2
                y4 = y_4
                out4_conv = conv(y4, 32, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv4')
                out2_conv = conv(y2, 32, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv2')
                out1_conv = conv(y1, 32, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv1')
                out4_conv = lrelu(out4_conv)
                out2_conv = lrelu(out2_conv)
                out1_conv = lrelu(out1_conv)

            with tf.variable_scope('last_conv'):
                out4 = self.last_conv(out4_conv, reuse=False)
                out2 = self.last_conv(out2_conv, reuse=True)
                out1 = self.last_conv(out1_conv, reuse=True)

                out4 = out4 + MS4
                out2 = out2 + MS2
                out1 = out1 + MS

        self.variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
        return out1, out2, out4

    def last_conv(self, input, reuse):
        with tf.variable_scope('last', reuse=reuse):
            x = conv(input, 4, kernel=3, stride=1, pad=1, pad_type='reflect', scope='last_conv')
            # x = tf.nn.tanh(x) / 2 + 0.5
            x = tf.nn.tanh(x)
        return x

    def conv_beforeU(self, input, reuse):
        with tf.variable_scope('beforeU', reuse=reuse):
            x = conv(input, 32, kernel=3, stride=1, pad=1, pad_type='reflect', scope='last_conv')
            x = lrelu(x)
        return x

    def inference_losses(self, hrms, hrms2, hrms4, gt, ms_org):
        def _tf_fspecial_gauss(size, sigma):
            x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

            x_data = np.expand_dims(x_data, axis=-1)
            x_data = np.expand_dims(x_data, axis=-1)

            y_data = np.expand_dims(y_data, axis=-1)
            y_data = np.expand_dims(y_data, axis=-1)

            x = tf.constant(x_data, dtype=tf.float32)
            y = tf.constant(y_data, dtype=tf.float32)

            g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
            return g / tf.reduce_sum(g)

        def grad(I):
            kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
            kernel = tf.expand_dims(kernel, axis=-1)
            kernel = tf.expand_dims(kernel, axis=-1)
            B, H, W, C = I.get_shape().as_list()
            for c in range(C):
                if c == 0:
                    grad = tf.nn.conv2d(tf.expand_dims(I[:, :, :, c], axis=-1), kernel, strides=[1, 1, 1, 1],
                                        padding='SAME')
                    # grad = abs(grad)
                else:
                    con = tf.nn.conv2d(tf.expand_dims(I[:, :, :, c], axis=-1), kernel, strides=[1, 1, 1, 1],
                                       padding='SAME')
                    # con = abs(con)
                    grad = tf.concat([grad, con], axis=-1)
            return grad

        def SSIM(img1, img2, size=11, sigma=1.5):
            window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
            k1 = 0.01
            k2 = 0.03
            L = 1  # depth of image (255 in case the image has a different scale)
            c1 = (k1 * L) ** 2
            c2 = (k2 * L) ** 2
            mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
            mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
            sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
            sigma1_2 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma1_2 + c2)) / (
                    (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
            value = tf.reduce_mean(ssim_map, axis=[1, 2, 3])
            return value

        GT2 = tf.nn.max_pool(gt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        GT4 = tf.nn.max_pool(GT2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        hrms2_c = tf.nn.max_pool(hrms2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        hrms_c = tf.nn.max_pool(hrms, ksize=[1, 2, 2, 1], strides=[1, 4, 4, 1], padding='SAME')

        loss_4 = (1 - tf.reduce_mean(SSIM(GT4[:, :, :, 0:1], hrms4[:, :, :, 0:1])) +
                  1 - tf.reduce_mean(SSIM(GT4[:, :, :, 1:2], hrms4[:, :, :, 1:2])) +
                  1 - tf.reduce_mean(SSIM(GT4[:, :, :, 2:3], hrms4[:, :, :, 2:3])) +
                  1 - tf.reduce_mean(SSIM(GT4[:, :, :, 3:4], hrms4[:, :, :, 3:4]))) * 40 + \
                 6 * tf.reduce_mean(tf.square(GT4 - hrms4)) + 40 * tf.reduce_mean(tf.square(grad(GT4) - grad(hrms4)))
        loss_2 = (1 - tf.reduce_mean(SSIM(GT2[:, :, :, 0:1], hrms2[:, :, :, 0:1])) +
                  1 - tf.reduce_mean(SSIM(GT2[:, :, :, 1:2], hrms2[:, :, :, 1:2])) +
                  1 - tf.reduce_mean(SSIM(GT2[:, :, :, 2:3], hrms2[:, :, :, 2:3])) +
                  1 - tf.reduce_mean(SSIM(GT2[:, :, :, 3:4], hrms2[:, :, :, 3:4]))) * 40 + \
                 1 * tf.reduce_mean(tf.square(GT2 - hrms2)) + 40 * tf.reduce_mean(tf.square(grad(GT2) - grad(hrms2)))\
                 + 5 * tf.reduce_mean(tf.square(hrms2_c - ms_org))
        loss_1 = (1 - tf.reduce_mean(SSIM(gt[:, :, :, 0:1], hrms[:, :, :, 0:1])) +
                  1 - tf.reduce_mean(SSIM(gt[:, :, :, 1:2], hrms[:, :, :, 1:2])) +
                  1 - tf.reduce_mean(SSIM(gt[:, :, :, 2:3], hrms[:, :, :, 2:3])) +
                  1 - tf.reduce_mean(SSIM(gt[:, :, :, 3:4], hrms[:, :, :, 3:4]))) * 40 + \
                 1 * tf.reduce_mean(tf.square(gt - hrms)) + 40 * tf.reduce_mean(tf.square(grad(gt) - grad(hrms)))\
                 + 5 * tf.reduce_mean(tf.square(hrms_c - ms_org))

        train_loss = 1 * loss_1 + 0.2 * loss_2 + 1 * loss_4

        # return train_loss, loss_1, loss_2, loss_4
        return train_loss


def conv(x, channels, kernel=3, stride=2, pad=0, pad_type='reflect', use_bias=True, sn=False, scope='conv',
         reuse=False):
    with tf.variable_scope(scope):
        if pad > 0:
            if (kernel - stride) % 2 == 0:
                pad_top = pad
                pad_bottom = pad
                pad_left = pad
                pad_right = pad

            else:
                pad_top = pad
                pad_bottom = kernel - stride - pad_top
                pad_left = pad
                pad_right = kernel - stride - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding='VALID',
                             reuse=reuse)
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias, reuse=reuse)
        return x


def deconv(x, filter_shape, output_shape, stride, trainable=True):
    filter_ = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable)
    return tf.nn.conv2d_transpose(
        value=x,
        filter=filter_,
        output_shape=output_shape,
        strides=[1, stride, stride, 1])


def lrelu(x, alpha=0.02):
    return tf.maximum(x, alpha * x)


def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)
