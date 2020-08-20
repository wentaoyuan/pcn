'''
MIT License

Copyright (c) 2018 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import tensorflow as tf
from tf_util import mlp, mlp_conv, chamfer, add_train_summary, add_valid_summary


class Model:
    def __init__(self, inputs, gt, alpha):
        self.num_output_points = 16384
        self.features = self.create_encoder(inputs)
        self.outputs = self.create_decoder(self.features)
        self.loss, self.update = self.create_loss(self.outputs, gt)
        self.visualize_ops = [inputs[0], self.outputs[0], gt[0]]
        self.visualize_titles = ['input', 'output', 'ground truth']

    def create_encoder(self, inputs):
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs, [128, 256])
            features_global = tf.reduce_max(features, axis=1, keep_dims=True, name='maxpool_0')
            features = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1])], axis=2)
        with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
            features = mlp_conv(features, [512, 1024])
            features = tf.reduce_max(features, axis=1, name='maxpool_1')
        return features

    def create_decoder(self, features):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            outputs = mlp(features, [1024, 1024, self.num_output_points * 3])
            outputs = tf.reshape(outputs, [-1, self.num_output_points, 3])
        return outputs

    def create_loss(self, outputs, gt):
        loss = chamfer(outputs, gt)
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)
        return loss, update_loss
