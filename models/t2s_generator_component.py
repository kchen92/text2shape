from lib.config import cfg
from lib.net_component import NetComponent
import lib.layers as layers

import tensorflow as tf


class Text2ShapeGenerator1(NetComponent):

    def __init__(self, is_training, reuse=False, name='t2s_generator_1'):
        super(Text2ShapeGenerator1, self).__init__(is_training, reuse=reuse, name=name)

    def build_architecture(self, inputs_dict, last_activation=tf.sigmoid):
        with tf.compat.v1.variable_scope('architecture'):
            x = inputs_dict['text_encoding_with_noise']
            print('\t\tinput', x.get_shape())

            # Conv1
            x = layers.dense(
                x,
                units=512 * 4 * 4 * 4,
                activation=None,
                name='fc1',
                reuse=self.reuse
            )
            x = tf.compat.v1.layers.batch_normalization(
                x,
                training=self.is_training,
                name='fc1_batch_norm',
                reuse=self.reuse
            )
            x = layers.relu(x, name='fc1_relu')
            x = layers.reshape(x, shape=[-1, 4, 4, 4, 512], scope='reshape_fc1')

            # Conv2
            x = layers.conv3d_transpose(
                x,
                num_output_channels=512,
                filter_size=4,
                stride=1,
                activation_fn=None,
                scope='conv_transpose2',
                reuse=self.reuse
            )
            x = tf.compat.v1.layers.batch_normalization(
                x,
                training=self.is_training,
                name='conv_tranpose2_batch_norm',
                reuse=self.reuse
            )
            x = layers.relu(x, name='conv_transpose2_relu')

            # Conv3
            x = layers.conv3d_transpose(
                x,
                num_output_channels=256,
                filter_size=4,
                stride=2,
                activation_fn=None,
                scope='conv_transpose3',
                reuse=self.reuse
            )
            x = tf.compat.v1.layers.batch_normalization(
                x,
                training=self.is_training,
                name='conv_tranpose3_batch_norm',
                reuse=self.reuse
            )
            x = layers.relu(x, name='conv_transpose3_relu')

            # Conv4
            x = layers.conv3d_transpose(
                x,
                num_output_channels=128,
                filter_size=4,
                stride=2,
                activation_fn=None,
                scope='conv_transpose4',
                reuse=self.reuse
            )
            x = tf.compat.v1.layers.batch_normalization(x, training=self.is_training, name='conv_transpose4_batch_norm',
                                              reuse=self.reuse)
            x = layers.relu(x, name='conv_transpose4_relu')

            # Conv5
            logits = layers.conv3d_transpose(
                x,
                num_output_channels=inputs_dict['num_output_channels'],
                filter_size=4,
                stride=2,
                padding='SAME',
                activation_fn=None,
                scope='conv_transpose6',
                reuse=self.reuse
            )

            sigmoid_output = last_activation(logits)

        return {'sigmoid_output': sigmoid_output, 'logits': logits}
