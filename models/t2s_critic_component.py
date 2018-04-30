from lib.config import cfg
from lib.net_component import CriticNetComponent
import lib.layers as layers

import tensorflow as tf


class Text2ShapeDiscriminator2(CriticNetComponent):

    def __init__(self, is_training, reuse=False, name='t2s_discriminator_2'):
        super(Text2ShapeDiscriminator2, self).__init__(is_training, reuse=reuse, name=name)

    def build_architecture(self, inputs_dict):
        print('--> building 3D GAN discriminator architecture')

        leaky_relu_fn = layers.leaky_relu(leak=0.2)

        with tf.variable_scope('architecture', reuse=self.reuse):
            x = inputs_dict['shape_batch']
            print('\t\tinput', x.get_shape())

            x = layers.conv3d(x, filters=64, kernel_size=4, strides=2, padding='SAME',
                              activation=None, name='conv1', reuse=self.reuse)
            x = leaky_relu_fn(x, name='conv1_lrelu')
            x = layers.conv3d(x, filters=128, kernel_size=4, strides=2, padding='SAME',
                              activation=None, name='conv2', reuse=self.reuse)
            x = leaky_relu_fn(x, name='conv2_lrelu')
            x = layers.conv3d(x, filters=256, kernel_size=4, strides=2, padding='SAME',
                              activation=None, name='conv3', reuse=self.reuse)
            x = leaky_relu_fn(x, name='conv3_lrelu')

            # Text embedding input
            embedding_fc_dim = 256

            # Add FC layer
            embedding_batch = inputs_dict['text_encoding_without_noise']
            fc_embedding_output = layers.dense(embedding_batch, embedding_fc_dim, activation=None,
                                               name='fc_embedding_1')
            fc_embedding_output = leaky_relu_fn(fc_embedding_output, name='fc_embedding_1_lrelu')

            # Add FC layer
            fc_embedding_output = layers.dense(fc_embedding_output, embedding_fc_dim,
                                               activation=None, name='fc_embedding_2')
            fc_embedding_output = leaky_relu_fn(fc_embedding_output, name='fc_embedding_2_lrelu')

            x = layers.conv3d(x, filters=512, kernel_size=4, strides=2, padding='SAME',
                              activation=None, name='conv4', reuse=self.reuse)
            x = leaky_relu_fn(x, name='conv4_lrelu')

            x = layers.conv3d(x, filters=256, kernel_size=2, strides=2, padding='SAME',
                              activation=None, name='conv5', reuse=self.reuse)
            x = leaky_relu_fn(x, name='conv5_lrelu')

            x = layers.reshape(x, [cfg.CONST.BATCH_SIZE, -1], scope='reshape_to_concat')
            x = layers.concat([x, fc_embedding_output], axis=1, name='concat_text_shape')

            # Add FC layer
            x = layers.dense(x, 128, activation=None, name='fc6')
            x = leaky_relu_fn(x, name='fc6_lrelu')

            # Add FC layer
            x = layers.dense(x, 64, activation=None, name='fc7')
            x = leaky_relu_fn(x, name='fc7_lrelu')

            # Add FC layer
            logits = layers.dense(x, 1, activation=None, name='fc8')

            sigmoid_output = tf.sigmoid(logits)

        return {'sigmoid_output': sigmoid_output, 'logits': logits}
