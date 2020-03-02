from lib.classifier import Classifier
import lib.layers as layers

import tensorflow as tf


class Classifier1(Classifier):

    def __init__(self, inputs_dict, is_training, reuse=False, name='classifier_1'):
        super(Classifier1, self).__init__(inputs_dict, is_training, reuse=reuse, name=name)

    def build_architecture(self):
        num_classes = self.num_classes
        x = self.placeholders['shape_batch']

        x = layers.conv3d(x, 64, 3, strides=2, padding='same', name='conv1', reuse=self.reuse)
        x = tf.compat.v1.layers.batch_normalization(x, training=self.is_training, name='conv1_batch_norm',
                                          reuse=self.reuse)
        x = layers.relu(x, name='conv1_relu')
        x = layers.conv3d(x, 128, 3, strides=2, padding='same', name='conv2', reuse=self.reuse)
        x = tf.compat.v1.layers.batch_normalization(x, training=self.is_training, name='conv2_batch_norm',
                                          reuse=self.reuse)
        x = layers.relu(x, name='conv2_relu')
        x = layers.conv3d(x, 256, 3, strides=2, padding='same', name='conv3', reuse=self.reuse)
        x = tf.compat.v1.layers.batch_normalization(x, training=self.is_training, name='conv3_batch_norm',
                                          reuse=self.reuse)
        x = layers.relu(x, name='conv3_relu')
        x = layers.avg_pooling3d(x, name='avg_pool4')
        encoder_output = x
        x = layers.dense(x, num_classes, name='fc5', reuse=self.reuse)
        prob = layers.softmax(x, name='softmax_layer')

        output_dict = {
            'logits': x,
            'probabilities': prob,
            'encoder_output': encoder_output,
        }

        return output_dict


class Classifier128(Classifier):
    """Classifier with 128 dim embeddings.
    """

    def __init__(self, inputs_dict, is_training, reuse=False, name='classifier_128'):
        super(Classifier128, self).__init__(inputs_dict, is_training, reuse=reuse, name=name)

    def build_architecture(self):
        x = self.placeholders['shape_batch']
        num_classes = self.num_classes  # Chair/table classification

        x = layers.conv3d(x, 64, 3, strides=2, padding='same', name='conv1', reuse=self.reuse)
        x = tf.compat.v1.layers.batch_normalization(x, training=self.is_training, name='conv1_batch_norm',
                                          reuse=self.reuse)
        x = layers.relu(x, name='conv1_relu')
        x = layers.conv3d(x, 128, 3, strides=2, padding='same', name='conv2', reuse=self.reuse)
        x = tf.compat.v1.layers.batch_normalization(x, training=self.is_training, name='conv2_batch_norm',
                                          reuse=self.reuse)
        x = layers.relu(x, name='conv2_relu')
        x = layers.conv3d(x, 256, 3, strides=2, padding='same', name='conv3', reuse=self.reuse)
        x = tf.compat.v1.layers.batch_normalization(x, training=self.is_training, name='conv3_batch_norm',
                                          reuse=self.reuse)
        x = layers.relu(x, name='conv3_relu')
        x = layers.avg_pooling3d(x, name='avg_pool4')
        x = layers.dense(x, 128, name='fc5', reuse=self.reuse)
        encoder_output = x
        x = layers.dense(x, num_classes, name='fc6', reuse=self.reuse)
        prob = layers.softmax(x, name='softmax_layer')

        output_dict = {
            'logits': x,
            'probabilities': prob,
            'encoder_output': encoder_output,
        }

        return output_dict
