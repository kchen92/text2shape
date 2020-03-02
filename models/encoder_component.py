from lib.config import cfg
from lib.net_component import NetComponent, TextEncoderNetComponent
from lib.utils import compute_sequence_length, extract_last_output, open_pickle
import lib.layers as layers

import tensorflow as tf
import tf_slim as slim


class CNNRNNTextEncoder(TextEncoderNetComponent):
    """CNN-RNN Text Encoder network.
    """
    def __init__(self, is_training, reuse=False, name='text_encoder_example'):
        # Hard code the margin
        super(CNNRNNTextEncoder, self).__init__(is_training, reuse=reuse, name=name)
        self._embedding_size = 128
        self._margin = 1

    def build_architecture(self, inputs_dict):
        """Builds the RNN text encoder.

        Returns:
            rnn_outputs: A list of outputs for all RNNs. This is a list even if
                there is one RNN being constructed.
        """

        caption_batch = inputs_dict['caption_batch']
        embedding = inputs_dict['embedding_batch']
        seq_length = compute_sequence_length(caption_batch)

        # Build convolutions
        with slim.arg_scope([slim.convolution, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=tf.keras.regularizers.l2(0.5 * (0.0005))):

            net = slim.convolution(embedding, 128, 3, scope='conv1')
            net = slim.convolution(net, 128, 3, scope='conv2')
            net = tf.compat.v1.layers.batch_normalization(net, training=self.is_training)
            # net = slim.pool(net, 2, 'MAX')  # change the sequence length
            net = slim.convolution(net, 256, 3, scope='conv3')
            net = slim.convolution(net, 256, 3, scope='conv4')
            net = tf.compat.v1.layers.batch_normalization(net, training=self.is_training)
            # net = slim.pool(net, 2, 'MAX')

            rnn_cell = tf.compat.v1.nn.rnn_cell.GRUCell(num_units=256)
            # initial_state = rnn_cell.zero_state(self._batch_size, tf.float32)

            outputs, final_state = tf.compat.v1.nn.dynamic_rnn(cell=rnn_cell,
                                                     inputs=net,
                                                     sequence_length=seq_length,
                                                     dtype=tf.float32,
                                                     scope='rnn')

            net = extract_last_output(outputs, seq_length)
            net = slim.fully_connected(net, 256, scope='fc5')
            net = slim.fully_connected(net, 128, activation_fn=None, scope='fc6')

        return {'encoder_output': net}

    @property
    def embedding_size(self):
        return self._embedding_size


class ShapeEncoder(NetComponent):

    def __init__(self, is_training, reuse=False, name='shape_encoder'):
        super(ShapeEncoder, self).__init__(is_training, reuse=reuse, name=name)

    def build_architecture(self, inputs_dict):
        x = inputs_dict['shape_batch']
        if cfg.CONST.DATASET == 'shapenet':
            num_classes = 2  # Chair/table classification
        elif cfg.CONST.DATASET == 'primitives':
            train_inputs_dict = open_pickle(cfg.DIR.PRIMITIVES_TRAIN_DATA_PATH)
            val_inputs_dict = open_pickle(cfg.DIR.PRIMITIVES_VAL_DATA_PATH)
            test_inputs_dict = open_pickle(cfg.DIR.PRIMITIVES_TEST_DATA_PATH)
            f = lambda inputs_dict: list(inputs_dict['category_matches'].keys())
            categories = f(train_inputs_dict) + f(val_inputs_dict) + f(test_inputs_dict)
            categories = list(set(categories))
            num_classes = len(categories)
        else:
            raise ValueError('Please select a valid dataset')

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
