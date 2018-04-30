from lib.config import cfg
from lib.utils import print_tensor_shapes

import tensorflow as tf


def build_raw_text_embedding_batch(max_caption_length, name, batch_size=None):
    if batch_size is None:
        batch_size = cfg.CONST.BATCH_SIZE
    return tf.placeholder(tf.int32,
                          shape=[batch_size, max_caption_length],
                          name=name)


def build_desc_component_batch(desc_component_shape, name):
    return tf.placeholder(tf.float32,
                          shape=[None] + desc_component_shape,
                          name=name)


def build_noise_batch(noise_size, name):
    return tf.placeholder(tf.float32,
                          shape=[None, noise_size],
                          name=name)


def build_shape_batch(shape_batch_shape, name):
    """Builds a shape placeholder and returns it.
    """
    real_shapes_ph = tf.placeholder(tf.float32,
                                    shape=shape_batch_shape,
                                    name=name)

    print_tensor_shapes([real_shapes_ph], prefix='----> ')
    return real_shapes_ph
