import tensorflow as tf


def adam_optimizer(learning_rate, name=None):
    return tf.train.AdamOptimizer(learning_rate, beta1=0.5, name=name)


def adam_optimizer_iwgan(learning_rate, name=None):
    return tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.9, name=name)


def rms_prop_optimizer(learning_rate, name=None):
    return tf.train.RMSPropOptimizer(learning_rate, name=name)


OPTIMIZER_PAIRS = {
    'adam': adam_optimizer,
    'adam_iwgan': adam_optimizer_iwgan,
    'rms_prop': rms_prop_optimizer
}


def get_optimizer(optimizer_name, learning_rate, name=None):
    optimizer_fn = OPTIMIZER_PAIRS[optimizer_name]
    return optimizer_fn(learning_rate, name)
