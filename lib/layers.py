from lib.config import cfg
import numpy as np

from tensorflow.python.ops import variable_scope
from tensorflow.compat.v1 import initializers
from tensorflow.python.ops import init_ops

import tensorflow as tf
import tf_slim as slim


def layer(op):
    """Decorator for network layers. Adds summaries for the outputs.
    """
    def layer_decorated(*args, **kwargs):
        output = op(*args, **kwargs)

        # Add histogram summary
        layer_name = None
        if 'name' in kwargs:
            assert 'scope' not in kwargs
            layer_name = kwargs['name']
        elif 'scope' in kwargs:
            assert 'name' not in kwargs
            layer_name = kwargs['scope']
        if layer_name is not None:
            tf.compat.v1.summary.histogram(layer_name + '_hist_summary', output)

        return output

    return layer_decorated


@layer
def unpooling_3d(voxel_tensor_batch, reuse, name='unpooling_3d'):
    """Unpooling 3D. Only upsamples by a factor of 2 in every spatial dimension.

    Args:
        voxel_tensor_batch: [batch, x, y, z, channel].
    """
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        # Use fixed batch size
        batch_shape = voxel_tensor_batch.get_shape().as_list()
        batch_shape[0] = cfg.CONST.BATCH_SIZE
        voxel_tensor_batch = tf.reshape(voxel_tensor_batch, batch_shape)

        x = voxel_tensor_batch[0]
        x_tile = tf.tile(voxel_tensor_batch, [1, 2, 2, 2, 1])

        # Get shape as int
        vx_shape = x.get_shape().as_list()
        batch_tile_shape = x_tile.get_shape().as_list()

        # Perform upsampling
        indices = []
        for i in range(vx_shape[0]):
            indices.append(i)
            indices.append(i + vx_shape[0])

        gather_indices = np.ones(batch_tile_shape + [5])
        for batch_idx in range(batch_tile_shape[0]):
            for x1, x_idx in enumerate(indices):
                for y1, y_idx in enumerate(indices):
                    for z1, z_idx in enumerate(indices):
                        for c_idx in range(batch_tile_shape[4]):
                            gather_indices[batch_idx, x1, y1, z1, c_idx] = np.array([batch_idx, x_idx, y_idx, z_idx, c_idx])
        gather_indices = gather_indices.astype(np.int32).tolist()
        y = tf.gather_nd(x_tile, gather_indices)
    print('\t\t{scope}'.format(scope=name), y.get_shape())
    return y


@layer
def conv3d(*args, **kwargs):
    """Adds a 3D convolution layer and prints the output shape.
    """
    output = tf.compat.v1.layers.conv3d(*args, **kwargs)
    # tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, output)
    if 'name' in kwargs:
        print('\t\t{scope}'.format(scope=kwargs['name']), output.get_shape())
    return output


@layer
def avg_pooling3d(x, name, axis=[1, 2, 3], keep_dims=False):
    # output = tf.layers.average_pooling3d(*args, **kwargs)
    # output = tf.nn.avg_pool3d(*args, **kwargs)
    output = tf.reduce_mean(input_tensor=x, axis=axis, keepdims=keep_dims, name=name)
    print('\t\t{scope}'.format(scope=name), output.get_shape())
    return output


@layer
def dense(*args, **kwargs):
    """Adds a fully connected layer and prints the output shape.
    """
    output = tf.compat.v1.layers.dense(*args, **kwargs)
    tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.ACTIVATIONS, output)
    if 'name' in kwargs:
        print('\t\t{scope}'.format(scope=kwargs['name']), output.get_shape())
    return output


def expand_dims(*args, **kwargs):
    output = tf.expand_dims(*args, **kwargs)
    if 'name' in kwargs:
        print('\t\t{scope}'.format(scope=kwargs['name']), output.get_shape())
    return output


def tile(*args, **kwargs):
    output = tf.tile(*args, **kwargs)
    if 'name' in kwargs:
        print('\t\t{scope}'.format(scope=kwargs['name']), output.get_shape())
    return output


def concat(*args, **kwargs):
    output = tf.concat(*args, **kwargs)
    if 'name' in kwargs:
        print('\t\t{scope}'.format(scope=kwargs['name']), output.get_shape())
    return output


def reshape(net, shape, scope):
    """Reshapes the input.
    """
    net = tf.reshape(net, shape=shape, name=scope)
    print('\t\t{scope}'.format(scope=scope), net.get_shape())
    return net


def softmax(logits, dim=-1, name=None):
    output = tf.nn.softmax(logits, axis=dim, name=name)
    if 'name' is not None:
        print('\t\t{scope}'.format(scope=name), output.get_shape())
    return output


def squeeze(input_tensor, axis=None, name=None):
    """Adds a squeeze layer (remove axes with dimension size 1).
    """
    net = tf.squeeze(input_tensor, axis=axis, name=name)  # tf 0.12.0rc: squeeze_dims -> axis
    print('\t\t{scope}'.format(scope=name), net.get_shape())
    return net


def smoothed_metric_loss(input_tensor, name='smoothed_metric_loss', margin=1):
    """
    input_tensor: require a tensor with predefined dimensions (No None dimension)
    """
    with tf.compat.v1.variable_scope(name):
        # Song et al., Deep Metric Learning via Lifted Structured Feature Embedding
        # Define feature X \in \mathbb{R}^{N \times C}
        X = input_tensor
        m = margin

        # Compute the pairwise distance
        Xe = tf.expand_dims(X, 1)
        if cfg.LBA.COSINE_DIST is True:
            assert (cfg.LBA.NORMALIZE is True) or (cfg.LBA.INVERTED_LOSS is True)
            assert ((cfg.LBA.NORMALIZE is True) and (margin < 1)) or (cfg.LBA.INVERTED_LOSS is True)
            D = tf.reduce_sum(input_tensor=tf.multiply(Xe, tf.transpose(a=Xe, perm=(1, 0, 2))), axis=2, keepdims=False)
            if cfg.LBA.INVERTED_LOSS is False:
                D = 1. - D
            else:
                D /= 128.
        else:
            Dsq = tf.reduce_sum(input_tensor=tf.square(Xe - tf.transpose(a=Xe, perm=(1, 0, 2))), axis=2, keepdims=False)
            D = tf.sqrt(Dsq + 1e-8)
        if cfg.LBA.INVERTED_LOSS is True:
            expmD = tf.exp(m + D)
        else:
            expmD = tf.exp(m - D)

        # Compute the loss
        # Assume that the input data is aligned in a way that two consecutive data form a pair
        batch_size = cfg.CONST.BATCH_SIZE

        # L_{ij} = \log (\sum_{i, k} exp\{m - D_{ik}\} + \sum_{j, l} exp\{m - D_{jl}\}) + D_{ij}
        # L = \frac{1}{2|P|}\sum_{(i,j)\in P} \max(0, J_{i,j})^2
        J_all = []
        for pair_ind in range(batch_size // 2):
            i = pair_ind * 2
            j = i + 1
            ind_rest = np.hstack([np.arange(0, pair_ind * 2),
                                  np.arange(pair_ind * 2 + 2, batch_size)])

            inds = [[i, k] for k in ind_rest]
            inds.extend([[j, l] for l in ind_rest])
            if cfg.LBA.INVERTED_LOSS is True:
                J_ij = tf.math.log(tf.reduce_sum(input_tensor=tf.gather_nd(expmD, inds))) - tf.gather_nd(D, [[i, j]])
            else:
                J_ij = tf.math.log(tf.reduce_sum(input_tensor=tf.gather_nd(expmD, inds))) + tf.gather_nd(D, [[i, j]])
            J_all.append(J_ij)

        J_all = tf.convert_to_tensor(value=J_all)
        loss = tf.divide(tf.reduce_mean(input_tensor=tf.square(tf.maximum(J_all, 0.))), 2.0, name='metric_loss')
        tf.compat.v1.losses.add_loss(loss)

    return loss


def triplet_loss(input_tensor, name='triplet_loss', margin=1):
    """Triplet loss.

    Args:
        input_tensor: require a tensor with predefined dimensions (No None dimension)
    """
    with tf.compat.v1.variable_scope(name):
        # Song et al., Deep Metric Learning via Lifted Structured Feature Embedding
        # Define feature X \in \mathbb{R}^{N \times C}
        X = input_tensor
        m = margin

        # Compute the pairwise distance
        Xe = tf.expand_dims(X, 1)
        Dsq = tf.reduce_sum(input_tensor=tf.square(Xe - tf.transpose(a=Xe, perm=(1, 0, 2))), axis=2)
        D = tf.sqrt(Dsq + 1e-8)
        mD = m - D

        # Compute the loss
        # Assume that the input data is aligned in a way that two consecutive data form a pair
        batch_size, _ = X.get_shape().as_list()

        # L_{ij} = \log (\sum_{i, k} exp\{m - D_{ik}\} + \sum_{j, l} exp\{m - D_{jl}\}) + D_{ij}
        # L = \frac{1}{2|P|}\sum_{(i,j)\in P} \max(0, J_{i,j})^2
        J_all = []
        for pair_ind in range(batch_size // 2):
            i = pair_ind * 2
            j = i + 1
            ind_rest = np.hstack([np.arange(0, pair_ind * 2),
                                  np.arange(pair_ind * 2 + 2, batch_size)])

            inds = [[i, k] for k in ind_rest]
            inds.extend([[j, l] for l in ind_rest])
            J_ij = tf.reduce_max(input_tensor=tf.gather_nd(mD, inds)) + tf.gather_nd(D, [[i, j]])
            J_all.append(J_ij)

        J_all = tf.convert_to_tensor(value=J_all)
        loss = tf.divide(tf.reduce_mean(input_tensor=tf.square(tf.maximum(J_all, 0))), 2.0, name='metric_loss')
        tf.compat.v1.losses.add_loss(loss)
    return loss


@layer
def cross_entropy_sequence_loss(logits, targets, sequence_length, max_length=None):
    """Calculates the per-example cross-entropy loss for a sequence of logits and
    masks out all losses passed the sequence length.

    Args:
        logits: Logits of shape `[T, B, vocab_size]`
        targets: Target classes of shape `[T, B]`
        sequence_length: An int32 tensor of shape `[B]` corresponding
                to the length of each input

    Returns:
        Loss: A tensor of shape [T, B] that contains the loss per example, per time step.
    """
    with tf.compat.v1.name_scope("cross_entropy_sequence_loss"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=targets)

        if max_length is None:
            max_length = targets.get_shape().as_list()[1]

        # Mask out the losses we don't care about
        loss_mask = tf.sequence_mask(sequence_length, max_length)
        losses = losses * tf.cast(loss_mask, dtype=tf.float32)
        loss_per_batch = tf.compat.v1.div(tf.reduce_sum(input_tensor=losses, axis=1), tf.reduce_sum(input_tensor=tf.cast(loss_mask, dtype=tf.float32), axis=1))
    return tf.reduce_mean(input_tensor=loss_per_batch)


@slim.add_arg_scope
def conv3d_transpose(
        inputs,
        num_output_channels,
        filter_size,
        stride,
        padding='SAME',
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.glorot_uniform(),
        weights_regularizer=None,
        biases_initializer=init_ops.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        trainable=True,
        scope=None):
    """Adds a convolution3d_transpose with an optional batch normalization layer.
    """
    if stride == 1:
        pass
    elif stride == 2:
        pass
    else:
        raise NotImplementedError('Stride > 2 not supported.')

    assert padding == 'SAME'  # Only SAME padding is currently supported

    # Compute weight/kernel dimensions
    num_input_channels = int(inputs.get_shape()[4])
    kernel_shape = [filter_size, filter_size, filter_size, num_output_channels, num_input_channels]

    # Compute output shape
    input_d_h_w = [int(inputs.get_shape()[i]) for i in range(1, 4)]
    output_d_h_w = [stride * i for i in input_d_h_w]
    output_shape = [cfg.CONST.BATCH_SIZE] + output_d_h_w + [num_output_channels]

    with variable_scope.variable_scope(scope, 'conv3d_transpose', reuse=reuse):
        weights = slim.model_variable('weights',
                                      shape=kernel_shape,
                                      dtype=tf.float32,
                                      initializer=weights_initializer,
                                      regularizer=weights_regularizer,
                                      trainable=trainable)

        if not normalizer_fn and biases_initializer:
            biases = None
        else:
            biases = slim.model_variable('biases',
                                         shape=output_shape,
                                         dtype=tf.float32,
                                         initializer=biases_initializer,
                                         regularizer=biases_regularizer,
                                         trainable=trainable)

        outputs = tf.nn.conv3d_transpose(inputs,
                                         weights,
                                         output_shape,
                                         [1, stride, stride, stride, 1],
                                         padding='SAME')

        if biases is not None:
            outputs = tf.add(outputs, biases)

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

    if scope is not None:
        print('\t\t{scope}'.format(scope=scope), outputs.get_shape())
        return outputs


def flatten(*args, **kwargs):
    output = slim.flatten(*args, **kwargs)
    if 'scope' in kwargs:
        print('\t\t{scope}'.format(scope=kwargs['scope']), output.get_shape())
    return output


# ------activation fns-------

@layer
def relu(x, name='relu'):
    x = tf.nn.relu(x, name=name)
    return x


# These are not activation functions because they have additional parameters.
#  However, they return an activation function with the specified parameters
#  'baked in'.
def leaky_relu(leak=0.2, name='leaky_relu'):
    @layer
    def lrelu(x, leak=leak, name=name):
        with tf.compat.v1.variable_scope(name):
            f1 = 0.5 * (1. + leak)
            f2 = 0.5 * (1. - leak)
            output = f1 * x + f2 * tf.abs(x)
            tf.compat.v1.summary.histogram(name + '_hist_summary', output)
        return output
    return lrelu


def test_unpooling_3d():
    placeholder = tf.compat.v1.placeholder(tf.float32, shape=[cfg.CONST.BATCH_SIZE, 8, 8, 8, 16])
    output = unpooling_3d(placeholder, name='unpooling_3d')
    with tf.compat.v1.Session() as sess:
        x = np.random.rand(cfg.CONST.BATCH_SIZE, 8, 8, 8, 16)
        out = sess.run(output, feed_dict={placeholder: x})


if __name__ == '__main__':
    test_unpooling_3d()
