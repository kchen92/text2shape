"""Modified from: https://github.com/haeusser/learning_by_association
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from lib.config import cfg


def build_walk_statistics(p_aba, equality_matrix):
    """Adds "walker" loss statistics to the graph.

    Args:
        p_aba: [N, N] matrix, where element [i, j] corresponds to the
            probability of the round-trip between supervised samples i and j.
            Sum of each row of 'p_aba' must be equal to one.
        equality_matrix: [N, N] boolean matrix, [i,j] is True, when samples
            i and j belong to the same class.
    """
    # Using the square root of the correct round trip probalilty as an estimate
    # of the current classifier accuracy.
    per_row_accuracy = 1.0 - tf.reduce_sum((equality_matrix * p_aba), 1)**0.5
    estimate_error = tf.reduce_mean(1.0 - per_row_accuracy, name=p_aba.name[:-2] + '_esterr')

    tf.summary.scalar('Stats_EstError', estimate_error)


def build_visit_loss(p, weight=1.0):
    """Add the "visit" loss to the model.

    Args:
        p: [N, M] tensor. Each row must be a valid probability distribution
            (i.e. sum to 1.0)
        weight: Loss weight.
    """
    visit_probability = tf.reduce_mean(p, [0], keep_dims=True, name='visit_prob')
    t_nb = tf.shape(p)[1]
    visit_loss = tf.losses.softmax_cross_entropy(
        tf.fill([1, t_nb], 1.0 / tf.cast(t_nb, tf.float32)),
        tf.log(1e-8 + visit_probability),
        weights=weight,
        scope='loss_visit')

    tf.summary.scalar('Loss_Visit', visit_loss)
    return visit_loss


def compute_matching_standard(a, b):
    """Use inner product.
    """
    match_ab = tf.matmul(a, b, transpose_b=True, name='match_ab')
    return match_ab, None, None, None, None


def compute_matching_mahalanobis(a, b, a_center=None, b_center=None, A=None):
    """Use Mahalanobis distance.
    """
    if a_center is None:
        a_dim = a.get_shape().as_list()[1]
        a_center = slim.model_variable('a_center', shape=[1, a_dim],
                                       initializer=tf.zeros_initializer())
    if b_center is None:
        b_dim = b.get_shape().as_list()[1]
        b_center = slim.model_variable('b_center', shape=[1, b_dim],
                                       initializer=tf.zeros_initializer())
    if A is None:
        A_shape = [a_dim, b_dim]
        assert A_shape[0] == A_shape[1]
        A_init_val = 0.2 * np.random.rand(A_shape[0], A_shape[1]) + np.eye(A_shape[0])
        A_initializer = tf.constant_initializer(A_init_val)
        A = slim.model_variable('A', shape=A_shape, initializer=A_initializer)

    M = (A + tf.transpose(A)) / 2.
    centered_a = a - a_center
    centered_b = b - b_center
    match_ab = tf.matmul(tf.matmul(centered_a, M), centered_b, transpose_b=True, name='match_ab')
    return match_ab, A, M, a_center, b_center


def build_semisup_loss(a, b, labels, walker_weight=1.0, visit_weight=1.0, a_center=None,
                       b_center=None, A=None):
    """Build the loss ops and tensors. Add semi-supervised classification loss to the model.

    The loss constist of two terms: "walker" and "visit".

    Args:
        a: [N, emb_size] tensor with supervised embedding vectors.
        b: [M, emb_size] tensor with unsupervised embedding vectors.
        labels : [N] tensor with labels for supervised embeddings.
        walker_weight: Weight coefficient of the "walker" loss.
        visit_weight: Weight coefficient of the "visit" loss.

    Returns:
        losses: Dictionary of losses, where the keys can be anything (such as 'generator_loss'),
            and the values are the actual losses. The sum of the values (losses) should be the
            total loss of the network.
    """
    # Build target probability distribution matrix based on uniform dist over correct labels
    equality_matrix = tf.equal(tf.reshape(labels, [-1, 1]), labels)
    equality_matrix = tf.cast(equality_matrix, tf.float32)
    p_target = (equality_matrix / tf.reduce_sum(equality_matrix, [1], keep_dims=True))

    if cfg.LBA.DIST_TYPE == 'standard':
        match_ab, A, M, a_center, b_center = compute_matching_standard(a, b)
    elif cfg.LBA.DIST_TYPE == 'mahalanobis':
        match_ab, A, M, a_center, b_center = compute_matching_mahalanobis(a, b, a_center=a_center,
                                                                          b_center=b_center, A=A)
    else:
        return ValueError('Please use a valid LBA.DIST_TYPE')
    p_ab = tf.nn.softmax(match_ab, name='p_ab')
    p_ba = tf.nn.softmax(tf.transpose(match_ab), name='p_ba')
    p_aba = tf.matmul(p_ab, p_ba, name='p_aba')

    build_walk_statistics(p_aba, equality_matrix)

    loss_aba = tf.losses.softmax_cross_entropy(
        p_target,
        tf.log(1e-8 + p_aba),
        weights=walker_weight,
        scope='loss_aba')
    visit_loss = build_visit_loss(p_ab, visit_weight)

    tf.summary.scalar('Loss_aba', loss_aba)

    losses = {
        'walker_loss': loss_aba,
        'visit_loss': visit_loss,
    }
    return losses, p_aba, p_target, A, M, a_center, b_center


def enforce_psd(A, M):
    e, v = tf.self_adjoint_eig(M)
    non_negative_e = tf.maximum(e, 0.)
    psd_M = tf.matmul(tf.matmul(v, tf.diag(non_negative_e)), tf.matrix_inverse(v))
    new_A = psd_M / 2
    assign_op = tf.assign(A, new_A)
    return assign_op


def build_lba_loss(text_encoder, shape_encoder, labels):
    """Build the LBA loss ops and tensors.

    Returns:
        losses: Dictionary of losses, where the keys can be anything (such as 'generator_loss'),
            and the values are the actual losses. The sum of the values (losses) should be the
            total loss of the network.
    """
    a_center = None
    b_center = None
    A = None
    enforce_psd_op_1 = None
    enforce_psd_op_2 = None
    if (cfg.LBA.MODEL_TYPE == 'MM') or (cfg.LBA.MODEL_TYPE == 'TST'):
        a = text_encoder.outputs['encoder_output']
        b = shape_encoder.outputs['encoder_output']
        with tf.variable_scope('tst_loss'):
            tst_loss, p_aba, p_target, A, M, a_center, b_center = build_semisup_loss(
                    a, b, labels,
                    walker_weight=cfg.LBA.WALKER_WEIGHT,
                    visit_weight=cfg.LBA.VISIT_WEIGHT)
        if cfg.LBA.DIST_TYPE == 'mahalanobis':
            enforce_psd_op_1 = enforce_psd(A, M)

    if (cfg.LBA.MODEL_TYPE == 'MM') or (cfg.LBA.MODEL_TYPE == 'STS'):
        b = text_encoder.outputs['encoder_output']
        a = shape_encoder.outputs['encoder_output']
        labels = np.array(range(cfg.CONST.BATCH_SIZE))
        if A is not None:
            A_transpose = tf.transpose(A)
        else:
            A_transpose = None
        with tf.variable_scope('sts_loss'):
            sts_loss, p_aba, p_target, A, M, _, _ = build_semisup_loss(
                    a, b, labels,
                    walker_weight=cfg.LBA.WALKER_WEIGHT,
                    visit_weight=cfg.LBA.VISIT_WEIGHT,
                    a_center=b_center,
                    b_center=a_center,
                    A=A_transpose)
        # if cfg.LBA.DIST_TYPE == 'mahalanobis':
        #     enforce_psd_op_2 = enforce_psd(A, M)

    losses = {
        'tst_walker_loss': tst_loss['walker_loss'] if cfg.LBA.MODEL_TYPE != 'STS' else tf.zeros([]),
        'tst_visit_loss': tst_loss['visit_loss'] if cfg.LBA.MODEL_TYPE != 'STS' else tf.zeros([]),
        'sts_walker_loss': sts_loss['walker_loss'] if cfg.LBA.MODEL_TYPE != 'TST' else tf.zeros([]),
        'sts_visit_loss': sts_loss['visit_loss'] if cfg.LBA.MODEL_TYPE != 'TST' else tf.zeros([]),
    }
    return losses, p_aba, p_target, enforce_psd_op_1, enforce_psd_op_2
