from lib.config import cfg
from lib.optimizers import get_optimizer
from lib.utils import get_trainable_variables_by_scope, print_tensor_shapes

import tensorflow as tf
import tf_slim as slim


def build_default_train_op(global_step, loss, trainable_vars):
    learning_rate = tf.compat.v1.train.exponential_decay(cfg.TRAIN.LEARNING_RATE,
                                               global_step,
                                               cfg.TRAIN.DECAY_STEPS,
                                               cfg.TRAIN.DECAY_RATE,
                                               staircase=cfg.TRAIN.STAIRCASE,
                                               name='lr_decay')
    tf.compat.v1.summary.scalar('learning_rate', learning_rate)

    optimizer = get_optimizer(cfg.TRAIN.OPTIMIZER, learning_rate, name='optimizer')
    train_op = slim.learning.create_train_op(
        loss,
        optimizer,
        global_step=global_step,
        variables_to_train=trainable_vars)
    return train_op


def default_train_step(sess, step, loss, train_op, feed_dict, summary_op, summary_writer):
    if summary_op is not None and ((step + 1) % cfg.TRAIN.SUMMARY_FREQ == 0):
        cur_loss, _, summary = sess.run([loss, train_op, summary_op], feed_dict=feed_dict)
        summary_writer.add_summary(summary, (step + 1))
    else:
        cur_loss, _ = sess.run([loss, train_op], feed_dict=feed_dict)
    return cur_loss


class Network(object):
    """Generic neural network abstract class.
    """
    def __init__(self, inputs_dict, is_training, reuse=False, name='network'):
        self._name = name
        self._inputs_dict = inputs_dict
        self._is_training = is_training
        self._reuse = reuse

        self._placeholders = None
        self._outputs = None
        self._vars_to_restore = None
        self._losses = None
        self._total_loss = None

        self.build_model()

        # Training
        self.train_ops = None
        self.summary_op = None
        self.summary_writer = None

        # Add summaries
        if self.losses is not None:  # Only None for generator and discriminator
            for loss in self.losses.values():
                if isinstance(loss, dict):
                    for x in loss.values():
                        tf.compat.v1.losses.add_loss(x)
                else:
                    tf.compat.v1.losses.add_loss(loss)
            if self.total_loss is not None:
                tf.compat.v1.summary.scalar('total_loss', self.total_loss)

    def build_placeholders(self):
        """Build the placeholders.
        """
        raise NotImplementedError('Must be implemented by a subclass.')

    def preprocess_inputs(self):
        """Preprocess the inputs.
        """
        raise NotImplementedError('Must be implemented by a subclass.')

    def build_architecture(self):
        """Build the architecture, not including placeholders and preprocessing.
        """
        raise NotImplementedError('Must be implemented by a subclass.')

    def build_model(self):
        """Build the network, including placeholders, preprocessing, and architecture, and loss.
        """
        print('building network:', self.name)

        with tf.compat.v1.variable_scope(self.name, reuse=self.reuse) as sc:
            self.build_placeholders()
            self.preprocess_inputs()
            self._outputs = self.build_architecture()

            # Create summaries for network outputs
            if isinstance(self._outputs, dict):
                for k, v in self._outputs.items():
                    if isinstance(v, tf.Tensor):
                        tf.compat.v1.summary.histogram(k + '_hist_summary', v)

            print('--> building loss')
            self._losses = self.build_loss()
            if self.losses is not None:
                self._total_loss = tf.add_n(list(self._losses.values()), name='total_loss')

            self._vars_to_restore = slim.get_variables_to_restore(include=[sc.name])

        print('--> done building {}'.format(self.name))

    def get_trainable_variables(self):
        """Return the trainable variables in the specified scope.

        Args:
            scope_name: Name of the scope as a string.

        Returns:
            trainable_vars: TensorFlow variables that are trainable and in the
                specified scope.
        """
        trainable_vars = get_trainable_variables_by_scope(self.name)
        return trainable_vars

    def build_loss(self):
        """Build the loss ops and tensors.

        Returns:
            losses: Dictionary of losses, where the keys can be anything (such as 'generator_loss'),
                and the values are the actual losses. The sum of the values (losses) should be the
                total loss of the network.
        """
        raise NotImplementedError('Must be implemented by a subclass.')

    def set_losses_and_total_loss(self, losses):
        """Sets the self.losses and self.total_loss of the network.

        Args:
            losses: Dictionary of losses, similar to the output of self.build_loss().
        """
        with tf.compat.v1.variable_scope(self.name):
            self._losses = losses
            self._total_loss = tf.add_n(list(self._losses.values()), name='total_loss')

            tf.compat.v1.summary.scalar('total_loss', self.total_loss)

        # Add losses to losses collection
        for loss in self.losses.values():
            tf.compat.v1.losses.add_loss(loss)

    def build_summary_ops(self, graph):
        """Build summary ops. Add summaries for variables, weights, biases, activations, and losses.

        Returns:
            summary_op: The (merged) summary op.
            summary_writer: A summary writer.
        """
        # add summaries
        slim.summarize_variables()
        slim.summarize_weights()
        slim.summarize_biases()
        slim.summarize_activations()
        slim.summarize_collection(tf.compat.v1.GraphKeys.LOSSES)

        with tf.compat.v1.name_scope('summary_ops'):
            summary_op = tf.compat.v1.summary.merge_all()
            summary_writer = tf.compat.v1.summary.FileWriter(cfg.DIR.LOG_PATH, graph=graph)
        self.summary_op = summary_op
        self.summary_writer = summary_writer

    def build_train_ops(self, global_step):
        """Builds the train op.

        Returns:
            train_op: A dictionary containing the train op.
        """
        trainable_vars = self.get_trainable_variables()
        print('trainable vars in {}'.format(self.name))
        print_tensor_shapes(trainable_vars, prefix='-->')
        train_op = build_default_train_op(global_step, self.total_loss, trainable_vars)
        self.train_ops = {'train_op': train_op}

    def get_feed_dict(self, minibatch):
        """Parse the minibatch data and return the feed dict.

        Args:
            minibatch: A dictionary of minibatch of data from the data process.

        Returns:
            feed_dict: A feed dict for both the generator and discriminator.
            batch_size: The size of the current minibatch.
        """
        raise NotImplementedError('Must be implemented by a subclass.')

    def get_minibatch(self, train_queue, data_timer=None):
        """Gets the next minibatch from the data process.
        """
        if data_timer is not None:
            data_timer.tic()
            minibatch = train_queue.get()
            data_timer.toc()
        else:
            minibatch = train_queue.get()
        return minibatch

    def train_step(self, sess, train_queue, step, data_timer=None):
        """Executes a train step, including saving the summaries if appropriate.

        Args:
            sess: Current session.
            train_queue: Data queue containing train set minibatches.
            step: The current training iteration (0-based indexing).

        Returns:
            loss: Loss of the network (for just one minibatch).
        """
        minibatch = self.get_minibatch(train_queue, data_timer=data_timer)
        feed_dict, batch_size = self.get_feed_dict(minibatch)
        cur_loss = default_train_step(sess, step, self.total_loss, self.train_ops['train_op'],
                                      feed_dict, self.summary_op, self.summary_writer)
        return {'loss': cur_loss}

    def forward_pass(self, sess, minibatch):
        """Computes a forward pass of the network for the given minibatch.

        Args:
            sess: Current session.
            minibatch: A minibatch of data.

        Returns:
            outputs: Outputs, as defined by the network.
        """
        feed_dict, batch_size = self.get_feed_dict(minibatch)
        outputs = sess.run(list(self.outputs.values()), feed_dict=feed_dict)
        outputs_dict = {}
        for idx, key in enumerate(self.outputs.keys()):
            outputs_dict[key] = outputs[idx]
        return outputs_dict

    @property
    def total_loss(self):
        return self._total_loss

    @property
    def placeholders(self):
        raise NotImplementedError('Must be implemented by a subclass.')

    @property
    def name(self):
        return self._name

    @property
    def outputs(self):
        return self._outputs

    @property
    def is_training(self):
        return self._is_training

    @property
    def losses(self):
        return self._losses

    @property
    def vars_to_restore(self):
        return self._vars_to_restore

    @property
    def reuse(self):
        return self._reuse

    @property
    def inputs_dict(self):
        return self._inputs_dict
