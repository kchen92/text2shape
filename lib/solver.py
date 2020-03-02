import numpy as np
import os
import tensorflow as tf
import tf_slim as slim

from datetime import datetime

from lib.classifier import Classifier
from lib.lba import LBA
from lib.config import cfg
from lib.data_process import get_while_running
from lib.utils import (Timer, print_tensor_shapes, get_num_iterations, print_train_step_data,
                       get_word_idx_mappings)


class Solver(object):
    """Solver for generic networks.
    """

    def __init__(self, net, graph, is_training):
        self.net = net
        self.graph = graph
        self.is_training = is_training
        self.num_epochs = cfg.TRAIN.NUM_EPOCHS
        self.train_timer = Timer()
        self.data_timer = Timer()

        self.global_step = slim.get_or_create_global_step()

        # Build basic ops and tensors
        if self.is_training:
            self.net.build_train_ops(self.global_step)

        if isinstance(net, Classifier):
            self.saver = tf.compat.v1.train.Saver(var_list=net.vars_to_restore, name='saver')
        else:
            self.saver = tf.compat.v1.train.Saver(max_to_keep=None, name='saver_all_var')  # Save all vars
        self.init_ops = self.build_init_ops()
        self.val_loss_ph = tf.compat.v1.placeholder(tf.float32, shape=(), name='val_loss_ph')
        self.net.build_summary_ops(self.graph)
        self.val_loss_summary = tf.compat.v1.summary.scalar(name='val_loss', tensor=self.val_loss_ph)

        print('saver variables:')
        print_tensor_shapes(net.vars_to_restore, prefix='-->')

    def build_init_ops(self):
        """Builds the init ops.

        Returns:
            init_op: Initialization op.
            ready_op: Initialization op.
            local_init_op: Initialization op.
        """
        with tf.compat.v1.name_scope('init_ops'):
            init_op = tf.compat.v1.global_variables_initializer()
            ready_op = tf.compat.v1.report_uninitialized_variables()
            local_init_op = tf.group(tf.compat.v1.local_variables_initializer(),
                                     tf.compat.v1.tables_initializer())
        return init_op, ready_op, local_init_op

    def restore_checkpoint(self, sess):
        """Restores the network to a previously saved checkpoint if a path is provided from the
        config.

        Args:
            sess: Current session.
        """
        if cfg.DIR.CKPT_PATH is not None:
            tf.compat.v1.logging.info('Restoring checkpoint.')
            self.saver.restore(sess, cfg.DIR.CKPT_PATH)
        else:
            tf.compat.v1.logging.info('Using network with random weights.')

    def train_step(self, sess, train_queue, step):
        """Executes a train step, including saving the summaries if appropriate.

        Args:
            sess: Current session.
            train_queue: Data queue containing train set minibatches.
            step: The current training iteration (0-based indexing).

        Returns:
            print_dict: Dictionary of items such as losses (for just one minibatch) to print.
        """
        print_dict = self.net.train_step(sess, train_queue, step, data_timer=self.data_timer)
        return print_dict

    def val_step(self, sess, val_queue):
        """Executes a validation step, which simply computes the loss.
        Args:
            sess: Current session.
            val_queue: Data queue containing validation set minibatches.
        Returns:
            val_loss: Loss for a single minibatch of validation data.
        """
        raise NotImplementedError('Must be implemented by a subclass.')

    def validate(self, sess, val_queue, step, num_val_iter):
        raise NotImplementedError('Must be implemented by a subclass.')

    def train(self, train_iters_per_epoch, train_queue, val_iters_per_epoch=None, val_queue=None):
        """Train the network, computing the validation loss if val_iters_per_epoch and val_queue are
        provided.

        Args:
            train_iters_per_epoch: Number of iterations in a single epoch of train data, as computed
                by the data process.
            train_queue: Data queue containing minibatches of train data.
            val_iters_per_epoch: Optional input describing the number of iterations in a single
                epoch of validation data, as computed by the data process.
            val_queue: Optional input representing the data queue containing minibatches of
                validation data.
        """
        if (val_iters_per_epoch is None and val_queue is not None) or \
                (val_iters_per_epoch is not None and val_queue is None):
            raise ValueError('Need to input both val size and val queue.')
        if val_iters_per_epoch is not None and val_queue is not None:
            run_validation = True
        else:
            run_validation = False

        print('-------------- BEGIN TRAINING --------------')
        num_train_iter = get_num_iterations(train_iters_per_epoch, num_epochs=cfg.TRAIN.NUM_EPOCHS,
                                            disp=True)
        num_val_iter = 20000 // cfg.CONST.BATCH_SIZE  # Evaluate on roughly 20000 samples
        if val_iters_per_epoch is not None:
            num_val_iter = min(num_val_iter, val_iters_per_epoch)

        with tf.compat.v1.Session() as sess:
            sess.run(self.init_ops)
            self.restore_checkpoint(sess)

            # Train loop
            for step in range(num_train_iter):
                # For randomized model
                # self.save(sess, step)
                # break

                self.train_timer.tic()
                print_dict = self.train_step(sess, train_queue, step)
                self.train_timer.toc()

                if (step + 1) % cfg.CONST.PRINT_FREQ == 0:
                    print_dict['queue size'] = (str(train_queue.qsize()) + '/'
                                                + str(cfg.CONST.QUEUE_CAPACITY))
                    print_dict['data fetch (sec/step)'] = '%.2f' % self.data_timer.average_time
                    print_dict['train step (sec/step)'] = '%.2f' % self.train_timer.average_time
                    print_train_step_data(print_dict, step)

                    # Reset timers
                    self.data_timer.reset()
                    self.train_timer.reset()

                if (run_validation is True) and ((step + 1) % cfg.TRAIN.VALIDATION_FREQ == 0):
                    validation_val = self.validate(sess, val_queue, step, num_val_iter)
                    if validation_val == -1:  # Training termination flag
                        tf.compat.v1.logging.info(
                                'Terminating train loop due to decreasing validation performance.')
                        break
                    else:
                        val_summary = sess.run(self.val_loss_summary,
                                               feed_dict={self.val_loss_ph: validation_val})
                        self.net.summary_writer.add_summary(val_summary, (step + 1))

                if (step + 1) % cfg.TRAIN.CKPT_FREQ == 0:
                    self.save(sess, step)

    def forward_pass_batches(self, sess, minibatch_generator):
        """Forward pass a series of minibatches from the minibatch generator.
        """
        minibatch_list = []
        outputs_list = []
        for step, minibatch in enumerate(minibatch_generator):
            np.random.seed(1234)
            try:
                outputs = self.net.forward_pass(sess, minibatch)
            except KeyError:
                outputs = self.net.forward_pass(sess, minibatch, full_feed_dict=True)
            # Reduce size of minibatch so we can pass through entire val set
            if isinstance(self.net, LBA):
                minibatch_save = {
                    'raw_embedding_batch': minibatch['raw_embedding_batch'],
                    'caption_label_batch': minibatch['caption_label_batch'],
                    'category_list': minibatch['category_list'],
                    'model_list': minibatch['model_list'],
                }
                minibatch = minibatch_save
            if isinstance(self.net, Classifier):
                minibatch_save = {
                    'class_label_batch': minibatch['class_label_batch'],
                    'model_id_list': minibatch['model_id_list'],
                }
                minibatch = minibatch_save
            minibatch_list.append(minibatch)
            outputs_list.append(outputs)

            if (step + 1) % 100 == 0:
                tf.compat.v1.logging.info('%s  Step: %d' % (str(datetime.now()), step + 1))

        return minibatch_list, outputs_list

    def val_phase_minibatch_generator(self, val_queue, num_val_iter):
        """Return a minibatch generator for the test phase.
        """
        for step in range(num_val_iter):
            minibatch = val_queue.get()
            minibatch['test_queue'] = True
            yield minibatch

    def evaluate(self, minibatch_list, outputs_list):
        """Do some evaluation of the outputs.
        """
        pass

    def test(self, test_process, test_queue, num_minibatches=None, save_outputs=False):
        """Compute (and optionally save) the outputs for the test set. This function only computes
        the outputs for num_minibatches minibatches.

        Args:
            test_process: Data process for the test data.
            test_queue: Queue containing minibatches of test data.
            num_minibatches: Number of minibatches to compute the outputs for.
            save_outputs: Boolean flag for whether or not to save the outputs.
        """
        with tf.compat.v1.Session() as sess:
            if cfg.DIR.CKPT_PATH is None:
                raise ValueError('Please provide a checkpoint.')
                sess.run(self.init_ops)
            else:
                self.restore_checkpoint(sess)

            def test_phase_minibatch_generator():
                for step, minibatch in enumerate(get_while_running(test_process, test_queue)):
                    if (num_minibatches is not None) and (step == num_minibatches):
                        break
                    yield minibatch

            minibatch_generator = test_phase_minibatch_generator()
            minibatch_list, outputs_list = self.forward_pass_batches(sess, minibatch_generator)
            self.evaluate(minibatch_list, outputs_list)

        if save_outputs:
            self.save_outputs(minibatch_list, outputs_list)

    def save(self, sess, step):
        """Save a checkpoint.
        """
        ckpt_path = os.path.join(cfg.DIR.LOG_PATH, 'model.ckpt')
        tf.compat.v1.logging.info('Saving checkpoint (step %d).', (step + 1))
        self.saver.save(sess, ckpt_path, global_step=(step + 1))

    def save_outputs(self, minibatch_list, outputs_list, filename=None):
        """Save the outputs (from the self.test).
        """
        raise NotImplementedError('Must be implemented by a subclass.')
