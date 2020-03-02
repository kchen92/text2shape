from lib.config import cfg
from lib.network import Network
from lib.utils import (
    print_tensor_shapes,
    compute_sequence_length,
    sample_z,
    get_learned_embedding_shape
)
from lib.optimizers import get_optimizer
import lib.placeholders as placeholders

import numpy as np
import tensorflow as tf
import tf_slim as slim


class CWGAN(Network):
    """Conditional Wasserstein text2shape GAN.
    """

    def __init__(self, inputs_dict, is_training, reuse=False, name='cw_gan'):
        self._placeholders = None

        # Fake
        self.text_encoder_fake = None
        self.t2s_generator = None
        self.fake_critic = None

        # Match
        self.text_encoder_mat = None
        self.mat_critic = None

        # Mismatch
        self.text_encoder_mis = None
        self.mis_critic = None

        # Shapes

        super(CWGAN, self).__init__(inputs_dict, is_training, reuse=reuse, name=name)

    def build_desc_component_placeholders(self):
        desc_component_shape = get_learned_embedding_shape()  # does not include batch size

        # Build text embedding placeholders
        desc_component_fake_match = placeholders.build_desc_component_batch(
            desc_component_shape,
            'desc_component_fake_match')
        desc_component_real_match = placeholders.build_desc_component_batch(
            desc_component_shape,
            'desc_component_real_match')
        desc_component_real_mismatch = placeholders.build_desc_component_batch(
            desc_component_shape,
            'desc_component_real_mismatch')
        desc_component_placeholders = {
            'desc_component_fake_match': desc_component_fake_match,
            'desc_component_real_match': desc_component_real_match,
            'desc_component_real_mismatch': desc_component_real_mismatch,
        }
        return desc_component_placeholders

    def build_raw_text_embedding_placeholders(self):
        raw_embedding_placeholders = {
            'raw_embedding_batch_fake_match': None,
            'raw_embedding_batch_real_match': None,
            'raw_embedding_batch_real_mismatch': None,
        }
        return raw_embedding_placeholders

    def build_placeholders(self):
        # Get description component placeholders
        desc_component_placeholders = self.build_desc_component_placeholders()
        desc_component_fake_match = desc_component_placeholders['desc_component_fake_match']
        desc_component_real_match = desc_component_placeholders['desc_component_real_match']
        desc_component_real_mismatch = desc_component_placeholders['desc_component_real_mismatch']

        # Get raw text embedding placeholders
        raw_embedding_placeholders = self.build_raw_text_embedding_placeholders()
        raw_embedding_fake_match = (
            raw_embedding_placeholders['raw_embedding_batch_fake_match'])
        raw_embedding_real_match = (
            raw_embedding_placeholders['raw_embedding_batch_real_match'])
        raw_embedding_real_mismatch = (
            raw_embedding_placeholders['raw_embedding_batch_real_mismatch'])

        # Build shape placeholders
        num_gen_output_channels = 4
        shape_batch_shape = [cfg.CONST.BATCH_SIZE,
                             cfg.CONST.N_VOX,
                             cfg.CONST.N_VOX,
                             cfg.CONST.N_VOX,
                             num_gen_output_channels]
        mat_shape_batch = placeholders.build_shape_batch(shape_batch_shape, 'match_voxel_tensor')
        mis_shape_batch = placeholders.build_shape_batch(shape_batch_shape, 'mismatch_voxel_tensor')

        # build noise placeholder
        noise_batch = placeholders.build_noise_batch(cfg.GAN.NOISE_SIZE, 'noise_component')

        # Build selector placeholder
        if cfg.CONST.IMPROVED_WGAN is True:
            selector_placeholder = tf.compat.v1.placeholder(tf.float32, [], name='real_fake_selector')
        else:
            selector_placeholder = None

        self._placeholders = {'fake_desc_component': desc_component_fake_match,
                              'mat_desc_component': desc_component_real_match,
                              'mat_shape_batch': mat_shape_batch,
                              'mis_desc_component': desc_component_real_mismatch,
                              'mis_shape_batch': mis_shape_batch,
                              'selector_placeholder': selector_placeholder,
                              'raw_embedding_batch_fake_match': raw_embedding_fake_match,
                              'raw_embedding_batch_real_match': raw_embedding_real_match,
                              'raw_embedding_batch_real_mismatch': raw_embedding_real_mismatch,
                              'noise_batch': noise_batch}

    def preprocess_inputs(self):
        pass

    def get_g_trainable_vars(self):
        return self.t2s_generator.get_trainable_variables()

    def build_train_ops(self, global_step):
        # Text encoder / generator train op
        self.g_global_step = tf.compat.v1.get_variable('g_global_step', shape=[], dtype=tf.int64,
                                             initializer=tf.compat.v1.zeros_initializer(), trainable=False)
        tf.compat.v1.summary.scalar('g_global_step', self.g_global_step)
        g_learning_rate = tf.compat.v1.train.exponential_decay(cfg.TRAIN.LEARNING_RATE,
                                                     self.g_global_step,
                                                     cfg.TRAIN.DECAY_STEPS,
                                                     cfg.TRAIN.DECAY_RATE,
                                                     staircase=cfg.TRAIN.STAIRCASE,
                                                     name='g_lr_decay')
        tf.compat.v1.summary.scalar('g_learning_rate', g_learning_rate)

        g_trainable_vars = self.get_g_trainable_vars()
        print('trainable vars for generator step')
        print_tensor_shapes(g_trainable_vars, prefix='-->')

        g_optimizer = get_optimizer(cfg.TRAIN.OPTIMIZER, g_learning_rate, name='g_optimizer')
        g_train_op = slim.learning.create_train_op(
            self.losses['generator_loss'],
            g_optimizer,
            global_step=self.g_global_step,
            variables_to_train=g_trainable_vars)

        # Create discriminator global step
        self.d_global_step = tf.compat.v1.get_variable('d_global_step', shape=[], dtype=tf.int64,
                                             initializer=tf.compat.v1.zeros_initializer(), trainable=False)
        tf.compat.v1.summary.scalar('d_global_step', self.d_global_step)

        # Critic train op
        d_initial_learning_rate = cfg.TRAIN.LEARNING_RATE * cfg.GAN.D_LEARNING_RATE_MULTIPLIER
        d_learning_rate = tf.compat.v1.train.exponential_decay(d_initial_learning_rate,
                                                     self.d_global_step,
                                                     cfg.TRAIN.DECAY_STEPS,
                                                     cfg.TRAIN.DECAY_RATE,
                                                     staircase=cfg.TRAIN.STAIRCASE,
                                                     name='d_lr_decay')
        tf.compat.v1.summary.scalar('d_learning_rate', d_learning_rate)

        d_trainable_vars = self.fake_critic.get_trainable_variables()
        print('trainable vars in {}'.format(self.fake_critic.name))
        print_tensor_shapes(d_trainable_vars, prefix='-->')

        d_optimizer = get_optimizer(cfg.GAN.D_OPTIMIZER, d_learning_rate, name='d_optimizer')
        d_train_op = slim.learning.create_train_op(
            self.losses['critic_loss'],
            d_optimizer,
            global_step=self.d_global_step,
            variables_to_train=d_trainable_vars)

        if cfg.CONST.IMPROVED_WGAN is False:
            # Get clip variables
            d_trainable_vars = self.fake_critic.get_trainable_variables()
            # d_trainable_vars_without_batch_norm = [v for v in d_trainable_vars
            #                                        if 'batch_norm' not in v.name]
            clip_vars = d_trainable_vars

            # Print clip variables
            print('variables to clip:')
            for v in clip_vars:
                print(v.name)

            clip_op = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in clip_vars]
        else:
            clip_op = None

        self.train_ops = {'generator': g_train_op,
                          'discriminator': d_train_op,
                          'clip_op': clip_op}

    def build_loss(self):
        d_fake_match_logits = self.fake_critic.outputs['logits']
        if self.is_training:
            d_real_match_logits = self.mat_critic.outputs['logits']
            d_real_mismatch_logits = self.mis_critic.outputs['logits']

            # Critic losses
            d_loss_fake_match = tf.multiply(tf.reduce_mean(input_tensor=d_fake_match_logits),
                                            float(cfg.WGAN.FAKE_MATCH_LOSS_COEFF),
                                            name='critic_fake_loss')
            d_loss_real_match = tf.multiply(tf.reduce_mean(input_tensor=tf.negative(d_real_match_logits)),
                                            float(cfg.WGAN.MATCH_LOSS_COEFF),
                                            name='critic_match_loss')
            d_loss_real_mismatch = tf.multiply(tf.reduce_mean(input_tensor=d_real_mismatch_logits),
                                               float(cfg.WGAN.FAKE_MISMATCH_LOSS_COEFF),
                                               name='critic_mismatch_loss')

            loss_gp_name = 'd_loss_gp'
            if cfg.CONST.IMPROVED_WGAN is True:

                gradients_dtext, gradients_dshape = tf.gradients(
                    ys=self.gp_critic.outputs['logits'],
                    xs=[self.gp_text_data,
                     self.gp_shape_data])
                gradients_dshape_reshaped = tf.reshape(gradients_dshape, [cfg.CONST.BATCH_SIZE, -1])

                slopes_text = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(gradients_dtext),
                                                    axis=[1]))
                slopes_shape = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(gradients_dshape_reshaped),
                                                     axis=[1]))

                gp_text = tf.reduce_mean(input_tensor=(slopes_text - 1.)**2, name='critic_gp_text')
                gp_shape = tf.reduce_mean(input_tensor=(slopes_shape - 1.)**2, name='critic_gp_shape')

                gradient_penalty = tf.add(gp_text, gp_shape, name='gradient_penalty_add')
                d_loss_gp = tf.multiply(float(cfg.WGAN.GP_COEFF), gradient_penalty,
                                        name=loss_gp_name)
            else:
                d_loss_gp = tf.zeros([], dtype=tf.float32, name=loss_gp_name)

            critic_loss = tf.add_n([d_loss_fake_match,
                                    d_loss_real_match,
                                    d_loss_real_mismatch,
                                    d_loss_gp],
                                   name='critic_loss')

            # Text encoder / generator loss
            g_loss = tf.reduce_mean(input_tensor=tf.negative(d_fake_match_logits), name='generator_fake_match')

            return {'critic_fake': d_loss_fake_match,
                    'critic_match': d_loss_real_match,
                    'critic_mismatch': d_loss_real_mismatch,
                    'critic_gp': d_loss_gp,
                    'critic_loss': critic_loss,
                    'generator_loss': g_loss}
        else:
            return None

    def build_gan(self, gan_inputs_dict):
        """Build generator pipeline.

        - Noise concatenation
        - Text to shape generator
        - Critic (fake)
        - Critic (mat)
        - Critic (mis)
        """
        # Concatenate description component with noise component
        fake_text_embedding = tf.concat(values=(gan_inputs_dict['desc_component_fake'],
                                                gan_inputs_dict['noise_batch']),
                                        axis=1, name='text_encoding_with_noise')

        # Text to shape generator
        num_gen_output_channels = 4
        t2s_generator_inputs_dict = {'text_encoding_with_noise': fake_text_embedding,
                                     'num_output_channels': num_gen_output_channels}
        t2s_generator = self.t2s_generator_class(self.is_training, reuse=self.reuse)
        t2s_generator.build_model(t2s_generator_inputs_dict)

        # Build fake critic
        fake_critic = self.t2s_critic_class(self.is_training, reuse=self.reuse)
        fake_critic_inputs_dict = {
            'text_encoding_without_noise': gan_inputs_dict['desc_component_fake'],
            'shape_batch': t2s_generator.outputs['sigmoid_output']}
        fake_critic.build_model(fake_critic_inputs_dict)

        # Build match critic
        mat_critic = self.t2s_critic_class(self.is_training, reuse=True)
        mat_critic_inputs_dict = {
            'text_encoding_without_noise': gan_inputs_dict['desc_component_mat'],
            'shape_batch': gan_inputs_dict['mat_shape_batch']}
        mat_critic.build_model(mat_critic_inputs_dict)

        # Build mismatch critic
        mis_critic = self.t2s_critic_class(self.is_training, reuse=True)
        mis_critic_inputs_dict = {
            'text_encoding_without_noise': gan_inputs_dict['desc_component_mis'],
            'shape_batch': gan_inputs_dict['mis_shape_batch']}
        mis_critic.build_model(mis_critic_inputs_dict)

        if cfg.CONST.IMPROVED_WGAN is True:
            # Get real data and fake data
            selector_placeholder = gan_inputs_dict['selector_placeholder']
            fake_shape_data = t2s_generator.outputs['sigmoid_output']
            real_shape_data = gan_inputs_dict['mat_shape_batch']
            fake_text_data = gan_inputs_dict['desc_component_fake']
            real_text_data = gan_inputs_dict['desc_component_mat']

            gp_shape_data = (selector_placeholder * fake_shape_data
                             + (1. - selector_placeholder) * real_shape_data)
            gp_text_data = (selector_placeholder * fake_text_data
                            + (1. - selector_placeholder) * real_text_data)

            # Build gradients critic
            gp_critic = self.t2s_critic_class(self.is_training, reuse=True)
            gp_critic_inputs_dict = {
                'text_encoding_without_noise': gp_text_data,
                'shape_batch': gp_shape_data}
            gp_critic.build_model(gp_critic_inputs_dict)
        else:
            gp_critic = None
            gp_shape_data = None
            gp_text_data = None

        self.text_embedding_with_noise = fake_text_embedding
        self.t2s_generator = t2s_generator
        self.fake_critic = fake_critic
        self.mat_critic = mat_critic
        self.mis_critic = mis_critic
        self.gp_critic = gp_critic
        self.gp_shape_data = gp_shape_data
        self.gp_text_data = gp_text_data

        outputs = {'t2s_generator_sigmoid_output': self.t2s_generator.outputs['sigmoid_output'],
                   'fake_critic_output': self.fake_critic.outputs['logits']}
        return outputs

    def build_gan_inputs_dict(self):
        gan_inputs_dict = {
            'noise_batch': self.placeholders['noise_batch'],
            'desc_component_fake': self.placeholders['fake_desc_component'],
            'desc_component_mat': self.placeholders['mat_desc_component'],
            'desc_component_mis': self.placeholders['mis_desc_component'],
            'mat_shape_batch': self.placeholders['mat_shape_batch'],
            'mis_shape_batch': self.placeholders['mis_shape_batch'],
            'selector_placeholder': self.placeholders['selector_placeholder'],
        }
        return gan_inputs_dict

    def build_model(self):
        """Builds the baseline text-to-shape model.
        """
        print('building network:', self.name)

        with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
            self.build_placeholders()
            self.preprocess_inputs()
            gan_inputs_dict = self.build_gan_inputs_dict()
            self._outputs = self.build_gan(gan_inputs_dict)
            self._losses = self.build_loss()
            self._total_loss = None

            self._vars_to_restore = (self.t2s_generator.vars_to_restore
                                     + self.fake_critic.vars_to_restore)

        print('--> done building {}!'.format(self.name))

    def get_feed_dict(self, minibatch):
        """Gets the feed dict. Use the learned embeddings rather than the raw embeddings.

        Args:
            minibatch: Minibatch of data.

        Returns:
            feed_dict: Feed dictionary.
            batch_size: Batch size.
        """
        # Parse minibatch data
        learned_embedding_batch_fake_match = minibatch['learned_embedding_batch_fake_match']
        learned_embedding_batch_real_match = minibatch['learned_embedding_batch_real_match']
        voxel_tensor_batch_real_match = minibatch['voxel_tensor_batch_real_match']
        voxel_tensor_batch_real_mismatch = minibatch['voxel_tensor_batch_real_mismatch']
        learned_embedding_batch_real_mismatch = minibatch['learned_embedding_batch_real_mismatch']

        # Set up disciminator and generator feed dicts
        batch_size = learned_embedding_batch_fake_match.shape[0]
        g_match_noise_batch = sample_z(batch_size)
        if cfg.GAN.INTERP is True:
            assert self.is_training
            interp = (0.5 * learned_embedding_batch_real_match[:2, :]
                      + 0.5 * learned_embedding_batch_real_mismatch[:2, :])
            learned_embedding_batch_fake_match[-2:, :] = interp

        feed_dict = {
            self.placeholders['fake_desc_component']: learned_embedding_batch_fake_match,
            self.placeholders['noise_batch']: g_match_noise_batch,

            self.placeholders['mat_desc_component']: learned_embedding_batch_real_match,
            self.placeholders['mat_shape_batch']: voxel_tensor_batch_real_match,

            self.placeholders['mis_desc_component']: learned_embedding_batch_real_mismatch,
            self.placeholders['mis_shape_batch']: voxel_tensor_batch_real_mismatch
        }

        if cfg.CONST.IMPROVED_WGAN is True:
            if self.is_training is True:
                feed_dict[self.placeholders['selector_placeholder']] = float(np.random.randint(2))
            elif self.is_training is False:
                feed_dict[self.placeholders['selector_placeholder']] = 1.
            else:
                raise ValueError

        return feed_dict, batch_size

    def discriminator_step(self, sess, d_feed_dict, step):
        """Perform a discriminator train step. Runs the standard GAN update but adds an additional
        weight clipping step to the discriminator/critic.

        Args:
            sess: Current session.
            d_feed_dict: Feed dict for the discriminator (train step).
            step: Step in train loop (same as generator global step).
        """
        # Take a gradient step
        d_train_op = self.train_ops['discriminator']
        eval_tensors = [
            d_train_op,
            self.losses['critic_loss'],
            self.losses['critic_fake'],
            self.losses['critic_match'],
            self.losses['critic_mismatch'],
            self.losses['critic_gp'],
        ]
        _, d_loss, d_fake, d_match, d_mismatch, d_gp = sess.run(eval_tensors, feed_dict=d_feed_dict)

        if cfg.CONST.IMPROVED_WGAN is False:
            # Clip discriminator/critic weights
            sess.run(self.train_ops['clip_op'])

        return d_loss, d_fake, d_match, d_mismatch

    def generator_step(self, sess, g_feed_dict, step):
        """Perform a generator train step.

        Args:
            sess: Current session
            g_feed_dict: Feed dict for the generator (train step).
            step: The current training iteration (0-based indexing), which is the same as generator
                global step.
        """
        # Compute the summary at every step, even if we don't save it
        g_train_op = self.train_ops['generator']
        if self.summary_op is not None:
            eval_ops = [self.losses['generator_loss'], self.losses['critic_loss'], g_train_op,
                        self.summary_op]
            g_loss, d_loss, _, summary = sess.run(eval_ops, feed_dict=g_feed_dict)
        else:
            eval_ops = [self.losses['generator_loss'], self.losses['critic_loss'], g_train_op]
            g_loss, d_loss, _ = sess.run(eval_ops, feed_dict=g_feed_dict)

        # Write summary
        if self.summary_op is not None and ((step + 1) % cfg.TRAIN.SUMMARY_FREQ == 0):
            self.summary_writer.add_summary(summary, (step + 1))

        return g_loss

    def train_step(self, sess, train_queue, step, data_timer=None):
        """WGAN train step. At each time step, we update the critic multiple times, while updating
        the generator only once. We also clip the weights in the critic
        (see self.discriminator_step).

        Args:
            sess: Current session.
            train_queue: Data queue containing train set minibatches.
            step: The current training iteration (0-based indexing).
        """
        # Compute number of critic train iterations for the current train step
        if (cfg.CONST.IMPROVED_WGAN is False) and ((step < cfg.WGAN.INTENSE_TRAINING_STEPS)
                                                   or (step % cfg.WGAN.INTENSE_TRAINING_FREQ == 0)):
            tf.compat.v1.logging.info(
                'Training critic for {} steps.'.format(cfg.WGAN.INTENSE_TRAINING_INTENSITY))
            num_critic_steps = cfg.WGAN.INTENSE_TRAINING_INTENSITY
        else:
            num_critic_steps = cfg.WGAN.NUM_CRITIC_STEPS

        # Train critic more than generator
        for _ in range(num_critic_steps):
            minibatch = self.get_minibatch(train_queue, data_timer=data_timer)
            feed_dict, _ = self.get_feed_dict(minibatch)
            d_loss, d_fake, d_match, d_mismatch = self.discriminator_step(sess, feed_dict, step)

        # Update the generator once every time step
        minibatch = self.get_minibatch(train_queue, data_timer=data_timer)
        feed_dict, _ = self.get_feed_dict(minibatch)
        g_loss = self.generator_step(sess, feed_dict, step)

        return {'generator loss': g_loss,
                'critic loss': d_loss,
                'critic fake match loss': d_fake,
                'critic real match loss': d_match,
                'critic real mismatch loss': d_mismatch}

    def get_test_feed_dict(self, minibatch):
        """Gets the feed dict.

        Args:
            minibatch: Minibatch of data.

        Returns:
            feed_dict: Feed dictionary.
            batch_size: Batch size.
        """
        # Parse minibatch data
        learned_embedding_batch = minibatch['learned_embedding_batch']

        # Set up disciminator and generator feed dicts
        batch_size = learned_embedding_batch.shape[0]
        g_match_noise_batch = sample_z(batch_size)
        feed_dict = {
            self.placeholders['fake_desc_component']: learned_embedding_batch,
            self.placeholders['noise_batch']: g_match_noise_batch,
        }

        return feed_dict, batch_size

    def forward_pass(self, sess, minibatch, full_feed_dict=False):
        """Computes a forward pass of the network for the given minibatch.

        Args:
            sess: Current session.
            minibatch: A minibatch of data.

        Returns:
            outputs: Outputs, as defined by the network.
        """
        if full_feed_dict is True:
            feed_dict, batch_size = self.get_feed_dict(minibatch)
            eval_tensors = [self.t2s_generator.outputs['sigmoid_output'],
                            self.fake_critic.outputs['logits'],
                            self.mat_critic.outputs['logits'],
                            self.mis_critic.outputs['logits']]
            (generator_output, critic_fake_match_output, critic_real_match_output,
             critic_real_mismatch_output) = sess.run(eval_tensors, feed_dict=feed_dict)
            outputs = {'t2s_generator_output': generator_output,
                       't2s_critic_output': critic_fake_match_output,
                       't2s_critic_real_match_output': critic_real_match_output,
                       't2s_critic_real_mismatch_output': critic_real_mismatch_output}
        else:
            feed_dict, batch_size = self.get_test_feed_dict(minibatch)
            eval_tensors = [self.t2s_generator.outputs['sigmoid_output'],
                            self.fake_critic.outputs['logits']]
            generator_output, critic_output = sess.run(eval_tensors,
                                                       feed_dict=feed_dict)
            outputs = {'t2s_generator_output': generator_output,
                       't2s_critic_output': critic_output}
        return outputs

    @property
    def placeholders(self):
        return self._placeholders

    @property
    def t2s_generator_class(self):
        raise NotImplementedError('Must be implemented by a subclass.')

    @property
    def t2s_critic_class(self):
        raise NotImplementedError('Must be implemented by a subclass.')
