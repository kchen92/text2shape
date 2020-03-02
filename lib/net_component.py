from lib.config import cfg
from lib.utils import print_tensor_shapes, compute_sequence_length, get_trainable_variables_by_scope

import tensorflow as tf
import tf_slim as slim


class NetComponent(object):
    """A network component.

    Simply an architecture without a loss or placeholders. Heavily modeled after the Network class.
    """

    def __init__(self, is_training, reuse=False, name='network_component', no_scope=False):
        self._name = name
        self._is_training = is_training
        self._reuse = reuse
        self._no_scope = no_scope
        self._outputs = None
        self._vars_to_restore = None

    def get_trainable_variables(self):
        """Return the trainable variables in the specified scope.

        Args:
            scope_name: Name of the scope as a string.

        Returns:
            trainable_vars: TensorFlow variables that are trainable and in the
                specified scope.
        """
        if self._no_scope is True:
            trainable_vars = self._trainable_vars
        else:
            trainable_vars = get_trainable_variables_by_scope(self.name)
        return trainable_vars

    def build_architecture(self, inputs_dict):
        """Build the architecture, not including placeholders and preprocessing.
        """
        raise NotImplementedError('Must be implemented by a subclass.')

    def build_model(self, inputs_dict):
        """Build the network component, without placeholders and loss.
        """
        print('building network:', self.name)

        def _build_model():
            self._outputs = self.build_architecture(inputs_dict)

            # Create summaries for network outputs
            if isinstance(self._outputs, dict):
                for k, v in self._outputs.items():
                    if isinstance(v, tf.Tensor):
                        tf.compat.v1.summary.histogram(k + '_hist_summary', v)

        if self._no_scope is True:
            _build_model()
            cur_variable_scope = tf.compat.v1.get_variable_scope()
            self._vars_to_restore = slim.get_variables_to_restore(include=[cur_variable_scope])
            self._trainable_vars = get_trainable_variables_by_scope(cur_variable_scope.name)
        else:
            with tf.compat.v1.variable_scope(self.name, reuse=self.reuse) as sc:
                _build_model()
                self._vars_to_restore = slim.get_variables_to_restore(include=[sc.name])

        print('--> done building {}'.format(self.name))

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
    def vars_to_restore(self):
        return self._vars_to_restore

    @property
    def reuse(self):
        return self._reuse


class TextEncoderNetComponent(NetComponent):

    def __init__(self, is_training, reuse=False, name='text_encoder_net_component'):
        super(TextEncoderNetComponent, self).__init__(is_training, reuse=reuse, name=name)

    def preprocess_inputs(self, inputs_dict):
        caption_batch, vocab_size = (inputs_dict['caption_batch'], inputs_dict['vocab_size'])
        input_batch = caption_batch

        print('--> building embedding layer')
        with tf.compat.v1.variable_scope('embedding_layer', reuse=self.reuse):
            embeddings = tf.Variable(
                tf.random.uniform([vocab_size, self.embedding_size], -1.0, 1.0),
                name='embedding_matrix')
            embedding_batch = tf.nn.embedding_lookup(params=embeddings,
                                                     ids=input_batch,
                                                     name='input_embedding_batch')

            # print shapes
            print_tensor_shapes([embedding_batch], prefix='----> ')

        seq_length = compute_sequence_length(input_batch)
        return {'embedding_batch': embedding_batch,
                'seq_length': seq_length}

    def build_model(self, inputs_dict):
        """Build the network component, without placeholders and loss.
        """
        print('building network:', self.name)

        with tf.compat.v1.variable_scope(self.name, reuse=self.reuse) as sc:
            processed_inputs = self.preprocess_inputs(inputs_dict)
            arch_inputs_dict = {**inputs_dict, **processed_inputs}  # Combine dicts
            self._outputs = self.build_architecture(arch_inputs_dict)

            # Create summaries for network outputs
            if isinstance(self._outputs, dict):
                for k, v in self._outputs.items():
                    if isinstance(v, tf.Tensor):
                        tf.compat.v1.summary.histogram(k + '_hist_summary', v)

            self._vars_to_restore = slim.get_variables_to_restore(include=[sc.name])

        print('--> done building {}'.format(self.name))

    @property
    def embedding_size(self):
        raise NotImplementedError('Must be implemented by a subclass.')


class LBANetComponent(NetComponent):

    def __init__(self, is_training, reuse=False, name='lba_net_component', no_scope=False):
        super(LBANetComponent, self).__init__(is_training, reuse=reuse, name=name,
                                              no_scope=no_scope)

    def build_architecture(self, inputs_dict):
        """Build the architecture, not including placeholders and preprocessing.

        Literally copy and pasted from lib.LBA (sorry!).
        """
        text_inputs_dict = {
            'caption_batch': inputs_dict['raw_embedding_batch'],
            'vocab_size': inputs_dict['vocab_size'],
        }
        self.text_encoder = self.text_encoder_class(self.is_training, self.reuse)
        self.text_encoder.build_model(text_inputs_dict)

        shape_inputs_dict = {'shape_batch': inputs_dict['shape_batch']}
        self.shape_encoder = self.shape_encoder_class(self.is_training, self.reuse)
        self.shape_encoder.build_model(shape_inputs_dict)

        # Change 'encoder_output' to be normalized
        if cfg.LBA.NORMALIZE is True:
            assert 'encoder_output' in self.text_encoder.outputs
            orig_text_encoder_output = self.text_encoder.outputs['encoder_output']
            self.text_encoder.outputs['encoder_output'] = tf.nn.l2_normalize(
                orig_text_encoder_output,
                axis=1,
                name='normalize_text_encoding',
            )

            assert 'encoder_output' in self.shape_encoder.outputs
            orig_shape_encoder_output = self.shape_encoder.outputs['encoder_output']
            self.shape_encoder.outputs['encoder_output'] = tf.nn.l2_normalize(
                orig_shape_encoder_output,
                axis=1,
                name='normalize_shape_encoding',
            )

        return {
            'text_encoder': self.text_encoder.outputs,
            'shape_encoder': self.shape_encoder.outputs,
        }

    @property
    def text_encoder_class(self):
        raise NotImplementedError('Must be implemented by a subclass.')

    @property
    def shape_encoder_class(self):
        raise NotImplementedError('Must be implemented by a subclass.')


class CriticNetComponent(NetComponent):

    def __init__(self, is_training, reuse=False, name='critic_net_component'):
        super(CriticNetComponent, self).__init__(is_training, reuse=reuse, name=name)
