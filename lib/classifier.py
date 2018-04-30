from lib.config import cfg
from lib.network import Network
from lib.placeholders import build_shape_batch

import numpy as np
import tensorflow as tf


class Classifier(Network):
    """Shape classifier.
    """

    def __init__(self, inputs_dict, is_training, reuse=False, name='classifer'):
        assert inputs_dict['class_labels'] is not None
        self.class_labels = inputs_dict['class_labels']
        self.num_classes = len(self.class_labels)
        super(Classifier, self).__init__(inputs_dict, is_training, reuse=reuse, name=name)

    def build_placeholders(self):
        """Builds the placeholder for the shape.
        """
        num_channels = 4
        shape_batch_shape = [None] + [cfg.CONST.N_VOX] * 3 + [num_channels]
        shape_placeholder = build_shape_batch(shape_batch_shape, name='shape_placeholder')

        label_placeholder = tf.placeholder(tf.int32, [None], name='label_placeholder')
        self._placeholders = {
            'shape_batch': shape_placeholder,
            'label_batch': label_placeholder,
        }

    def preprocess_inputs(self):
        """No preprocessing required.
        """
        pass

    def build_loss(self):
        """Default loss: softmax cross entropy loss.

        Each label should be a single int32 label number/index rather than a 1-hot
        vector of float32 numbers.
        """
        labels = self.placeholders['label_batch']
        logits = self.outputs['logits']
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return {'loss': cross_entropy}

    def get_feed_dict(self, minibatch):
        """Get the feed dict for training.
        """
        batch_size = minibatch['voxel_tensor_batch'].shape[0]
        feed_dict = {
            self.placeholders['shape_batch']: minibatch['voxel_tensor_batch'],
            self.placeholders['label_batch']: minibatch['class_label_batch'],
        }
        return feed_dict, batch_size

    def forward_pass(self, sess, minibatch):
        """Computes a forward pass of the network for the given minibatch.

        Args:
            sess: Current session.
            minibatch: A minibatch of data.

        Returns:
            outputs: Outputs, as defined by the network.
        """
        feed_dict, batch_size = self.get_feed_dict(minibatch)
        eval_tensors = [
            self.outputs['logits'],
            self.losses['loss'],
            self.outputs['probabilities'],
            self.outputs['encoder_output']
        ]
        outputs = sess.run(eval_tensors, feed_dict=feed_dict)
        outputs_dict = {
            'class_logits': outputs[0],
            'loss': outputs[1],
            'prediction': np.argmax(outputs[0], axis=1),
            'probabilities': outputs[2],
            'encoder_outputs': outputs[3],
        }
        return outputs_dict

    @staticmethod
    def set_up_classification(data_dict):
        """Create a mapping from model ID/category to label index.
        """
        # Build list of (category, model_id) tuples
        assert cfg.CONST.DATASET == 'shapenet'
        if 'caption_tuples' in data_dict:
            caption_tuples = data_dict['caption_tuples']
        elif 'caption_embedding_tuples' in data_dict:
            caption_tuples = data_dict['caption_embedding_tuples']
        else:
            raise KeyError('inputs dict does not contain proper keys.')

        data_list = []
        for tup in caption_tuples:
            data_list.append((tup[1], tup[2]))
        data_list = list(set(data_list))  # data_list contains unique (category, model_id) tuples
        print('Number of models:', len(data_list))

        # Build class labels
        categories_list = sorted(list(set([tup[0] for tup in data_list])))
        category2label = {category: idx for idx, category in enumerate(categories_list)}
        print('Number of categories:', len(category2label))

        return data_list, category2label

    @property
    def placeholders(self):
        return self._placeholders
