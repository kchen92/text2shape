from lib.config import cfg
from lib.layers import smoothed_metric_loss
from lib.network import Network, default_train_step
from lib.utils import open_pickle
import lib.placeholders as placeholders
import lib.losses as losses

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class LBA(Network):
    """Implementation of Learning by association.
    """

    def __init__(self, inputs_dict, is_training, reuse=False, name='lba_net'):
        step = slim.get_or_create_global_step()
        self.ema = tf.train.ExponentialMovingAverage(0.99, step)
        self.p_aba = None
        self.p_target = None

        self.category2label = self._build_category2label()
        super(LBA, self).__init__(inputs_dict, is_training, reuse=reuse, name=name)

    def build_placeholders(self):
        """Build the placeholders.
        """
        if cfg.CONST.DATASET == 'shapenet':
            n_captions = cfg.CONST.BATCH_SIZE * cfg.LBA.N_CAPTIONS_PER_MODEL
        elif cfg.CONST.DATASET == 'primitives':
            n_captions = cfg.CONST.BATCH_SIZE * cfg.LBA.N_CAPTIONS_PER_MODEL / cfg.LBA.N_PRIMITIVE_SHAPES_PER_CATEGORY
        else:
            raise ValueError('Please select a valid dataset.')

        # Build raw text description placeholders
        max_caption_length = self.inputs_dict['max_caption_length']
        raw_text_batch = placeholders.build_raw_text_embedding_batch(
            max_caption_length,
            'raw_text_batch',
            batch_size=n_captions,
        )

        # Build shape placeholders
        num_channels = 4
        shape_batch_shape = [
            cfg.CONST.BATCH_SIZE,
            cfg.CONST.N_VOX,
            cfg.CONST.N_VOX,
            cfg.CONST.N_VOX,
            num_channels,
        ]
        shape_batch = placeholders.build_shape_batch(shape_batch_shape, 'shape_batch')

        caption_labels = tf.placeholder(
            tf.int32,
            shape=[n_captions],
            name='caption_label_batch',
        )

        shape_labels = tf.placeholder(
            tf.int32,
            shape=[cfg.CONST.BATCH_SIZE],
            name='shape_label_batch',
        )

        self._placeholders = {
            'raw_embedding_batch': raw_text_batch,
            'shape_batch': shape_batch,
            'caption_label_batch': caption_labels,
            'shape_label_batch': shape_labels,
        }

    def preprocess_inputs(self):
        """Preprocess the inputs.
        """
        pass

    def build_architecture(self):
        """Build the architecture, not including placeholders and preprocessing.
        """
        text_inputs_dict = {
            'caption_batch': self.placeholders['raw_embedding_batch'],
            'vocab_size': self.inputs_dict['vocab_size'],
        }
        self.text_encoder = self.text_encoder_class(self.is_training, self.reuse)
        self.text_encoder.build_model(text_inputs_dict)
        shape_inputs_dict = {'shape_batch': self.placeholders['shape_batch']}
        self.shape_encoder = self.shape_encoder_class(self.is_training, self.reuse)
        self.shape_encoder.build_model(shape_inputs_dict)

        # Change 'encoder_output' to be normalized
        if cfg.LBA.NORMALIZE is True:
            assert 'encoder_output' in self.text_encoder.outputs
            orig_text_encoder_output = self.text_encoder.outputs['encoder_output']
            self.text_encoder.outputs['encoder_output'] = tf.nn.l2_normalize(
                orig_text_encoder_output,
                dim=1,
                name='normalize_text_encoding',
            )

            assert 'encoder_output' in self.shape_encoder.outputs
            orig_shape_encoder_output = self.shape_encoder.outputs['encoder_output']
            self.shape_encoder.outputs['encoder_output'] = tf.nn.l2_normalize(
                orig_shape_encoder_output,
                dim=1,
                name='normalize_shape_encoding',
            )

        return {
            'text_encoder': self.text_encoder.outputs,
            'shape_encoder': self.shape_encoder.outputs,
        }

    def train_step(self, sess, train_queue, step, data_timer=None):
        minibatch = self.get_minibatch(train_queue, data_timer=data_timer)
        feed_dict, batch_size = self.get_feed_dict(minibatch)
        cur_loss = default_train_step(sess, step, self.total_loss, self.train_ops['train_op'],
                                      feed_dict, self.summary_op, self.summary_writer)

        # Get losses to print out
        losses = [
            self.losses['tst_walker_loss'],
            self.losses['tst_visit_loss'],
            self.losses['sts_walker_loss'],
            self.losses['sts_visit_loss'],
            self.losses['classification_loss'],
        ]
        if cfg.LBA.COSINE_DIST is True:
            losses.append(self.losses['metric_tt'])
            losses.append(self.losses['metric_st'])

            if cfg.LBA.NORMALIZE is False:
                losses.append(self.losses['weighted_text_norm'])
                losses.append(self.losses['weighted_shape_norm'])
        loss_vals = sess.run(losses, feed_dict=feed_dict)

        out = {
            'tst_walker_loss': loss_vals[0],
            'tst_visit_loss': loss_vals[1],
            'sts_walker_loss': loss_vals[2],
            'sts_visit_loss': loss_vals[3],
            'classification_loss after weighting': loss_vals[4],
            'total_loss': cur_loss,
        }

        if cfg.LBA.COSINE_DIST is True:
            out['metric_tt'] = loss_vals[5]
            out['metric_st_and_ts'] = loss_vals[6]
            if cfg.LBA.NORMALIZE is False:
                out['weighted text norm'] = loss_vals[7]
                out['weighted shape norm'] = loss_vals[8]

        # Project matrix to make it PSD
        if (cfg.LBA.DIST_TYPE == 'mahalanobis') and (step % cfg.LBA.PROJECT_FREQ == 0):
            sess.run(self.psd_ops)
        return out

    def build_loss(self):
        if cfg.LBA.NO_LBA is False:
            lba_losses, self.p_aba, self.p_target, psd_op_1, psd_op_2 = losses.build_lba_loss(
                self.text_encoder,
                self.shape_encoder,
                self.placeholders['caption_label_batch']
            )
            if cfg.LBA.DIST_TYPE == 'mahalanobis':
                psd_ops = []
                if psd_op_1 is not None:
                    psd_ops.append(psd_op_1)
                if psd_op_2 is not None:
                    psd_ops.append(psd_op_2)
                assert psd_ops

                self.psd_ops = psd_ops

            if cfg.LBA.CLASSIFICATION is True:
                labels = self.placeholders['shape_label_batch']
                logits = self.outputs['shape_encoder']['logits']
                cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
                lba_losses['classification_loss'] = cfg.LBA.CLASSIFICATION_MULTIPLIER * cross_entropy
            else:
                lba_losses['classification_loss'] = tf.zeros([])
        else:
            lba_losses = {
                'tst_walker_loss': tf.zeros([]),
                'tst_visit_loss': tf.zeros([]),
                'sts_walker_loss': tf.zeros([]),
                'sts_visit_loss': tf.zeros([]),
                'classification_loss': tf.zeros([]),
            }

        if self.is_training and (cfg.LBA.COSINE_DIST is True):
            assert (cfg.LBA.NORMALIZE is True) or (cfg.LBA.INVERTED_LOSS is True)
            assert cfg.LBA.N_CAPTIONS_PER_MODEL == 2
            if cfg.LBA.INVERTED_LOSS is True:
                cur_margin = 1.
            else:
                cur_margin = 0.5

            text_embeddings = self.outputs['text_encoder']['encoder_output']
            shape_embeddings = self.outputs['shape_encoder']['encoder_output']
            indices = [i // 2 for i in range(cfg.CONST.BATCH_SIZE * cfg.LBA.N_CAPTIONS_PER_MODEL)]
            if cfg.CONST.DATASET == 'shapenet':
                shape_embeddings_rep = tf.gather(shape_embeddings, indices, axis=0)
            elif cfg.CONST.DATASET == 'primitives':
                assert cfg.LBA.N_PRIMITIVE_SHAPES_PER_CATEGORY == 2
                shape_embeddings_rep = shape_embeddings
            else:
                raise ValueError('Please select a valid datset.')

            embeddings = text_embeddings
            metric_tt = smoothed_metric_loss(embeddings, name='smoothed_metric_loss_tt', margin=cur_margin)

            if cfg.CONST.DATASET == 'shapenet':
                mask_ndarray = np.asarray([0., 1.] * cfg.CONST.BATCH_SIZE)[:, np.newaxis]
            elif cfg.CONST.DATASET == 'primitives':
                assert cfg.LBA.N_PRIMITIVE_SHAPES_PER_CATEGORY == 2
                assert cfg.CONST.BATCH_SIZE % cfg.LBA.N_PRIMITIVE_SHAPES_PER_CATEGORY == 0
                mask_ndarray = np.asarray([0., 1.] * (cfg.CONST.BATCH_SIZE
                                          // cfg.LBA.N_PRIMITIVE_SHAPES_PER_CATEGORY))[:, np.newaxis]
            else:
                raise ValueError('Please select a valid datset.')
            mask = tf.constant(mask_ndarray, dtype=tf.float32, name='mask_st')
            inverted_mask = 1. - mask
            embeddings = tf.multiply(text_embeddings, mask) + tf.multiply(shape_embeddings_rep, inverted_mask)
            metric_st = smoothed_metric_loss(embeddings, name='smoothed_metric_loss_st', margin=cur_margin)

            embeddings = tf.multiply(text_embeddings, inverted_mask) + tf.multiply(shape_embeddings_rep, mask)
            # metric_ts = smoothed_metric_loss(embeddings, name='smoothed_metric_loss_ts', margin=cur_margin)

            lba_losses['metric_tt'] = cfg.LBA.METRIC_MULTIPLIER * metric_tt
            lba_losses['metric_st'] = 2. * cfg.LBA.METRIC_MULTIPLIER * metric_st

            if cfg.LBA.NORMALIZE is False:  # Add a penalty on the embedding norms
                text_norms = tf.norm(text_embeddings, axis=1, name='text_norm')
                unweighted_txt_loss = tf.reduce_mean(tf.maximum(0., text_norms - cfg.LBA.MAX_NORM))
                shape_norms = tf.norm(shape_embeddings, axis=1, name='shape_norm')
                unweighted_shape_loss = tf.reduce_mean(tf.maximum(0., shape_norms - cfg.LBA.MAX_NORM))
                lba_losses['weighted_text_norm'] = cfg.LBA.TEXT_NORM_MULTIPLIER * unweighted_txt_loss
                lba_losses['weighted_shape_norm'] = cfg.LBA.SHAPE_NORM_MULTIPLIER * unweighted_shape_loss
        return lba_losses

    def categorylist2labellist(self, category_list, test_queue=False):
        """Convert a category list to a shape labels ndarray batch.
        """
        if cfg.CONST.DATASET == 'shapenet':
            shape_labels = [self.category2label[cat] for cat in category_list]
            if len(shape_labels) > cfg.CONST.BATCH_SIZE:  # TST, MM
                shape_label_batch = np.asarray(shape_labels[::cfg.LBA.N_CAPTIONS_PER_MODEL])
            else:  # STS mode, validation
                shape_label_batch = np.asarray(shape_labels)
            return shape_label_batch
        elif cfg.CONST.DATASET == 'primitives':
            shape_labels = [self.category2label[cat] for cat in category_list
                            for _ in range(cfg.LBA.N_PRIMITIVE_SHAPES_PER_CATEGORY)]
            if (cfg.LBA.MODEL_TYPE == 'TST') or (cfg.LBA.MODEL_TYPE == 'MM'):
                shape_label_batch = np.asarray(shape_labels[::cfg.LBA.N_CAPTIONS_PER_MODEL])
            elif (cfg.LBA.MODEL_TYPE == 'STS'):  # STS mode, validation
                if test_queue:
                    shape_label_batch = np.asarray(shape_labels)[::cfg.LBA.N_PRIMITIVE_SHAPES_PER_CATEGORY]
                else:
                    shape_label_batch = np.asarray(shape_labels)
            return shape_label_batch
        else:
            raise ValueError('Please select a valid dataset.')

    def get_feed_dict(self, minibatch):
        """Parse the minibatch data and return the feed dict.

        Args:
            minibatch: A dictionary of minibatch of data from the data process.

        Returns:
            feed_dict: A feed dict for both the generator and discriminator.
            batch_size: The size of the current minibatch.
        """
        shape_label_batch = self.categorylist2labellist(minibatch['category_list'],
                                                        minibatch.get('test_queue', False))
        feed_dict = {
            self.placeholders['raw_embedding_batch']: minibatch['raw_embedding_batch'],
            self.placeholders['shape_batch']: minibatch['voxel_tensor_batch'],
            self.placeholders['caption_label_batch']: minibatch['caption_label_batch'],
            self.placeholders['shape_label_batch']: shape_label_batch,
        }
        batch_size = cfg.CONST.BATCH_SIZE  # Number of models, not number of captions
        return feed_dict, batch_size

    def forward_pass(self, sess, minibatch):
        """Computes a forward pass of the network for the given minibatch.

        Args:
            sess: Current session.
            minibatch: A minibatch of data.

        Returns:
            outputs: Outputs, as defined by the network.
        """
        # Use proper batch size
        text_batch_size = minibatch['raw_embedding_batch'].shape[0]
        if cfg.CONST.DATASET == 'shapenet':
            if text_batch_size < cfg.CONST.BATCH_SIZE * cfg.LBA.N_CAPTIONS_PER_MODEL:
                orig_raw_embedding_batch = minibatch['raw_embedding_batch']
                orig_caption_label_batch = minibatch['caption_label_batch']
                minibatch['raw_embedding_batch'] = np.vstack((
                        minibatch['raw_embedding_batch'],
                        np.zeros((cfg.CONST.BATCH_SIZE * cfg.LBA.N_CAPTIONS_PER_MODEL - text_batch_size,
                                  minibatch['raw_embedding_batch'].shape[1]))
                ))
                minibatch['caption_label_batch'] = np.hstack((
                        minibatch['caption_label_batch'],
                        np.zeros((cfg.CONST.BATCH_SIZE * cfg.LBA.N_CAPTIONS_PER_MODEL - text_batch_size))
                ))

        feed_dict, batch_size = self.get_feed_dict(minibatch)
        eval_tensors = [
            self.outputs['text_encoder']['encoder_output'],
            self.outputs['shape_encoder']['encoder_output'],
            self.outputs['shape_encoder']['logits'],
        ]
        if cfg.LBA.NO_LBA is False:
            eval_tensors.append(self.p_aba)
            eval_tensors.append(self.p_target)
        outputs = sess.run(eval_tensors, feed_dict=feed_dict)

        if cfg.CONST.DATASET == 'shapenet':
            # Get only relevant outputs
            if text_batch_size < cfg.CONST.BATCH_SIZE * cfg.LBA.N_CAPTIONS_PER_MODEL:
                outputs[0] = outputs[0][:cfg.CONST.BATCH_SIZE]
                minibatch['raw_embedding_batch'] = orig_raw_embedding_batch
                minibatch['caption_label_batch'] = orig_caption_label_batch

        text_encoder_output = {
            'encoder_output': outputs[0]
        }
        shape_encoder_output = {
            'encoder_output': outputs[1]
        }
        logits = outputs[2]
        if cfg.LBA.NO_LBA is False:
            p_aba_val = outputs[3]
            p_target_val = outputs[4]
        else:
            p_aba_val = None
            p_target_val = None
        outputs_dict = {
            'text_encoder': text_encoder_output,
            'shape_encoder': shape_encoder_output,
            'p_aba': p_aba_val,
            'p_target': p_target_val,
            'logits': logits,
            'prediction': np.argmax(logits, axis=1),
        }
        return outputs_dict

    @property
    def placeholders(self):
        return self._placeholders

    @property
    def text_encoder_class(self):
        raise NotImplementedError('Must be implemented by a subclass.')

    @property
    def shape_encoder_class(self):
        raise NotImplementedError('Must be implemented by a subclass.')

    def _build_category2label(self):
        if cfg.CONST.DATASET == 'shapenet':
            category2label = {'03001627': 0, '04379243': 1}
        elif cfg.CONST.DATASET == 'primitives':
            train_inputs_dict = open_pickle(cfg.DIR.PRIMITIVES_TRAIN_DATA_PATH)
            val_inputs_dict = open_pickle(cfg.DIR.PRIMITIVES_VAL_DATA_PATH)
            test_inputs_dict = open_pickle(cfg.DIR.PRIMITIVES_TEST_DATA_PATH)
            f = lambda inputs_dict: list(inputs_dict['category_matches'].keys())
            categories = f(train_inputs_dict) + f(val_inputs_dict) + f(test_inputs_dict)
            categories = list(set(categories))
            category2label = {cat: idx for idx, cat in enumerate(categories)}
        else:
            raise ValueError('Please select a valid dataset.')
        return category2label
