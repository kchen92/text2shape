import os
import pickle
import numpy as np
import tensorflow as tf

from lib.config import cfg
from lib.solver import Solver
import tools.eval.eval_text_encoder as eval_text_encoder


class TextEncoderSolver(Solver):
    """Solver for the text encoder model.
    """

    def __init__(self, net, graph, is_training):
        self.val_acc = np.asarray([0.] * 5)
        self.val_ckpts = [None] * 5
        super(TextEncoderSolver, self).__init__(net, graph, is_training)

        # Add a summary for validation performance
        self.val_perf_placeholder = tf.placeholder(np.float32, shape=[], name='val_perf')
        self.val_perf_summary = tf.summary.scalar('val_perf', self.val_perf_placeholder)

    def compute_metrics(self, embeddings_dict):
        tf.logging.info('Using L2 distance.')
        metric = 'minkowski'
        try:
            return eval_text_encoder.compute_metrics(cfg.CONST.DATASET, embeddings_dict,
                                                     metric=metric, concise=True)
        except ValueError as e:
            print('Caught ValueError! Skipping evaluation.')

    def get_captions_tensor(self, minibatch):
        """Return the raw embedding captions tensor.
        """
        return minibatch['caption_batch']

    def get_category(self, category_list, i):
        return category_list[i]

    def get_model_id(self, model_id_list, i):
        return model_id_list[i]

    def get_caption_embedding(self, outputs, i=None):
        """Get the caption embedding at index i from the outputs dict.
        """
        if i is not None:
            return outputs['encoder_output'][i]
        else:
            return outputs['encoder_output']

    def get_shape_embedding(self, outputs, i=None):
        """Get the shape embedding at index i from the outputs dict. This is None for the generic
        text encoder (which doesn't learn shape embeddings).
        """
        return None

    def consolidate_caption_tuples(self, minibatch_list, outputs_list, embedding_type='text'):
        """Form a list of tuples which each have the form:
        (caption, category, model_id, caption_embedding)
        """
        caption_tuples = []
        seen_text = []
        seen_shapes = []
        for minibatch, outputs in zip(minibatch_list, outputs_list):
            captions_tensor = self.get_captions_tensor(minibatch)
            category_list = minibatch['category_list']
            model_list = minibatch['model_list']
            for i in range(captions_tensor.shape[0]):
                if embedding_type == 'shape':
                    caption = None
                else:
                    caption = captions_tensor[i]
                category = self.get_category(category_list, i)
                model_id = self.get_model_id(model_list, i)
                if embedding_type == 'text':
                    caption_embedding_as_tuple = tuple(caption.tolist())
                    if not cfg.CONST.TEST_ALL_TUPLES and (caption_embedding_as_tuple in seen_text):
                        continue
                    else:
                        caption_embedding = self.get_caption_embedding(outputs, i=i)
                        seen_text.append(caption_embedding_as_tuple)
                elif embedding_type == 'shape':
                    if not cfg.CONST.TEST_ALL_TUPLES and (model_id in seen_shapes):
                        continue
                    else:
                        caption_embedding = self.get_shape_embedding(outputs, i=i)
                        seen_shapes.append(model_id)
                else:
                    return ValueError('Please use a valid embedding type (text or shape).')
                caption_tuple = (caption, category, model_id, caption_embedding)

                caption_tuples.append(caption_tuple)
        return caption_tuples

    def get_outputs_dict(self, sess, val_queue, num_val_iter):
        minibatch_generator = self.val_phase_minibatch_generator(val_queue, num_val_iter)
        minibatch_list, outputs_list = self.forward_pass_batches(sess, minibatch_generator)

        caption_tuples = self.consolidate_caption_tuples(minibatch_list, outputs_list)
        outputs_dict = {'caption_embedding_tuples': caption_tuples,
                        'dataset_size': len(caption_tuples)}
        return outputs_dict

    def validate(self, sess, val_queue, step, num_val_iter):
        """Executes a validation step, which simply computes the loss.

        Args:
            sess: Current session.
            val_queue: Data queue containing validation set minibatches.

        Returns:
            val_loss: Loss for a single minibatch of validation data.
        """
        tf.logging.info('Running validation.')

        outputs_dict = self.get_outputs_dict(sess, val_queue, num_val_iter)

        pr_at_k = self.compute_metrics(outputs_dict)
        assert len(pr_at_k) == 4
        precision, recall, recall_rate, ndcg = pr_at_k

        # Check if we should terminate training
        cur_val_acc = precision[4]  # Precision @ 5

        # Add validation summary
        val_perf_summary = sess.run(self.val_perf_summary,
                                    feed_dict={self.val_perf_placeholder: cur_val_acc})
        self.net.summary_writer.add_summary(val_perf_summary, (step + 1))

        print('Previous validation accuracies:')
        print(self.val_acc)
        print('Current validation accuracy:', cur_val_acc)
        if all(self.val_acc > cur_val_acc):
            print('Best checkpoint:', self.val_ckpts[np.argmax(self.val_acc)])
            return -1  # -1 is the termination flag
        else:  # Update val acc list
            self.val_acc = np.roll(self.val_acc, 1)
            self.val_acc[0] = cur_val_acc
            self.val_ckpts = np.roll(self.val_ckpts, 1)
            self.val_ckpts[0] = step + 1
            return cur_val_acc

    def save_outputs(self, minibatch_list, outputs_list, filename='text_embeddings.p'):
        """Saves the caption embeddings after they have processed through the text encoder.
        The caption embeddings are written to a pickle file that saves the outputs_dict.

        outputs_dict['caption_embedding_tuples']: A list of tuples. Each tuple contains the
            following (caption, category, model_id, caption_embedding), where caption is the list
            of word indices (from the minibatch input), and caption embedding is the embedding.
        outputs_dict['dataset_size']: Size of the dataset (length of the list).

        Args:
            minibatch_list:
            outputs_list:
            filename:
        """
        caption_tuples = self.consolidate_caption_tuples(minibatch_list, outputs_list)
        class_labels = (self.net.inputs_dict['class_labels']
                        if 'class_labels' in self.net.inputs_dict else None)
        outputs_dict = {'caption_embedding_tuples': caption_tuples,
                        'dataset_size': len(caption_tuples),
                        'class_labels': class_labels}

        tf.logging.info('Saving outputs.')
        output_path = os.path.join(cfg.DIR.LOG_PATH, filename)
        with open(output_path, 'wb') as f:
            pickle.dump(outputs_dict, f)
        tf.logging.info('Saved outputs to: {}'.format(output_path))

        # Print results
        if not cfg.CONST.TEST_ALL_TUPLES:
            self.compute_metrics(outputs_dict)

        return outputs_dict


class TextEncoderCosDistSolver(TextEncoderSolver):

    def __init__(self, net, graph, is_training):
        super(TextEncoderCosDistSolver, self).__init__(net, graph, is_training)

    def compute_metrics(self, embeddings_dict):
        tf.logging.info('Using cosine distance.')
        metric = 'cosine'
        try:
            return eval_text_encoder.compute_metrics(cfg.CONST.DATASET, embeddings_dict,
                                                     metric=metric, concise=True)
        except ValueError as e:
            print('Caught ValueError! Skipping evaluation.')


class LBASolver(TextEncoderCosDistSolver):
    """Solver for the LBA models.
    """

    def __init__(self, net, graph, is_training):
        super(LBASolver, self).__init__(net, graph, is_training)

    def get_captions_tensor(self, minibatch):
        """Return the raw embedding captions tensor.

        This method was created so that we can derive subclasses for TextEncoderSolver that are
        applicable specifically to LBA models.
        """
        return minibatch['raw_embedding_batch']

    def get_category(self, category_list, i):
        if cfg.LBA.MODEL_TYPE == 'STS':
            i = int(np.floor(i / 2))
        return category_list[i]

    def get_model_id(self, model_id_list, i):
        if cfg.LBA.MODEL_TYPE == 'STS':
            i = int(np.floor(i / 2))
        return model_id_list[i]

    def get_caption_embedding(self, outputs, i=None):
        """Get the caption embedding at index i from the outputs dict.

        This method was created so that we can derive subclasses for TextEncoderSolver that are
        applicable specifically to LBA models.
        """
        if i is not None:
            return outputs['text_encoder']['encoder_output'][i]
        else:
            return outputs['text_encoder']['encoder_output']

    def get_shape_embedding(self, outputs, i=None):
        """Get the shape embedding at index i from the outputs dict. This is None for the generic
        text encoder (which doesn't learn shape embeddings).
        """
        if i is not None:
            return outputs['shape_encoder']['encoder_output'][i]
        else:
            return outputs['shape_encoder']['encoder_output']

    def save_outputs(self, minibatch_list, outputs_list, filename='text_embeddings.p'):
        if not ((cfg.CONST.DATASET == 'primitives') and (cfg.LBA.TEST_MODE == 'shape')):
            print('-------------------- SAVING TEXT EMBEDDINGS ---------------------------')
            text_dict = super(LBASolver, self).save_outputs(
                    minibatch_list, outputs_list, filename=filename)

        if not ((cfg.CONST.DATASET == 'primitives') and (cfg.LBA.TEST_MODE == 'text')):
            # Save shape embeddings
            print('-------------------- SAVING SHAPE EMBEDDINGS --------------------------')
            filename = 'shape_embeddings.p'
            caption_tuples = self.consolidate_caption_tuples(minibatch_list, outputs_list,
                                                             embedding_type='shape')
            class_labels = (self.net.inputs_dict['class_labels']
                            if 'class_labels' in self.net.inputs_dict else None)
            outputs_dict = {'caption_embedding_tuples': caption_tuples,
                            'dataset_size': len(caption_tuples),
                            'class_labels': class_labels}

            tf.logging.info('Saving outputs.')
            output_path = os.path.join(cfg.DIR.LOG_PATH, filename)
            with open(output_path, 'wb') as f:
                pickle.dump(outputs_dict, f)
            tf.logging.info('Saved outputs to: {}'.format(output_path))

            # Print results
            if not cfg.CONST.TEST_ALL_TUPLES:
                self.compute_metrics(outputs_dict)

        if not cfg.CONST.DATASET == 'primitives':
            # Combine text embeddings and shape embeddings
            print('------------ Combined text and shape embeddings --------------')
            combined_dict = {
                'caption_embedding_tuples': text_dict['caption_embedding_tuples'] + outputs_dict['caption_embedding_tuples'],
                'dataset_size': text_dict['dataset_size'] + outputs_dict['dataset_size'],
            }
            tf.logging.info('Saving outputs.')
            output_path = os.path.join(cfg.DIR.LOG_PATH, 'text_and_shape_embeddings.p')
            with open(output_path, 'wb') as f:
                pickle.dump(combined_dict, f)
            tf.logging.info('Saved outputs to: {}'.format(output_path))

            if not cfg.CONST.TEST_ALL_TUPLES:
                self.compute_metrics(combined_dict)

    def val_phase_text_minibatch_generator(self):
        """Return a minibatch generator for the val/test phase for TEXT only.
        """
        # Modify self.caption_tuples so it does not contain multiple instances of the same caption
        new_tuples = []
        seen_captions = []
        for cur_tup in self._val_inputs_dict['caption_tuples']:
            cur_caption = tuple(cur_tup[0].tolist())
            if cur_caption not in seen_captions:
                seen_captions.append(cur_caption)
                new_tuples.append(cur_tup)
        caption_tuples = new_tuples

        # Collect all captions in the validation set
        raw_caption_list = [tup[0] for tup in caption_tuples]
        category_list = [tup[1] for tup in caption_tuples]
        model_list = [tup[2] for tup in caption_tuples]
        caption_list = raw_caption_list

        vx_tensor_shape = [cfg.CONST.N_VOX, cfg.CONST.N_VOX, cfg.CONST.N_VOX, 4]
        zeros_tensor = np.zeros([cfg.CONST.BATCH_SIZE] + vx_tensor_shape)
        caption_label_batch = np.asarray(list(range(cfg.CONST.BATCH_SIZE)))
        n_captions = len(caption_list)
        n_loop_captions = n_captions - (n_captions % cfg.CONST.BATCH_SIZE)
        tf.logging.info('Number of captions: {}'.format(n_captions))
        tf.logging.info(
                'Number of captions to loop through for validation: {}'.format(n_loop_captions))
        tf.logging.info(
                'Number of batches to loop through for validation: {}'.format(n_loop_captions / cfg.CONST.BATCH_SIZE))
        for start in range(0, n_loop_captions, cfg.CONST.BATCH_SIZE):
            captions = caption_list[start:(start + cfg.CONST.BATCH_SIZE)]
            minibatch = {
                'raw_embedding_batch': np.asarray(captions),
                'voxel_tensor_batch': zeros_tensor,
                'caption_label_batch': caption_label_batch,
                'category_list': category_list[start:(start + cfg.CONST.BATCH_SIZE)],
                'model_list': model_list[start:(start + cfg.CONST.BATCH_SIZE)],
                'test_queue': True,
            }
            yield minibatch

    def get_classification_accuracy(self, minibatch_list, outputs_list):
        tf.logging.info('Running validation.')

        correct = []
        for minibatch, outputs in zip(minibatch_list, outputs_list):
            shape_label_batch = self.net.categorylist2labellist(minibatch['category_list'])
            correct.extend(np.equal(shape_label_batch, outputs['prediction']).tolist())

        tf.logging.info('Evaluated {} samples.'.format(len(correct)))
        cur_val_acc = sum(correct) / len(correct)
        tf.logging.info('Classification accuracy: {}'.format(cur_val_acc))
        return cur_val_acc

    def get_outputs_dict(self, sess, val_queue, num_val_iter):
        # Shape encodings
        tf.logging.info('--> Computing shape encodings.')
        minibatch_generator = self.val_phase_minibatch_generator(val_queue, num_val_iter)
        shape_minibatch_list, shape_outputs_list = self.forward_pass_batches(sess,
                                                                             minibatch_generator)
        shape_caption_tuples = self.consolidate_caption_tuples(shape_minibatch_list,
                                                               shape_outputs_list,
                                                               embedding_type='shape')

        if cfg.LBA.CLASSIFICATION is True:
            classification_acc = self.get_classification_accuracy(shape_minibatch_list,
                                                                  shape_outputs_list)

        # Text encodings
        tf.logging.info('--> Computing text encodings.')
        minibatch_generator = self.val_phase_text_minibatch_generator()
        text_minibatch_list, text_outputs_list = self.forward_pass_batches(sess,
                                                                           minibatch_generator)
        text_caption_tuples = self.consolidate_caption_tuples(text_minibatch_list,
                                                              text_outputs_list,
                                                              embedding_type='text')

        all_caption_tuples = shape_caption_tuples + text_caption_tuples

        # Logging
        tf.logging.info('Number of computed shape encodings for validation: {}'.format(
                len(shape_caption_tuples)))
        tf.logging.info('Number of computed text encodings for validation: {}'.format(
                len(text_caption_tuples)))
        tf.logging.info('Total number of computed encodings for validation: {}'.format(
                len(all_caption_tuples)))

        outputs_dict = {'caption_embedding_tuples': all_caption_tuples,
                        'dataset_size': len(all_caption_tuples)}
        return outputs_dict

    def train(self, train_iters_per_epoch, train_queue, val_iters_per_epoch=None, val_queue=None,
              val_inputs_dict=None):
        if val_inputs_dict is not None:
            self._val_inputs_dict = val_inputs_dict
        super(TextEncoderSolver, self).train(train_iters_per_epoch, train_queue,
                                             val_iters_per_epoch=val_iters_per_epoch,
                                             val_queue=val_queue)
