import numpy as np
import os
import pickle
import tensorflow as tf

from lib.config import cfg
from lib.solver import Solver
from lib.utils import write_list_to_txt, write_sentences_txt, get_json_path


class ClassifierSolver(Solver):
    """Solver for generic classifier models.
    """

    def __init__(self, net, graph, is_training):
        super(ClassifierSolver, self).__init__(net, graph, is_training)
        self.val_acc = np.asarray([0.] * 5)
        self.val_ckpts = [None] * 5

    def validate(self, sess, val_queue, step, num_val_iter):
        tf.compat.v1.logging.info('Running validation.')

        minibatch_generator = self.val_phase_minibatch_generator(val_queue, num_val_iter)
        minibatch_list, outputs_list = self.forward_pass_batches(sess, minibatch_generator)

        correct = []
        for minibatch, outputs in zip(minibatch_list, outputs_list):
            correct.extend(np.equal(minibatch['class_label_batch'], outputs['prediction']).tolist())

        tf.compat.v1.logging.info('Evaluated {} samples.'.format(len(correct)))
        cur_val_acc = sum(correct) / len(correct)
        tf.compat.v1.logging.info('Accuracy: {}'.format(cur_val_acc))

        # Check if we should terminate training
        tf.compat.v1.logging.info('Previous validation accuracies:')
        tf.compat.v1.logging.info(self.val_acc)
        tf.compat.v1.logging.info('Current validation accuracy: {}'.format(cur_val_acc))
        if all(self.val_acc > cur_val_acc):
            tf.compat.v1.logging.info('Best checkpoint: {}'.format(self.val_ckpts[np.argmax(self.val_acc)]))
            tf.compat.v1.logging.info('Best accuracy: {}'.format(np.amax(self.val_acc)))
            return -1  # -1 is the termination flag
        else:  # Update val acc list
            self.val_acc = np.roll(self.val_acc, 1)
            self.val_acc[0] = cur_val_acc
            self.val_ckpts = np.roll(self.val_ckpts, 1)
            self.val_ckpts[0] = step + 1
            return cur_val_acc

    def save_outputs(self, minibatch_list, outputs_list, filename='end2end_gan_outputs.p'):
        """Save the outputs (from the self.test).

        Mostly copied from End2EndGANTestSolver.

        minibatch_list:
        outputs_list:
        filename:
        """
        data_list = []
        model_id_list = []
        gt_class_label_list = []
        output_class_label_list = []
        for minibatch, outputs in zip(minibatch_list, outputs_list):
            # Parse the minibatch
            cur_gt_class_label_list = minibatch['class_label_batch']
            cur_model_id_list = minibatch['model_id_list']

            # Store each data item into a list
            batch_size = len(cur_model_id_list)
            for i in range(batch_size):
                gt_class_label = cur_gt_class_label_list[i]
                encoder_output = outputs['encoder_outputs'][i]
                cur_model_id = cur_model_id_list[i]
                output_class_logits = outputs['class_logits']
                output_loss = outputs['loss']
                output_prediction = outputs['prediction'][0]  # Predicted class as an integer index  # NOTE: Should 0 be i instead?
                data_dict = {
                    'model_id': cur_model_id,
                    'gt_class_label': gt_class_label,
                    'output_class_logits': output_class_logits,
                    'output_loss': output_loss,
                    'output_class_label': output_prediction,
                    'encoder_outputs': encoder_output,
                }

                data_list.append(data_dict)
                gt_class_label_list.append(gt_class_label)
                output_class_label_list.append(output_prediction)
                model_id_list.append(cur_model_id)

        acc = np.mean(np.equal(np.asarray(gt_class_label_list, dtype=np.int32),
                               np.asarray(output_class_label_list, dtype=np.int32)))
        print('Prediction accuracy:', acc)

        # Save path should be a subdirectory in the cfg.DIR.LOG_PATH
        _, ext = os.path.splitext(cfg.DIR.CKPT_PATH)
        if ext == '':
            save_dir = cfg.DIR.LOG_PATH
        else:
            save_dir = os.path.join(cfg.DIR.LOG_PATH, ext[1:])
            os.makedirs(save_dir, exist_ok=False)
        tf.compat.v1.logging.info('Outputs will be saved to: %s' % save_dir)

        # Save class labels and predictions if applicable
        txt_output_path = os.path.join(save_dir, 'classes_gt.txt')
        write_list_to_txt(gt_class_label_list, txt_output_path, add_numbers=True)
        tf.compat.v1.logging.info('Saved outputs to: {}'.format(txt_output_path))

        txt_output_path = os.path.join(save_dir, 'classes_pred.txt')
        write_list_to_txt(output_class_label_list, txt_output_path, add_numbers=True)
        tf.compat.v1.logging.info('Saved outputs to: {}'.format(txt_output_path))

        # Save classifier "encoder outputs" to pickle file
        encoder_outputs_dict = {data_dict['model_id']: data_dict['encoder_outputs']
                                for data_dict in data_list}
        encoder_output_path = os.path.join(save_dir, 'encoder_outputs.p')
        with open(encoder_output_path, 'wb') as f:
            pickle.dump(encoder_outputs_dict, f)
        tf.compat.v1.logging.info('Saved outputs to: {}'.format(encoder_output_path))

        # Write all data (in the list) to a pickle file
        tf.compat.v1.logging.info('Saving all outputs in list to pickle file.')
        output_path = os.path.join(save_dir, filename)
        with open(output_path, 'wb') as f:
            pickle.dump(data_list, f)
        tf.compat.v1.logging.info('Saved outputs to: {}'.format(output_path))
