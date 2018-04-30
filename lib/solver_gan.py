import numpy as np
import os
import pickle
import tensorflow as tf

from collections import Counter
from datetime import datetime

from lib.classifier import Classifier
from lib.config import cfg
from lib.solver import Solver
from lib.utils import (Timer, undo_rescale_voxel_tensor,
                       sample_z, write_list_to_txt, write_sentences_txt,
                       get_json_path, AttributeWriter)


class End2EndGANDebugSolver(Solver):
    """Solver for GAN.
    """

    def __init__(self, net, graph, is_training):
        super(End2EndGANDebugSolver, self).__init__(net, graph, is_training)

    def evaluate(self, minibatch_list, outputs_list):
        pass

    def save_outputs(self, minibatch_list, outputs_list, filename='end2end_gan_outputs'):
        """Save the outputs (from the self.test).

        minibatch_list:
        outputs_list:
        filename:
        """
        data_fake_match_list = []
        data_real_match_list = []
        data_real_mismatch_list = []
        gt_voxel_tensor_fake_match_list = []
        gt_voxel_tensor_real_match_list = []
        gt_voxel_tensor_real_mismatch_list = []
        output_voxel_tensor_list = []
        for minibatch, outputs in zip(minibatch_list, outputs_list):
            # Parse the minibatch
            raw_embedding_batch_fake_match = minibatch['raw_embedding_batch_fake_match']
            voxel_tensor_batch_fake_match = minibatch['voxel_tensor_batch_fake_match']
            raw_embedding_batch_real_match = minibatch['raw_embedding_batch_real_match']
            voxel_tensor_batch_real_match = minibatch['voxel_tensor_batch_real_match']
            raw_embedding_batch_real_mismatch = minibatch['raw_embedding_batch_real_mismatch']
            voxel_tensor_batch_real_mismatch = minibatch['voxel_tensor_batch_real_mismatch']
            category_list_fake_match = minibatch['category_list_fake_match']
            model_list_fake_match = minibatch['model_list_fake_match']

            # Store each data item into a list
            batch_size = raw_embedding_batch_real_match.shape[0]
            for i in range(batch_size):
                output_voxel_tensor = outputs['t2s_generator_output'][i]
                output_critic_fake_match_score = outputs['t2s_critic_output'][i]
                output_critic_real_match_score = outputs['t2s_critic_real_match_output'][i]
                output_critic_real_mismatch_score = outputs['t2s_critic_real_mismatch_output'][i]

                raw_embedding_tensor_fake_match = raw_embedding_batch_fake_match[i]
                gt_voxel_tensor_fake_match = voxel_tensor_batch_fake_match[i]
                raw_embedding_tensor_real_match = raw_embedding_batch_real_match[i]
                gt_voxel_tensor_real_match = voxel_tensor_batch_real_match[i]
                raw_embedding_tensor_real_mismatch = raw_embedding_batch_real_mismatch[i]
                gt_voxel_tensor_real_mismatch = voxel_tensor_batch_real_mismatch[i]
                category_fake_match = category_list_fake_match[i]
                model_id_fake_match = model_list_fake_match[i]

                data_dict_fake_match = {
                    'raw_caption_embedding': raw_embedding_tensor_fake_match,
                    'gt_voxel_tensor': gt_voxel_tensor_fake_match,
                    'category': category_fake_match,
                    'model_id': model_id_fake_match,
                    'output_voxel_tensor': output_voxel_tensor,
                    'output_critic_score': output_critic_fake_match_score
                }
                data_dict_real_match = {
                    'raw_caption_embedding': raw_embedding_tensor_real_match,
                    'gt_voxel_tensor': gt_voxel_tensor_real_match,
                    'output_critic_score': output_critic_real_match_score
                }
                data_dict_real_mismatch = {
                    'raw_caption_embedding': raw_embedding_tensor_real_mismatch,
                    'gt_voxel_tensor': gt_voxel_tensor_real_mismatch,
                    'output_critic_score': output_critic_real_mismatch_score
                }

                data_fake_match_list.append(data_dict_fake_match)
                data_real_match_list.append(data_dict_real_match)
                data_real_mismatch_list.append(data_dict_real_mismatch)
                gt_voxel_tensor_fake_match_list.append(gt_voxel_tensor_fake_match)
                gt_voxel_tensor_real_match_list.append(gt_voxel_tensor_real_match)
                gt_voxel_tensor_real_mismatch_list.append(gt_voxel_tensor_real_mismatch)
                output_voxel_tensor_list.append(output_voxel_tensor)

        # Save path should be a subdirectory in the cfg.DIR.LOG_PATH
        _, ext = os.path.splitext(cfg.DIR.CKPT_PATH)
        if ext == '':
            save_dir = cfg.DIR.LOG_PATH
        else:
            save_dir = os.path.join(cfg.DIR.LOG_PATH, ext[1:])
            os.makedirs(save_dir, exist_ok=False)
        tf.logging.info('Outputs will be saved to: %s' % save_dir)

        # Print categories to a txt file
        tf.logging.info('Saving categories to a txt file.')
        txt_output_path = os.path.join(save_dir, 'categories_fake_match.txt')
        categories_fake_match_list = [cur_dict['category'] for cur_dict in data_fake_match_list]
        write_list_to_txt(categories_fake_match_list, txt_output_path, add_numbers=True)
        tf.logging.info('Saved categories to: {}'.format(txt_output_path))

        # Write ground truth model IDs to a txt file
        tf.logging.info('Saving all ground truth model IDs to a txt file.')
        txt_output_path = os.path.join(save_dir, 'model_ids_fake_match.txt')
        model_id_fake_match_list = [cur_dict['model_id'] for cur_dict in data_fake_match_list]
        write_list_to_txt(model_id_fake_match_list, txt_output_path, add_numbers=True)
        tf.logging.info('Saved model IDs to: {}'.format(txt_output_path))

        # # Convert raw text embeddings to sentences and save to txt file
        txt_output_path = os.path.join(save_dir, 'sentences_fake_match.txt')
        json_path = get_json_path()
        write_sentences_txt(json_path, data_fake_match_list, txt_output_path)

        # Convert raw text embeddings to sentences and save to txt file
        txt_output_path = os.path.join(save_dir, 'sentences_real_match.txt')
        json_path = get_json_path()
        write_sentences_txt(json_path, data_real_match_list, txt_output_path)

        # Convert raw text embeddings to sentences and save to txt file
        txt_output_path = os.path.join(save_dir, 'sentences_real_mismatch.txt')
        json_path = get_json_path()
        write_sentences_txt(json_path, data_real_mismatch_list, txt_output_path)

        # Write critic scores to a txt file
        tf.logging.info('Saving critic scores to a txt file.')
        txt_output_path = os.path.join(save_dir, 'critic_scores_fake_match.txt')
        critic_scores_list = [cur_dict['output_critic_score'][0]
                              for cur_dict in data_fake_match_list]
        write_list_to_txt(critic_scores_list, txt_output_path, add_numbers=True)
        tf.logging.info('Saved outputs to: {}'.format(txt_output_path))

        # Write critic scores to a txt file
        tf.logging.info('Saving critic scores to a txt file.')
        txt_output_path = os.path.join(save_dir, 'critic_scores_real_match.txt')
        critic_scores_list = [cur_dict['output_critic_score'][0]
                              for cur_dict in data_real_match_list]
        write_list_to_txt(critic_scores_list, txt_output_path, add_numbers=True)
        tf.logging.info('Saved outputs to: {}'.format(txt_output_path))

        # Write critic scores to a txt file
        tf.logging.info('Saving critic scores to a txt file.')
        txt_output_path = os.path.join(save_dir, 'critic_scores_real_mismatch.txt')
        critic_scores_list = [cur_dict['output_critic_score'][0]
                              for cur_dict in data_real_mismatch_list]
        write_list_to_txt(critic_scores_list, txt_output_path, add_numbers=True)
        tf.logging.info('Saved outputs to: {}'.format(txt_output_path))

        # Write all data (in the list) to a pickle file
        tf.logging.info('Saving all outputs in list to pickle file.')
        output_path = os.path.join(save_dir, filename + '_fake_match.p')
        with open(output_path, 'wb') as f:
            pickle.dump(data_fake_match_list, f)
        tf.logging.info('Saved outputs to: {}'.format(output_path))

        # Write all data (in the list) to a pickle file
        tf.logging.info('Saving all outputs in list to pickle file.')
        output_path = os.path.join(save_dir, filename + '_real_match.p')
        with open(output_path, 'wb') as f:
            pickle.dump(data_real_match_list, f)
        tf.logging.info('Saved outputs to: {}'.format(output_path))

        # Write all data (in the list) to a pickle file
        tf.logging.info('Saving all outputs in list to pickle file.')
        output_path = os.path.join(save_dir, filename + '_real_mismatch.p')
        with open(output_path, 'wb') as f:
            pickle.dump(data_real_mismatch_list, f)
        tf.logging.info('Saved outputs to: {}'.format(output_path))

        # Save all voxel tensors to npy files
        num_outputs = len(gt_voxel_tensor_real_match_list)
        output_voxel_tensor_filenames_list = []
        for i, (output_voxel_tensor, gt_voxel_tensor_fake_match, gt_voxel_tensor_real_match,
                gt_voxel_tensor_real_mismatch) in enumerate(
                zip(output_voxel_tensor_list, gt_voxel_tensor_fake_match_list,
                    gt_voxel_tensor_real_match_list, gt_voxel_tensor_real_mismatch_list)):

            # Outputs
            output_voxel_tensor_filename = os.path.join(save_dir,
                                                        '%04d_' % i + 'voxel_tensor_output')
            np.save(output_voxel_tensor_filename, output_voxel_tensor)
            output_voxel_tensor_filenames_list.append(output_voxel_tensor_filename)

            # Ground truth
            # gt_voxel_tensor_filename = os.path.join(save_dir, '%04d_'
            #                                         % i + 'voxel_tensor_fake_match')
            # np.save(gt_voxel_tensor_filename, gt_voxel_tensor_fake_match)

            # gt_voxel_tensor_filename = os.path.join(save_dir, '%04d_'
            #                                         % i + 'voxel_tensor_real_match')
            # np.save(gt_voxel_tensor_filename, gt_voxel_tensor_real_match)

            # gt_voxel_tensor_filename = os.path.join(save_dir, '%04d_'
            #                                         % i + 'voxel_tensor_real_mismatch')
            # np.save(gt_voxel_tensor_filename, gt_voxel_tensor_real_mismatch)

            if (i + 1) % 20 == 0:
                tf.logging.info('Saved %d out of %d voxel tensors.' % (i + 1, num_outputs))
        tf.logging.info('Done! Voxel tensors saved to: {}'.format(save_dir))
        tf.logging.info('To render, run: python -m tools.generate_nrrd {}'.format(save_dir))
