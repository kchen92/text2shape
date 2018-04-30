"""Data processes for classifiers.
"""

import numpy as np

from lib.config import cfg
from lib.data_process import DataProcess
from lib.preprocess import load_voxel
from lib.utils import augment_voxel_tensor, rescale_voxel_tensor


class ShapeClassifierDataProcess(DataProcess):

    def __init__(self, data_queue, data_dict, batch_size=None, repeat=True):
        if batch_size is None:
            batch_size = cfg.CONST.BATCH_SIZE

        # Build list of (category, model_id) tuples
        assert cfg.CONST.DATASET == 'shapenet'
        if 'caption_tuples' in data_dict:
            self.caption_tuples = data_dict['caption_tuples']
        elif 'caption_embedding_tuples' in data_dict:
            self.caption_tuples = data_dict['caption_embedding_tuples']
        else:
            raise KeyError('inputs dict does not contain proper keys.')

        # data_paths is a list of unique (category, model_id) tuples
        data_paths = data_dict['category_model_list']
        self.class_labels = data_dict['class_labels']
        super(ShapeClassifierDataProcess, self).__init__(data_queue, data_paths,
                                                         batch_size=batch_size, repeat=repeat)

    def run(self):
        # Run the loop until exit flag is set
        while not self.exit.is_set() and self.cur < self.num_data:
            # Ensure that the network sees (almost) all data per epoch
            db_inds = self.get_next_minibatch()

            voxel_tensor_list = []
            class_label_list = []
            model_id_list = []

            # Build the batch
            for db_ind in db_inds:
                while True:
                    try:
                        cur_category, cur_model_id = self.data_paths[db_ind]
                        cur_voxel_tensor = load_voxel(cur_category, cur_model_id)
                        cur_voxel_tensor = augment_voxel_tensor(cur_voxel_tensor,
                                                                max_noise=cfg.TRAIN.AUGMENT_MAX)
                        cur_class_label = self.class_labels[cur_category]

                    except FileNotFoundError:  # Retry if we don't have binvoxes
                        # print('ERROR: Cannot find file with the following model ID:', cur_model_id)
                        db_ind = np.random.randint(self.num_data)
                        continue
                    break

                voxel_tensor_list.append(cur_voxel_tensor)
                class_label_list.append(cur_class_label)

                model_id_list.append(cur_model_id)
                # if cfg.CONST.DATASET == 'shapenet':
                #     model_id_list.append(cur_model_id)
                # elif cfg.CONST.DATASET == 'primitives':
                #     model_id_list.append(cur_category)

            voxel_tensor_batch = np.array(voxel_tensor_list).astype(np.float32)
            class_label_batch = np.array(class_label_list).astype(np.int32)

            batch_data = {
                'voxel_tensor_batch': voxel_tensor_batch,
                'class_label_batch': class_label_batch,
                'model_id_list': model_id_list,
            }

            # The following will wait until the queue frees
            self.data_queue.put(batch_data, block=True)

        print('Exiting enqueue process')
