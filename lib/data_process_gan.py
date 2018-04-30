import numpy as np
import pickle

from lib.config import cfg
from lib.data_process import DataProcess, print_error
from lib.preprocess import load_voxel
from lib.utils import augment_voxel_tensor, rescale_voxel_tensor, convert_embedding_list_to_batch


class GANDataProcessTestPhase(DataProcess):
    """Data process for GAN.
    """

    def __init__(self, data_queue, data_dict, repeat=False):
        """Initialize the Data Process.

        Args:
            data_queue:
            data_dict: A dict with keys 'caption_tuples' and 'caption_matches'. caption_tuples is
                a list of caption tuples, where each caption tuple is (caption, model_category,
                model_id). caption_matches is a dict where the key is any model ID and the value
                is a list of the indices (ints) of caption tuples that describe the same model ID.
        """
        if 'caption_tuples' in data_dict:
            self.caption_tuples = data_dict['caption_tuples']
        elif 'caption_embedding_tuples' in data_dict:
            self.caption_tuples = data_dict['caption_embedding_tuples']
        else:
            raise KeyError('inputs dict does not contain proper keys.')
        self.max_sentence_length = len(self.caption_tuples[0][0])
        self.class_labels = data_dict.get('class_labels')
        if cfg.DIR.PROBLEMATIC_NRRD_PATH is not None:
            with open(cfg.DIR.PROBLEMATIC_NRRD_PATH, 'rb') as f:
                self.bad_model_ids = pickle.load(f)
        else:
            self.bad_model_ids = None
        super(GANDataProcessTestPhase, self).__init__(data_queue, self.caption_tuples,
                                                          batch_size=cfg.CONST.BATCH_SIZE,
                                                          repeat=repeat)

    def is_bad_model_id(self, model_id):
        """Code reuse.
        """
        if self.bad_model_ids is not None:
            return model_id in self.bad_model_ids
        else:
            return False

    def get_learned_embedding(self, caption_tuple):
        return None

    def get_caption_data(self, db_ind):
        """Gets the caption data corresponding to the index specified by db_ind.

        NOTE: Copied directly from GANDataProcess.

        Args:
            db_ind: The integer index corresponding to the index of the caption in the dataset.

        Returns:
            cur_raw_embedding
            cur_category
            cur_model_id
            cur_voxel_tensor
        """
        while True:
            caption_tuple = self.caption_tuples[db_ind]
            cur_raw_embedding = caption_tuple[0]
            cur_category = caption_tuple[1]
            cur_model_id = caption_tuple[2]

            if self.is_bad_model_id(cur_model_id):
                db_ind = np.random.randint(self.num_data)  # Choose new caption
                continue

            try:
                cur_learned_embedding = self.get_learned_embedding(caption_tuple)
                cur_voxel_tensor = load_voxel(cur_category, cur_model_id)
                cur_voxel_tensor = augment_voxel_tensor(cur_voxel_tensor,
                                                        max_noise=cfg.TRAIN.AUGMENT_MAX)
                if self.class_labels is not None:
                    cur_class_label = self.class_labels[cur_category]
                else:
                    cur_class_label = None

            except FileNotFoundError:  # Retry if we don't have binvoxes
                db_ind = np.random.randint(self.num_data)
                continue
            break
        caption_data = {'raw_embedding': cur_raw_embedding,
                        'learned_embedding': cur_learned_embedding,
                        'category': cur_category,
                        'model_id': cur_model_id,
                        'voxel_tensor': cur_voxel_tensor,
                        'class_label': cur_class_label}
        return caption_data

    @print_error
    def run(self):
        # Run the loop until exit flag is set
        while not self.exit.is_set() and self.cur < self.num_data:
            # Ensure that the network sees (almost) all data per epoch
            db_inds = self.get_next_minibatch()

            raw_embedding_list = []
            learned_embedding_list = []
            category_list = []
            model_id_list = []
            voxel_tensor_list = []
            class_label_list = []

            # Build the batch
            for db_ind in db_inds:
                caption_data = self.get_caption_data(db_ind)
                cur_raw_embedding = caption_data['raw_embedding']
                cur_learned_embedding = caption_data['learned_embedding']
                cur_category = caption_data['category']
                cur_model_id = caption_data['model_id']
                cur_voxel_tensor = caption_data['voxel_tensor']
                cur_class_label = caption_data['class_label']

                raw_embedding_list.append(cur_raw_embedding)
                learned_embedding_list.append(cur_learned_embedding)
                model_id_list.append(cur_model_id)
                category_list.append(cur_category)
                voxel_tensor_list.append(cur_voxel_tensor)
                class_label_list.append(cur_class_label)

            raw_embedding_batch = np.array(raw_embedding_list).astype(np.int32)
            learned_embedding_batch = np.array(learned_embedding_list).astype(np.float32)
            voxel_tensor_batch = np.array(voxel_tensor_list).astype(np.float32)
            if self.class_labels is not None:
                class_label_batch = np.array(class_label_list).astype(np.int32)
            else:
                class_label_batch = None

            batch_data = {
                'raw_embedding_batch': raw_embedding_batch,
                'learned_embedding_batch': learned_embedding_batch,
                'voxel_tensor_batch': voxel_tensor_batch,
                'category_list': category_list,
                'model_id_list': model_id_list,
                'class_label_batch': class_label_batch,
            }

            # The following will wait until the queue frees
            self.data_queue.put(batch_data, block=True)

        print('Exiting enqueue process')


class GANDataProcess(GANDataProcessTestPhase):
    """Data process for GAN.

    Generates the following:
        - embedding batch (for g_match)
        - voxel batch + matching embedding batch (matching - for d_real_match)
        - voxel batch + mismatching embedding batch (mismatching - for d_real_mismatch)
    """

    def __init__(self, data_queue, data_dict, repeat):
        """Initialize the Data Process.

        Args:
            data_queue:
            data_dict: A dict with keys 'caption_tuples' and 'caption_matches'. caption_tuples is
                a list of caption tuples, where each caption tuple is (caption, model_category,
                model_id). caption_matches is a dict where the key is any model ID and the value
                is a list of the indices (ints) of caption tuples that describe the same model ID.
        """
        super(GANDataProcess, self).__init__(data_queue, data_dict, repeat=repeat)

    def shuffle_db_inds(self):
        """Instead of self.perm being a list of shuffled indices like in the standard DataProcess,
        self.perms is now a list of four permutation lists. In other words, self.perms[0] is what
        self.perm used to be. self.perms[0] is a list of shuffled indices.
        """
        # Randomly permute the training roidb
        if self.repeat:
            self.perms = [np.random.permutation(np.arange(self.num_data)) for _ in range(4)]
        else:
            # self.perms = [np.random.permutation(np.arange(self.num_data)) for _ in range(4)]
            print('----------------------------------------- not shuffling data')
            self.perms = [np.arange(self.num_data) for _ in range(4)]  # Dont shuffle
        self.cur = 0

    def get_next_minibatch(self):
        """Instead of db_inds being a list of indices, it is now a list of a list of indices.
        db_inds[0] is a list of indices for the given minibatch, as is db_inds[1], etc.
        """
        if (self.cur + self.batch_size) >= self.num_data and self.repeat:
            self.shuffle_db_inds()

        db_inds = [perm[self.cur:min(self.cur + self.batch_size, self.num_data)]
                   for perm in self.perms]
        self.cur += self.batch_size
        return db_inds

    def get_fake_match_batch(self, db_inds):
        """Constructs a batch of text embeddings according to db_inds.

        Args:
            db_inds: A list of indices (of the data samples) for the minibatch.

        Returns:
            embedding_batch: some-dimensional tensor.
        """
        raw_embedding_list = []
        learned_embedding_list = []
        voxel_tensor_list = []
        class_label_list = []
        category_list = []
        model_list = []

        for db_ind in db_inds:
            caption_data = self.get_caption_data(db_ind)
            raw_embedding_list.append(caption_data['raw_embedding'])
            learned_embedding_list.append(caption_data['learned_embedding'])
            voxel_tensor_list.append(caption_data['voxel_tensor'])
            class_label_list.append(caption_data['class_label'])
            category_list.append(caption_data['category'])
            model_list.append(caption_data['model_id'])
        raw_embedding_batch = np.array(raw_embedding_list).astype(np.int32)
        learned_embedding_batch = convert_embedding_list_to_batch(learned_embedding_list,
                                                                  np.float32)
        voxel_tensor_batch = np.array(voxel_tensor_list).astype(np.float32)
        if self.class_labels is not None:
            class_label_batch = np.array(class_label_list).astype(np.int32)
        else:
            class_label_batch = None
        return (
            raw_embedding_batch,
            learned_embedding_batch,
            voxel_tensor_batch,
            class_label_batch,
            category_list,
            model_list,
        )

    def get_real_match_batch(self, db_inds):
        """Constructs a batch of text encodings b1 and a corresponding batch of voxel tensors v1
        where b1[i] is the text embedding/caption corresponding to v1[i].

        Args:
            db_inds: A list of indices (of the data samples) for the minibatch.

        Returns:
            embedding_batch:
            voxel_tensor_batch:
        """
        raw_embedding_list = []
        learned_embedding_list = []
        voxel_tensor_list = []

        # Build the batch
        for db_ind in db_inds:
            caption_data = self.get_caption_data(db_ind)
            cur_raw_embedding = caption_data['raw_embedding']
            cur_learned_embedding = caption_data['learned_embedding']
            cur_voxel_tensor = caption_data['voxel_tensor']
            raw_embedding_list.append(cur_raw_embedding)
            learned_embedding_list.append(cur_learned_embedding)
            voxel_tensor_list.append(cur_voxel_tensor)
        raw_embedding_batch = np.array(raw_embedding_list).astype(np.int32)
        learned_embedding_batch = convert_embedding_list_to_batch(learned_embedding_list,
                                                                  np.float32)
        voxel_tensor_batch = np.array(voxel_tensor_list).astype(np.float32)

        return raw_embedding_batch, learned_embedding_batch, voxel_tensor_batch

    def get_real_mismatch_batch(self, db_inds):
        """Constructs a batch of text encodings b1 and a corresponding batch of voxel tensors v1
        where b1[i] is a text embedding/caption that does not correspond to v1[i].

        Args:
            db_inds: A list of indices (of the data samples) for the minibatch.
        """
        raw_embedding_list = []
        learned_embedding_list = []
        voxel_tensor_list = []

        # Build the batch
        for db_ind in db_inds:
            # Get a sample of data
            caption_data = self.get_caption_data(db_ind)
            # cur_raw_embedding = caption_data['raw_embedding']
            cur_category = caption_data['category']
            cur_model_id = caption_data['model_id']
            cur_voxel_tensor = caption_data['voxel_tensor']

            # Get a sample of data that doesn't match with the already selected sample
            while True:
                db_ind_mismatch = np.random.randint(self.num_data)
                caption_data = self.get_caption_data(db_ind_mismatch)
                cur_raw_embedding = caption_data['raw_embedding']
                cur_learned_embedding = caption_data['learned_embedding']
                cur_category_mismatch = caption_data['category']
                cur_model_id_mismatch = caption_data['model_id']
                # cur_voxel_tensor_mismatch = caption_data['voxel_tensor']

                if cfg.CONST.DATASET == 'shapenet':
                    if cur_model_id_mismatch == cur_model_id:  # We did not find a mismatching sample
                        continue  # Try a different data sample and hope it is a mismatching sample
                    break
                elif cfg.CONST.DATASET == 'primitives':
                    if cur_category_mismatch == cur_category:  # We did not find a mismatching sample
                        continue  # Try a different data sample and hope it is a mismatching sample
                    break
                else:
                    raise ValueError('Please use a supported dataset (shapenet, primitives).')

            raw_embedding_list.append(cur_raw_embedding)
            learned_embedding_list.append(cur_learned_embedding)
            voxel_tensor_list.append(cur_voxel_tensor)
        raw_embedding_batch = np.array(raw_embedding_list).astype(np.int32)
        learned_embedding_batch = convert_embedding_list_to_batch(learned_embedding_list,
                                                                  np.float32)
        voxel_tensor_batch = np.array(voxel_tensor_list).astype(np.float32)

        return raw_embedding_batch, learned_embedding_batch, voxel_tensor_batch

    @print_error
    def run(self):
        # Run the loop until exit flag is set
        while not self.exit.is_set() and self.cur < self.num_data:
            # Ensure that the network sees (almost) all data per epoch
            db_inds = self.get_next_minibatch()

            # fake / matching
            (raw_embedding_batch_fake_match, learned_embedding_batch_fake_match,
             voxel_tensor_batch_fake_match, class_label_batch_fake_match,
             category_list_fake_match, model_list_fake_match) = self.get_fake_match_batch(db_inds[0])

            # fake / mismatching
            pass

            # real / matching
            (raw_embedding_batch_real_match, learned_embedding_batch_real_match,
             voxel_tensor_batch_real_match) = self.get_real_match_batch(db_inds[2])

            # real / mismatching
            (raw_embedding_batch_real_mismatch, learned_embedding_batch_real_mismatch,
             voxel_tensor_batch_real_mismatch) = self.get_real_mismatch_batch(db_inds[3])

            batch_data = {
                'raw_embedding_batch_fake_match': raw_embedding_batch_fake_match,
                'learned_embedding_batch_fake_match': learned_embedding_batch_fake_match,
                'voxel_tensor_batch_fake_match': voxel_tensor_batch_fake_match,
                'class_label_batch_fake_match': class_label_batch_fake_match,
                'category_list_fake_match': category_list_fake_match,
                'model_list_fake_match': model_list_fake_match,
                'raw_embedding_batch_real_match': raw_embedding_batch_real_match,
                'learned_embedding_batch_real_match': learned_embedding_batch_real_match,
                'voxel_tensor_batch_real_match': voxel_tensor_batch_real_match,
                'raw_embedding_batch_real_mismatch': raw_embedding_batch_real_mismatch,
                'learned_embedding_batch_real_mismatch': learned_embedding_batch_real_mismatch,
                'voxel_tensor_batch_real_mismatch': voxel_tensor_batch_real_mismatch,
            }

            # The following will wait until the queue frees
            self.data_queue.put(batch_data, block=True)

        print('Exiting enqueue process')


class CWGANMetricEmbeddingDataProcess(GANDataProcess):
    """Data process conditional GAN. Uses embeddings generated from deep metric learning.
    """

    def __init__(self, data_queue, data_dict, repeat):
        """Initialize the Data Process.

        Args:
            data_queue:
            data_dict: A dict with keys 'caption_tuples' and 'caption_matches'. caption_tuples is
                a list of caption tuples, where each caption tuple is (caption, model_category,
                model_id). caption_matches is a dict where the key is any model ID and the value
                is a list of the indices (ints) of caption tuples that describe the same model ID.
        """
        super(CWGANMetricEmbeddingDataProcess, self).__init__(data_queue, data_dict, repeat=repeat)

    def get_learned_embedding(self, caption_tuple):
        cur_learned_embedding = caption_tuple[3]
        return cur_learned_embedding
