import numpy as np
import pickle
import random
from collections import Counter

from lib.config import cfg
from lib.data_process import DataProcess, print_error, kill_processes
from lib.preprocess import load_voxel
from lib.utils import get_unique_el_mapping, get_attr_from_cat


class CaptionDataProcess(DataProcess):

    def __init__(self, data_queue, data_dict, repeat):
        """Initialize the Data Process. In this Data Process, we add matching
        caption pairs to each batch. For example,
        batch = [caption_1_1, caption_1_2, caption_2_1, caption_2_2, ..., caption_n_1, caption_n_2]
        where caption_i_d is the dth caption for the ith model.

        Args:
            data_queue:
            data_dict: A dict with keys 'caption_tuples' and 'caption_matches'. caption_tuples is
                a list of caption tuples, where each caption tuple is (caption, model_category,
                model_id). caption_matches is a dict where the key is any model ID and the value
                is a list of the indices (ints) of caption tuples that describe the same model ID.
        """
        self.caption_tuples = data_dict['caption_tuples']
        super(CaptionDataProcess, self).__init__(data_queue, self.caption_tuples,
                                                 batch_size=cfg.CONST.BATCH_SIZE, repeat=repeat)
        assert (cfg.CONST.BATCH_SIZE % 2) == 0
        self.caption_matches = data_dict['caption_matches']
        self.max_sentence_length = len(self.caption_tuples[0][0])
        self.iters_per_epoch = self.num_data // (self.batch_size // 2)  # overwrite iters_per_epoch

        if cfg.DIR.PROBLEMATIC_NRRD_PATH is not None:
            with open(cfg.DIR.PROBLEMATIC_NRRD_PATH, 'rb') as f:
                self.bad_model_ids = pickle.load(f)
        else:
            self.bad_model_ids = None

    def is_bad_model_id(self, model_id):
        """Code reuse.
        """
        if self.bad_model_ids is not None:
            return model_id in self.bad_model_ids
        else:
            return False

    def get_next_minibatch(self):
        half_batch_size = self.batch_size // 2
        if (self.cur + half_batch_size) >= self.num_data and self.repeat:
            self.shuffle_db_inds()

        db_inds = self.perm[self.cur:min(self.cur + half_batch_size, self.num_data)]
        self.cur += half_batch_size
        return db_inds

    def verify_batch(self, data_list):
        assert len(data_list) == cfg.CONST.BATCH_SIZE
        counter = Counter(data_list)
        for _, v in counter.items():
            assert v == 2

    def get_matching_tuples(self, db_ind, model_id_list, category_list):
        # Find a pair of captions such that:
        # 1. They are not the same caption.
        # 2. They correspond to the same model.
        # 3. There are no other captions in the batch that corresopnd to the same model.
        while True:
            caption_tuple = self.caption_tuples[db_ind]
            cur_category = caption_tuple[1]
            cur_model_id = caption_tuple[2]

            if self.is_bad_model_id(cur_model_id):
                db_ind = np.random.randint(self.num_data)  # Choose new caption
                continue
            try:
                # Make sure the batch has a unique set of model IDs
                if cfg.CONST.DATASET == 'shapenet':
                    assert cur_model_id not in model_id_list
                elif cfg.CONST.DATASET == 'primitives':
                    assert cur_category not in category_list
                else:
                    raise ValueError('Please select a valid dataset.')

                matching_caption_tuple = self.load_matching_caption_tuple(db_ind)

            except AssertionError:  # Retry if only one caption for current model
                db_ind = np.random.randint(self.num_data)  # Choose new caption
                continue
            break
        return caption_tuple, matching_caption_tuple

    @print_error
    def run(self):
        # Run the loop until exit flag is set
        while not self.exit.is_set() and self.cur < self.num_data:
            # Ensure that the network sees (almost) all data per epoch
            db_inds = self.get_next_minibatch()

            data_list = []  # captions
            category_list = []  # categories
            model_list = []  # models
            model_id_list = []
            for db_ind in db_inds:

                caption_tuple, matching_caption_tuple = self.get_matching_tuples(db_ind,
                                                                                 model_id_list,
                                                                                 category_list)

                model_id_list.append(caption_tuple[2])
                data_list.append(caption_tuple[0])  # 0th element is the caption
                data_list.append(matching_caption_tuple[0])  # 0th element is the caption
                category_list.append(caption_tuple[1])
                category_list.append(matching_caption_tuple[1])
                model_list.append(caption_tuple[2])
                model_list.append(matching_caption_tuple[2])

            if cfg.CONST.DATASET == 'shapenet':
                self.verify_batch(model_list)
            elif cfg.CONST.DATASET == 'primitives':
                self.verify_batch(category_list)
            else:
                raise ValueError('Please select a valid dataset.')

            captions_tensor = np.array(data_list).astype(np.int32)
            batch_data = {
                'caption_batch': captions_tensor,
                'category_list': category_list,
                'model_list': model_list,
            }

            # The following will wait until the queue frees
            self.data_queue.put(batch_data, block=True)

        print('Exiting enqueue process')

    def load_matching_caption_tuple(self, db_ind):
        """Loads a caption tuple that corresponds to the same model as the caption at index db_ind
        but makes sure that it's not the same exact caption.

        Args:
            db_ind: Index in the database of the reference caption.
        """
        caption_tuple = self.caption_tuples[db_ind]
        if cfg.CONST.DATASET == 'primitives':
            category = caption_tuple[1]
            match_idxs = self.caption_matches[category]
        else:
            model_id = caption_tuple[2]
            match_idxs = self.caption_matches[model_id]

        assert len(match_idxs) > 1

        # Select a caption from the matching caption list
        selected_idx = db_ind
        while selected_idx == db_ind:
            selected_idx = random.choice(match_idxs)

        if cfg.CONST.DATASET == 'primitives':
            assert category == self.caption_tuples[selected_idx][1]
        else:
            assert model_id == self.caption_tuples[selected_idx][2]

        return self.caption_tuples[selected_idx]


class CaptionDataProcessTestPhase(DataProcess):

    def __init__(self, data_queue, data_dict, repeat=False):
        """Initialize the Data Process. In this Data Process, each batch is composed of batch_size
        captions. We simply sample from the set of all captions, so each caption is only seen once
        (strictly) in each epoch for a given data process.

        Args:
            data_queue:
            data_dict: A dict with keys 'caption_tuples' and 'caption_matches'. caption_tuples is
                a list of caption tuples, where each caption tuple is (caption, model_category,
                model_id). caption_matches is a dict where the key is any model ID and the value
                is a list of the indices (ints) of caption tuples that describe the same model ID.
            repeat: Boolean flag indicating whether to continue adding to the queue after the epoch
                has ended.
        """
        self.caption_tuples = data_dict['caption_tuples']
        self.caption_matches = data_dict['caption_matches']
        self.max_sentence_length = len(self.caption_tuples[0][0])
        super(CaptionDataProcessTestPhase, self).__init__(data_queue, self.caption_tuples,
                                                          batch_size=cfg.CONST.BATCH_SIZE,
                                                          repeat=repeat)

    @print_error
    def run(self):
        # Run the loop until exit flag is set
        while not self.exit.is_set() and self.cur < self.num_data:
            # Ensure that the network sees (almost) all data per epoch
            db_inds = self.get_next_minibatch()

            data_list = []
            category_list = []  # categories
            model_list = []  # models
            for db_ind in db_inds:
                caption_tuple = self.caption_tuples[db_ind]

                data_list.append(caption_tuple[0])  # 0th element is the caption
                category_list.append(caption_tuple[1])
                model_list.append(caption_tuple[2])

            captions_tensor = np.array(data_list).astype(np.int32)
            batch_data = {
                'caption_batch': captions_tensor,
                'category_list': category_list,
                'model_list': model_list,
            }

            # The following will wait until the queue frees
            self.data_queue.put(batch_data, block=True)


class LBADataProcess(DataProcess):
    """Data process that returns a raw caption batch and a shape batch.
    """

    def __init__(self, data_queue, data_dict, repeat):
        self.caption_tuples = data_dict['caption_tuples']
        if cfg.CONST.DATASET == 'shapenet':
            self.caption_matches = data_dict['caption_matches']
        elif cfg.CONST.DATASET == 'primitives':
            self.caption_matches = data_dict['category_matches']
            self.category2modelid = data_dict['category2modelid']
        else:
            raise ValueError('Please select a valid dataset.')
        self.matches_keys = list(self.caption_matches.keys())
        self.n_captions_per_model = cfg.LBA.N_CAPTIONS_PER_MODEL
        if cfg.CONST.DATASET == 'shapenet':
            self.n_unique_shape_categories = cfg.CONST.BATCH_SIZE
            self.n_models_per_batch = self.n_unique_shape_categories
        elif cfg.CONST.DATASET == 'primitives':
            assert cfg.CONST.BATCH_SIZE % cfg.LBA.N_PRIMITIVE_SHAPES_PER_CATEGORY == 0
            self.n_unique_shape_categories = cfg.CONST.BATCH_SIZE // cfg.LBA.N_PRIMITIVE_SHAPES_PER_CATEGORY
            self.n_models_per_batch = cfg.CONST.BATCH_SIZE
        else:
            raise ValueError('Please select a valid dataset.')
        # self.n_models_per_batch = int(cfg.CONST.BATCH_SIZE / self.n_captions_per_model)
        # assert (cfg.CONST.BATCH_SIZE % self.n_captions_per_model) == 0
        # assert cfg.CONST.DATASET == 'shapenet'
        # NOTE: If we want to modify this for primitives data, we will have to change the call to
        # load_voxel so that we input a category and model ID.
        super(LBADataProcess, self).__init__(data_queue, range(len(self.caption_matches)),
                                             batch_size=self.n_unique_shape_categories,
                                             repeat=repeat)
        self.max_sentence_length = len(self.caption_tuples[0][0])

        lengths = []
        for cur_tup in self.caption_matches.values():
            lengths.append(len(cur_tup))
        counter = Counter(lengths)
        print('Dataset statistics')
        print('--> Format - num captions: num models with num captions')
        print('-->', counter)

        if (cfg.CONST.DATASET == 'shapenet') and (cfg.DIR.PROBLEMATIC_NRRD_PATH is not None):
            with open(cfg.DIR.PROBLEMATIC_NRRD_PATH, 'rb') as f:
                self.bad_model_ids = pickle.load(f)
        else:
            self.bad_model_ids = None

    def is_bad_model_id(self, model_id):
        """Code reuse.
        """
        if self.bad_model_ids is not None:
            return model_id in self.bad_model_ids
        else:
            return False

    def verify_batch(self, caption_tuples):
        """Simply verify that all caption tuples correspond to the same category and model ID.
        """
        category = caption_tuples[0][1]
        model_id = caption_tuples[0][2]
        for tup in caption_tuples:
            assert tup[1] == category
            if cfg.CONST.DATASET == 'shapenet':
                assert tup[2] == model_id
        return category, model_id

    @print_error
    def run(self):
        """Category and model lists dynamically change size depending on whether it is STS or TST
        mode.
        """
        # Run the loop until exit flag is set
        while not self.exit.is_set() and self.cur < self.num_data:
            # Ensure that the network sees (almost) all data per epoch
            db_inds = self.get_next_minibatch()

            shapes_list = []
            captions_list = []
            category_list = []
            model_id_list = []
            for db_ind in db_inds:  # Loop through each selected shape
                selected_shapes = []
                while True:
                    # cur_key is the model ID for shapenet, category for primitives
                    cur_key = self.matches_keys[db_ind]
                    caption_idxs = self.caption_matches[cur_key]
                    if len(caption_idxs) < self.n_captions_per_model:
                        db_ind = np.random.randint(self.num_data)
                        continue
                    selected_caption_idxs = random.sample(caption_idxs, k=self.n_captions_per_model)
                    selected_tuples = [self.caption_tuples[idx] for idx in selected_caption_idxs]
                    cur_category, cur_model_id = self.verify_batch(selected_tuples)

                    # Select shapes/models
                    if cfg.CONST.DATASET == 'shapenet':
                        selected_model_ids = [cur_model_id]
                    elif cfg.CONST.DATASET == 'primitives':
                        category_model_ids = self.category2modelid[cur_category]
                        selected_model_ids = random.sample(category_model_ids,
                                                           k=cfg.LBA.N_PRIMITIVE_SHAPES_PER_CATEGORY)
                    else:
                        raise ValueError('Please select a valid dataset')

                    for cur_model_id in selected_model_ids:
                        if self.is_bad_model_id(cur_model_id):
                            db_ind = np.random.randint(self.num_data)  # Choose new caption
                            continue
                        try:
                            cur_shape = load_voxel(cur_category, cur_model_id)
                        except FileNotFoundError:
                            print('ERROR: Cannot find file with the following model ID:', cur_key)
                            db_ind = np.random.randint(self.num_data)
                            continue
                        selected_shapes.append(cur_shape)
                    break
                selected_captions = [tup[0] for tup in selected_tuples]
                captions_list.extend(selected_captions)
                for selected_shape in selected_shapes:
                    shapes_list.append(selected_shape)
                if cfg.LBA.MODEL_TYPE == 'STS':
                    category_list.append(cur_category)
                    model_id_list.append(cur_model_id)
                elif cfg.LBA.MODEL_TYPE == 'TST' or cfg.LBA.MODEL_TYPE == 'MM':
                    cur_categories = [cur_category for _ in selected_captions]
                    cur_model_ids = [cur_model_id for _ in selected_captions]
                    category_list.extend(cur_categories)
                    model_id_list.extend(cur_model_ids)
                else:
                    raise ValueError('Please select a valid LBA mode.')

            # Length is number of captions
            # Index/label indicates which captions come from the same shape
            label_list = [x for x in range(self.n_unique_shape_categories)
                          for _ in range(self.n_captions_per_model)]

            batch_captions = np.array(captions_list).astype(np.int32)
            batch_shapes = np.array(shapes_list).astype(np.float32)
            batch_label = np.array(label_list).astype(np.int32)

            # The following will wait until the queue frees
            batch_data = {
                'raw_embedding_batch': batch_captions,
                'voxel_tensor_batch': batch_shapes,
                'caption_label_batch': batch_label,
                'category_list': category_list,
                'model_list': model_id_list,
            }
            self.data_queue.put(batch_data, block=True)


class LBADataProcessTestPhase(DataProcess):

    def __init__(self, data_queue, data_dict, repeat=False):
        """Initialize the Data Process. In this Data Process, each batch is composed of batch_size
        captions. We simply sample from the set of all captions, so each caption is only seen once
        (strictly) in each epoch for a given data process.

        Args:
            data_queue:
            data_dict: A dict with keys 'caption_tuples' and 'caption_matches'. caption_tuples is
                a list of caption tuples, where each caption tuple is (caption, model_category,
                model_id). caption_matches is a dict where the key is any model ID and the value
                is a list of the indices (ints) of caption tuples that describe the same model ID.
            repeat: Boolean flag indicating whether to continue adding to the queue after the epoch
                has ended.
        """
        assert cfg.LBA.TEST_MODE is not None
        self.mode = cfg.LBA.TEST_MODE
        if cfg.CONST.DATASET == 'shapenet':
            self.caption_matches = data_dict['caption_matches']
            self.caption_tuples = data_dict['caption_tuples']
        elif cfg.CONST.DATASET == 'primitives':
            self.caption_matches = data_dict['modelid_matches']

            if (self.mode == 'text') and not cfg.CONST.TEST_ALL_TUPLES:
                # Modify self.caption_tuples so it does not contain multiple instances of the same caption
                new_tuples = []
                seen_captions = []
                for cur_tup in data_dict['caption_tuples']:
                    cur_caption = tuple(cur_tup[0].tolist())
                    if cur_caption not in seen_captions:
                        seen_captions.append(cur_caption)
                        new_tuples.append(cur_tup)
                # new_dataset_size = len(new_tuples)
                self.caption_tuples = new_tuples
            elif (self.mode == 'shape') or cfg.CONST.TEST_ALL_TUPLES:
                self.caption_tuples = data_dict['caption_tuples']
            else:
                raise ValueError('Please select a valid LBA test mode.')
        else:
            raise ValueError('Please select a valid dataset.')
        self.matches_keys = list(self.caption_matches.keys())
        self.max_sentence_length = len(self.caption_tuples[0][0])
        if cfg.CONST.TEST_ALL_TUPLES:
            # Since we use caption_tuples instead of caption_matches, we need to be in text mode
            assert cfg.LBA.TEST_MODE == 'text'
        if (cfg.LBA.TEST_MODE == 'text') or cfg.CONST.TEST_ALL_TUPLES:
            super(LBADataProcessTestPhase, self).__init__(data_queue, self.caption_tuples,
                                                          batch_size=cfg.CONST.BATCH_SIZE,
                                                          repeat=repeat)
        elif cfg.LBA.TEST_MODE == 'shape':
            super(LBADataProcessTestPhase, self).__init__(data_queue, self.caption_matches,
                                                          batch_size=cfg.CONST.BATCH_SIZE,
                                                          repeat=repeat)
        else:
            raise ValueError('Please enter a valid LBA test mode.')

        if self.iters_per_epoch == 0:
            print('iters per epoch is 0! setting to 1.')
            self.iters_per_epoch = 1

    @print_error
    def run(self):
        # Run the loop until exit flag is set
        while not self.exit.is_set() and self.cur < self.num_data:
            # Ensure that the network sees (almost) all data per epoch
            db_inds = self.get_next_minibatch()

            data_list = []
            category_list = []  # categories
            model_list = []  # models
            shapes_list = []

            continue_while_loop = False
            for db_ind in db_inds:
                if self.mode == 'text':
                    caption_tuple = self.caption_tuples[db_ind]
                elif self.mode == 'shape':
                    cur_key = self.matches_keys[db_ind]
                    caption_idxs = self.caption_matches[cur_key]

                    # Pick the first caption tuple in the matches keys list
                    caption_tuple = self.caption_tuples[caption_idxs[0]]
                else:
                    raise ValueError('Please enter a valid LBA test mode')

                cur_category = caption_tuple[1]
                cur_model_id = caption_tuple[2]
                try:
                    cur_shape = load_voxel(cur_category, cur_model_id)
                except FileNotFoundError:
                    assert len(db_inds) == 1
                    print('File not found.')
                    print('Category:', cur_category)
                    print('Model ID:', cur_model_id)
                    print('Skipping.')
                    db_ind = np.random.randint(self.num_data)  # Choose new caption
                    continue_while_loop = True
                    break

                data_list.append(caption_tuple[0])  # 0th element is the caption
                category_list.append(cur_category)
                model_list.append(cur_model_id)
                shapes_list.append(cur_shape)

            if continue_while_loop is True:
                continue

            batch_captions = np.array(data_list).astype(np.int32)
            batch_shapes = np.array(shapes_list).astype(np.float32)

            if cfg.LBA.TEST_MODE == 'text':
                # Length is number of captions
                # Index/label indicates which captions come from the same shape
                if cfg.CONST.DATASET == 'shapenet':
                    # Map IDs for each shape
                    ids = {}
                    next_id = 0
                    for model_id in model_list:
                        if model_id not in ids:
                            ids[model_id] = next_id
                            next_id += 1

                    label_list = [ids[model_id] for model_id in model_list]
                    batch_label = np.array(label_list).astype(np.int32)
                elif cfg.CONST.DATASET == 'primitives':
                    # Map IDs for each shape
                    ids = {}
                    next_id = 0
                    for category_id in category_list:
                        if category_id not in ids:
                            ids[category_id] = next_id
                            next_id += 1

                    label_list = [ids[category_id] for category_id in category_list]
                    batch_label = np.array(label_list).astype(np.int32)
                else:
                    raise ValueError('Please select a valid dataset.')
            elif cfg.LBA.TEST_MODE == 'shape':
                batch_label = np.array(range(cfg.CONST.BATCH_SIZE))
            else:
                raise ValueError('Please select a valid LBA test phase mode.')

            batch_data = {
                'raw_embedding_batch': batch_captions,
                'voxel_tensor_batch': batch_shapes,
                'caption_label_batch': batch_label,
                'category_list': category_list,
                'model_list': model_list,
            }

            # The following will wait until the queue frees
            self.data_queue.put(batch_data, block=True)


def test_caption_process():
    from multiprocessing import Queue
    from lib.config import cfg
    from lib.utils import open_pickle, print_sentences

    cfg.CONST.DATASET = 'primitives'
    cfg.CONST.SYNTH_EMBEDDING = False

    asdf_captions = open_pickle(cfg.DIR.PRIMITIVES_VAL_DATA_PATH)

    data_queue = Queue(3)

    data_process = CaptionDataProcess(data_queue, asdf_captions, repeat=True)
    data_process.start()
    caption_batch = data_queue.get()
    captions_tensor, category_list, model_list = caption_batch

    assert captions_tensor.shape[0] == len(category_list)
    assert len(category_list) == len(model_list)

    for i in range(len(category_list)):
        print('---------- %03d ------------' % i)
        caption = captions_tensor[i]
        category = category_list[i]
        model_id = model_list[i]
        # print('Caption:', caption)
        # print('Converted caption:')

        # Generate sentence
        # data_list = [{'raw_caption_embedding': caption}]
        # print_sentences(json_path, data_list)

        print('Category:', category)
        # print('Model ID:', model_id)

    kill_processes(data_queue, [data_process])


def test_lba_process():
    from multiprocessing import Queue
    from lib.config import cfg
    from lib.utils import open_pickle, print_sentences, get_json_path

    cfg.CONST.BATCH_SIZE = 8
    cfg.CONST.DATASET = 'shapenet'
    cfg.CONST.SYNTH_EMBEDDING = False

    caption_data = open_pickle(cfg.DIR.VAL_DATA_PATH)
    data_queue = Queue(3)
    json_path = get_json_path()

    data_process = LBADataProcess(data_queue, caption_data, repeat=True)
    data_process.start()
    caption_batch = data_queue.get()
    category_list = caption_batch['category_list']
    model_list = caption_batch['model_list']

    for k, v in caption_batch.items():
        if isinstance(v, list):
            print('Key:', k)
            print('Value length:', len(v))
        elif isinstance(v, np.ndarray):
            print('Key:', k)
            print('Value shape:', v.shape)
        else:
            print('Other:', k)
    print('')

    for i in range(len(category_list)):
        print('---------- %03d ------------' % i)
        category = category_list[i]
        model_id = model_list[i]

        # Generate sentence
        for j in range(data_process.n_captions_per_model):
            caption_idx = data_process.n_captions_per_model * i + j
            caption = caption_batch['raw_embedding_batch'][caption_idx]
            # print('Caption:', caption)
            # print('Converted caption:')
            data_list = [{'raw_caption_embedding': caption}]
            print_sentences(json_path, data_list)
            print('Label:', caption_batch['caption_label_batch'][caption_idx])

        print('Category:', category)
        print('Model ID:', model_id)

    kill_processes(data_queue, [data_process])


if __name__ == '__main__':
    # test_compat_fn_process()
    test_lba_process()
