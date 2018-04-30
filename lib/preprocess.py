from lib.nrrd_rw import read_nrrd
from lib.data_io import get_voxel_file

import numpy as np


def load_voxel(category, model_id):
    """Loads the voxel tensors given the model category and model ID.

    Args:
        category: Model category (e.g. '03001627')
        model_id: Model ID (e.g. '587ee5822bb56bd07b11ae648ea92233')

    Returns:
        voxel_tensor: Voxel tensor of shape (height x width x depth x channels).
    """
    voxel_fn = get_voxel_file(category, model_id)
    voxel_tensor = read_nrrd(voxel_fn)
    return voxel_tensor


class TextPreprocessor(object):
    def __init__(self, inputs_list, swap_model_category=False):
        self.__captions = inputs_list['captions']
        self.__word_to_idx = inputs_list['word_to_idx']
        self.__idx_to_word = inputs_list['idx_to_word']
        self.__max_caption_length = (inputs_list['max_length'] if inputs_list['max_length'] != 0
                                     else self.compute_max_caption_length())
        self.__vocab_size = len(self.__word_to_idx) + 1
        self.__input_shape = (self.__max_caption_length)
        self.__dtype = np.int32
        self.__swap_model_category = swap_model_category

        self.print_dataset_info()

    def compute_max_caption_length(self):
        """Compute the length of the longest caption in the dataset.
        """
        max_len = -1
        for cur_caption in self.__captions:
            if len(cur_caption['caption']) > max_len:
                max_len = len(cur_caption['caption'])
        return max_len

    def preprocess(self):
        """Preprocesses all of the captions in the dataset and puts them in a
        list. Also puts the matching captions (AKA captions that correspond to
        the same model) into a dictionary.

        Returns:
            processed_inputs: A list of tuples, where each tuple is
                (processed caption, category, model_id)
            caption_matches: A dictionary containing caption matches. A key in
                this dictionary is a tuple of (model category, model ID). The
                values are a list of indices for the captions in self._captions
                that match the model category and model ID.
        """
        processed_inputs = []
        caption_matches = {}
        for caption_idx, caption_tuple in enumerate(self.__captions):
            cur_caption = caption_tuple['caption']
            cur_category = caption_tuple['category']
            cur_model_id = caption_tuple['model']

            processed_caption = self.preprocess_caption(cur_caption)
            processed_tuple = (processed_caption, cur_category, cur_model_id)

            # update processed_inputs list
            processed_inputs.append(processed_tuple)

            # update caption_matches_dict
            if self.swap_model_category is True:
                dict_key = cur_category
            else:
                dict_key = cur_model_id
            cur_matches = caption_matches.get(dict_key, [])
            old_len = len(cur_matches)
            cur_matches.append(caption_idx)
            caption_matches[dict_key] = cur_matches
            assert (old_len + 1) == len(caption_matches.get(dict_key, []))

        return processed_inputs, caption_matches

    def preprocess_caption(self, cur_caption):
        """Preprocesses the caption by converting each word into a word index.

        The 0-index is reserved for the <EOS> or "no character" character. This
        tells us when a caption has ended.

        The last word in the vocabulary corresponds to the <UNK> token,
        according to the vocab/json generator.

        Args:
            cur_caption: The current caption to preprocess. This should be a
                list of words (strings).

        Returns:
            processed_text: A NumPy vector of word indices representing the
                processed caption.
        """
        word_to_idx = self.__word_to_idx

        # a processed text sample is a max caption length numpy array/vector
        # remember that indexes in word_to_idx are 1-indexed
        processed_text = np.zeros(self.__input_shape, dtype=self.__dtype)
        for idx, cur_word in enumerate(cur_caption):
            processed_text[idx] = word_to_idx[cur_word]

        return processed_text

    def print_dataset_info(self):
        print('-------------- DATASET INFO ----------------')
        print('vocabulary size:', self.__vocab_size)
        print('max caption length:', self.__max_caption_length)
        print('number of captions', len(self.__captions))

    @property
    def input_shape(self):
        return self.__input_shape

    @property
    def dtype(self):
        return self.__dtype

    @property
    def captions(self):
        return self.__captions

    @property
    def vocab_size(self):
        return self.__vocab_size

    @property
    def max_caption_length(self):
        return self.__max_caption_length

    @property
    def word_to_idx(self):
        return self.__word_to_idx

    @property
    def idx_to_word(self):
        return self.__idx_to_word

    @property
    def swap_model_category(self):
        return self.__swap_model_category
