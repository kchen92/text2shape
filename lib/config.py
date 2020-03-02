"""Configuration file for text2shape.

Modified from: https://github.com/chrischoy/3D-R2N2/blob/master/lib/config.py
"""
from easydict import EasyDict as edict

import os


__C = edict()
# Consumers can get config by:
#   import setup_cfg
cfg = __C


# #
# # Common
# #

__C.CONST = edict()
__C.CONST.DATASET = 'shapenet'  # (shapenet, primitives) - can be modified by command line
__C.CONST.IMPROVED_WGAN = False
__C.CONST.SHAPENET_CT_CLASSIFIER = False
__C.CONST.SYNTH_EMBEDDING = False
__C.CONST.N_VOX = 32
__C.CONST.N_MINIBATCH_TEST = None  # Number of minibatches to use for test phase
__C.CONST.BATCH_SIZE = 100
__C.CONST.RNG_SEED = 123  # Consider removing later
__C.CONST.NUM_WORKERS = 3
__C.CONST.QUEUE_CAPACITY = 20  # Maximum number of minibatches in data queue
__C.CONST.PRINT_FREQ = 20
__C.CONST.TEST_ALL_TUPLES = False
__C.CONST.REED_CLASSIFIER = False

# Network
__C.NETWORK = None
__C.VAL_NETWORK = 'Classifier128'


#
# Directories
#
__C.DIR = edict()
# Path where taxonomy.json is stored
__C.DIR.RGB_VOXEL_PATH = r'C:\Users\kohei\minerva_ai\text2shape\text2shape-data\shapenet\nrrd_256_filter_div_32_solid\%s\%s.nrrd'  # TODO: Modify this
__C.DIR.LOG_PATH = './output/default'
__C.DIR.CKPT_PATH = None
__C.DIR.VAL_CKPT_PATH = None
__C.DIR.DATA_PATH = r'C:\Users\kohei\minerva_ai\text2shape\text2shape-data'
__C.DIR.SHAPENET_DATA_PATH = os.path.join(__C.DIR.DATA_PATH, 'shapenet')
__C.DIR.RAW_CAPTION_CSV = r'C:\Users\kohei\minerva_ai\text2shape\text2shape-data\captions.tablechair.csv'  # TODO: Modify this
__C.DIR.PROBLEMATIC_NRRD_PATH = os.path.abspath(os.path.join(__C.DIR.SHAPENET_DATA_PATH, 'problematic_nrrds_shapenet_unverified_256_filtered_div_with_err_textures.p'))
__C.DIR.JSON_PATH = os.path.abspath(os.path.join(__C.DIR.SHAPENET_DATA_PATH, 'shapenet.json'))
__C.DIR.TRAIN_DATA_PATH = os.path.abspath(os.path.join(__C.DIR.SHAPENET_DATA_PATH, 'processed_captions_train.p'))
__C.DIR.VAL_DATA_PATH = os.path.abspath(os.path.join(__C.DIR.SHAPENET_DATA_PATH, 'processed_captions_val.p'))
__C.DIR.TEST_DATA_PATH = os.path.abspath(os.path.join(__C.DIR.SHAPENET_DATA_PATH, 'processed_captions_test.p'))
__C.DIR.SHAPENET_METRIC_EMBEDDINGS_TRAIN = os.path.abspath(os.path.join(__C.DIR.SHAPENET_DATA_PATH, 'shapenet-embeddings', 'text_embeddings_train.p'))
__C.DIR.SHAPENET_METRIC_EMBEDDINGS_VAL = os.path.abspath(os.path.join(__C.DIR.SHAPENET_DATA_PATH, 'shapenet-embeddings', 'text_embeddings_val.p'))
__C.DIR.SHAPENET_METRIC_EMBEDDINGS_TEST = os.path.abspath(os.path.join(__C.DIR.SHAPENET_DATA_PATH, 'shapenet-embeddings', 'text_embeddings_test.p'))

# Synthetic primitives dataset
__C.DIR.PRIMITIVES_RGB_VOXEL_PATH = r'C:\Users\kohei\minerva_ai\text2shape\text2shape-data\primitives\%s\%s'  # TODO: Modify this
__C.DIR.PRIMITIVES_DATA_PATH = os.path.join(__C.DIR.DATA_PATH, 'primitives')
__C.DIR.PRIMITIVES_JSON_PATH = os.path.join(__C.DIR.PRIMITIVES_DATA_PATH, 'primitives.json')
__C.DIR.PRIMITIVES_TRAIN_DATA_PATH = os.path.abspath(os.path.join(__C.DIR.PRIMITIVES_DATA_PATH, 'processed_captions_train.p'))
__C.DIR.PRIMITIVES_VAL_DATA_PATH = os.path.abspath(os.path.join(__C.DIR.PRIMITIVES_DATA_PATH, 'processed_captions_val.p'))
__C.DIR.PRIMITIVES_TEST_DATA_PATH = os.path.abspath(os.path.join(__C.DIR.PRIMITIVES_DATA_PATH, 'processed_captions_test.p'))
__C.DIR.PRIMITIVES_ALL_SPLITS_DATA_PATH = os.path.abspath(os.path.join(__C.DIR.PRIMITIVES_DATA_PATH, 'combined_splits.p'))
__C.DIR.PRIMITIVES_METRIC_EMBEDDINGS_TRAIN = None
__C.DIR.PRIMITIVES_METRIC_EMBEDDINGS_VAL = None
__C.DIR.PRIMITIVES_METRIC_EMBEDDINGS_TEST = None

__C.DIR.TOOLKIT_PATH = r'C:\Users\kohei\minerva_ai\sstk'  # TODO: Modify this


# #
# # Training
# #
__C.TRAIN = edict()
__C.TRAIN.NUM_EPOCHS = 100
__C.TRAIN.AUGMENT_MAX = 0
__C.TRAIN.OPTIMIZER = 'adam'
__C.TRAIN.LEARNING_RATE = 1e-4
__C.TRAIN.DECAY_STEPS = 2500
__C.TRAIN.DECAY_RATE = 0.95
__C.TRAIN.STAIRCASE = True
__C.TRAIN.CKPT_FREQ = 2500
__C.TRAIN.SUMMARY_FREQ = 250
__C.TRAIN.VALIDATION_FREQ = __C.TRAIN.CKPT_FREQ


# #
# # GAN
# #
__C.GAN = edict()
__C.GAN.NOISE_DIST = 'uniform'
__C.GAN.NOISE_MEAN = 0.
__C.GAN.NOISE_STDDEV = 1.
__C.GAN.NOISE_UNIF_ABS_MAX = 1.
__C.GAN.NOISE_SIZE = 32
__C.GAN.D_OPTIMIZER = 'adam'
__C.GAN.D_LEARNING_RATE_MULTIPLIER = 1.  # initial discriminator learning rate
__C.GAN.D_ACCURACY_THRESHOLD = 0.8
__C.GAN.COLOR_EVALUATOR = 'hsv'  # Doesn't matter
__C.GAN.COLOR_METRIC = 'EMD'
__C.GAN.INTERP = False


# #
# # WGAN
# #
__C.WGAN = edict()
__C.WGAN.NUM_CRITIC_STEPS = 5
__C.WGAN.INTENSE_TRAINING_STEPS = 25
__C.WGAN.INTENSE_TRAINING_FREQ = 500
__C.WGAN.INTENSE_TRAINING_INTENSITY = 100
__C.WGAN.GP_COEFF = 10.
__C.WGAN.MATCH_LOSS_COEFF = 2.
__C.WGAN.FAKE_MATCH_LOSS_COEFF = 1.
__C.WGAN.FAKE_MISMATCH_LOSS_COEFF = 1.
__C.WGAN.TEXT2TEXT_WEIGHT = 10.
__C.WGAN.SHAPE2SHAPE_WEIGHT = 10.


# #
# # LBA
# #
__C.LBA = edict()
__C.LBA.MODEL_TYPE = 'TST'
__C.LBA.N_CAPTIONS_PER_MODEL = 2
__C.LBA.VISIT_WEIGHT = 0.25
__C.LBA.WALKER_WEIGHT = 1.0
__C.LBA.NORMALIZE = True
__C.LBA.TEST_MODE = None
__C.LBA.DIST_TYPE = 'standard'
__C.LBA.PROJECT_FREQ = 10
__C.LBA.NO_LBA = False
__C.LBA.CLASSIFICATION = False
__C.LBA.CLASSIFICATION_MULTIPLIER = 10.
__C.LBA.COSINE_DIST = True
__C.LBA.METRIC_MULTIPLIER = 1.
__C.LBA.INVERTED_LOSS = True
__C.LBA.MAX_NORM = 10.
__C.LBA.TEXT_NORM_MULTIPLIER = 2.
__C.LBA.SHAPE_NORM_MULTIPLIER = 2.
__C.LBA.N_PRIMITIVE_SHAPES_PER_CATEGORY = 2


# #
# # Config modifification
# #

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b.keys():
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options.
    """
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line).
    """
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d.keys()
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d.keys()
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert (d[subkey] is None) or isinstance(value, type(d[subkey])), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
