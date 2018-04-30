from functools import partial

from lib.config import cfg
from models.cwgan_models import CWGAN1
from models.classifier_models import Classifier1, Classifier128
from models.lba_models import LBA1
from lib.data_process_encoder import (CaptionDataProcess, CaptionDataProcessTestPhase,
                                      LBADataProcess, LBADataProcessTestPhase)
from lib.data_process_gan import (GANDataProcess, GANDataProcessTestPhase,
                                  CWGANMetricEmbeddingDataProcess)
from lib.data_process_classifier import ShapeClassifierDataProcess


MODELS = [
    CWGAN1,
    Classifier1,
    Classifier128,
    LBA1,
]


def get_models():
    """Returns a tuple of models.
    """
    return MODELS


def get_data_process_pairs(NetClass, is_training):
    """Returns the DataProcess class corresponding to the input NetClass.

    Args:
        NetClass: The network class.
        is_training: Boolean flag indicating whether the network is training or not.

    Returns:
        data_process_class: The DataProcess (sub)class corresponding with the input NetClass.
    """
    CWGANProcess = CWGANMetricEmbeddingDataProcess
    if is_training:
        LBAProcess = LBADataProcess
    else:
        if cfg.LBA.TEST_MODE is None:
            LBAProcess = LBADataProcess
        else:
            LBAProcess = LBADataProcessTestPhase
    if cfg.CONST.SHAPENET_CT_CLASSIFIER is True:
        ClassifierProcess = ShapeClassifierDataProcess
    else:
        ClassifierProcess = GANDataProcessTestPhase

    DATA_PROCESS_PAIRS = {
        'CWGAN1': CWGANProcess,
        'Classifier1': ClassifierProcess,
        'Classifier128': ClassifierProcess,
        'LBA1': LBAProcess,
    }

    return DATA_PROCESS_PAIRS[NetClass]


def load_model(name):
    """Creates and returns an instance of the model given its class name.

    Args:
        name: Name of the model (e.g. 'CWGAN1').

    Returns:
        NetClass: The network class corresponding with the input name.
    """
    all_models = get_models()
    model_dict = {model.__name__: model for model in all_models}

    if name not in model_dict:  # invalid model name
        print('Invalid model:', name)

        # print a list of valid model names
        print('Options are:')
        for model in all_models:
            print('\t* {}'.format(model.__name__))
        raise ValueError('Please select a valid model.')

    NetClass = model_dict[name]

    return NetClass
