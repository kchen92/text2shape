from lib.lba import LBA
from models.encoder_component import CNNRNNTextEncoder, ShapeEncoder


class LBA1(LBA):

    def __init__(self, inputs_dict, is_training, reuse=False, name='lba_net1'):
        self._text_encoder_class = CNNRNNTextEncoder
        self._shape_encoder_class = ShapeEncoder
        super(LBA1, self).__init__(inputs_dict, is_training, reuse=reuse, name=name)

    @property
    def text_encoder_class(self):
        return self._text_encoder_class

    @property
    def shape_encoder_class(self):
        return self._shape_encoder_class
