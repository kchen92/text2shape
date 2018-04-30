from lib.config import cfg
from lib.net_component import LBANetComponent
from models.encoder_component import CNNRNNTextEncoder, ShapeEncoder


class LBAComponent1(LBANetComponent):
    """Same model structure as LBA1.
    """

    def __init__(self, is_training, reuse=False, name='lba_component_1', no_scope=False):
        self._text_encoder_class = CNNRNNTextEncoder
        self._shape_encoder_class = ShapeEncoder
        super(LBAComponent1, self).__init__(is_training, reuse=reuse, name=name, no_scope=no_scope)

    @property
    def text_encoder_class(self):
        return self._text_encoder_class

    @property
    def shape_encoder_class(self):
        return self._shape_encoder_class
