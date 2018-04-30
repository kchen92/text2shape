from lib.cwgan import CWGAN
from models.t2s_generator_component import Text2ShapeGenerator1
from models.t2s_critic_component import Text2ShapeDiscriminator2


class CWGAN1(CWGAN):
    def __init__(self, inputs_dict, is_training, reuse=False, name='cwgan_1'):
        self._t2s_generator_class = Text2ShapeGenerator1
        self._t2s_critic_class = Text2ShapeDiscriminator2
        super(CWGAN1, self).__init__(inputs_dict, is_training, reuse=reuse, name=name)

    @property
    def t2s_generator_class(self):
        return self._t2s_generator_class

    @property
    def t2s_critic_class(self):
        return self._t2s_critic_class
