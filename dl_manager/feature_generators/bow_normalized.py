from .bow import AbstractBOW


class BOWNormalized(AbstractBOW):

    @staticmethod
    def get_word_value(divider):
        return 1/divider
