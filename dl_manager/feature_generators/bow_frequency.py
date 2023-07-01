from .bow import AbstractBOW


class BOWFrequency(AbstractBOW):

    @staticmethod
    def get_word_value(divider):
        return 1
