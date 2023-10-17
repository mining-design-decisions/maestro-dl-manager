import random

from ..config.core import Config
from ..config.arguments import Argument
from ..config.constraints import Constraint

from . import base


class RandomUpSampler(base.AbstractUpSampler):

    def upsample(self, indices, targets, labels, keys, *features):
        # Delegate to super and use upsample_class instead
        return super().upsample(indices, targets, labels, keys, *features)

    def upsample_class(self, indices, target, labels, keys, *features):
        new = tuple(random.choices(indices, k=target - len(indices)))
        return (
            labels[new],
            keys[new]
            *(
                f[new] for f in features
            )
        )

    @classmethod
    def get_constraints(cls) -> list[Constraint]:
        return super().get_constraints()

    @classmethod
    def get_arguments(cls):
        return super().get_arguments()
