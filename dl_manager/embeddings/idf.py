import collections
import json
import math
import pathlib

from .embedding_generator import AbstractEmbeddingGenerator
from ..config.arguments import Argument, IntArgument
from ..config.core import Config
from ..config.constraints import Constraint


class IDFGenerator(AbstractEmbeddingGenerator):
    def generate_embedding(
        self, issues: list[list[str]], path: pathlib.Path, conf: Config
    ):
        # Generate a frequency table
        frequencies = collections.defaultdict(int)
        for issue in issues:
            for word in set(issue):
                frequencies[word] += 1
        # Select words to be contained in dictionary
        threshold = self.params["min-doc-count"]
        dictionary = [
            word for word, frequency in frequencies.items() if frequency >= threshold
        ]
        # Let's make the dictionary sorted
        dictionary.sort()
        # Compute idf
        n = len(issues)
        idf = {term: math.log10(n / freq) for term, freq in frequencies.items()}
        # Save dictionary
        with open(path, "w") as file:
            json.dump({"layout": dictionary, "idf": idf}, file)

    @classmethod
    def get_constraints(cls) -> list[Constraint]:
        return super().get_constraints()

    @classmethod
    def get_arguments(cls) -> dict[str, Argument]:
        return super(IDFGenerator, IDFGenerator).get_arguments() | {
            "min-doc-count": IntArgument(
                name="min-doc-count",
                description="Minimum document frequency for a word to be included in the embedding.",
                minimum=0,
            )
        }
