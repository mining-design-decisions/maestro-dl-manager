def bulk_clean_text_parallel(documents: list[str],
                             formatting_handling: str,
                             num_threads: int) -> list[str]: ...


def bulk_replace_parallel(documents: list[list[str]],
                          needles: list[list[str]],
                          replacement: list[str],
                          num_threads: int) -> list[list[str]]: ...


def bulk_replace_parallel_string(documents: list[list[str]],
                                 needles: list[str],
                                 replacement: str,
                                 num_threads: int) -> list[list[str]]: ...



class Tagger:
    def __init__(self,
                 weights: dict[str, dict[str, float]],
                 classes: set[str],
                 tagdict: dict[str, str]): ...

    def tag(self, sentence: list[str]) -> list[tuple[str, str]]: ...

    def bulk_tag(self,
                 documents: list[list[list[str]]]) -> list[list[list[tuple[str, str]]]]: ...

    def bulk_tag_parallel(self,
                          documents: list[list[list[str]]],
                          num_threads: int) -> list[list[list[tuple[str, str]]]]: ...
