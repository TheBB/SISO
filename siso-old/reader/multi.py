from .reader import Reader

from typing import List


class MultiReader(Reader):

    readers: List[Reader]

    @classmethod
    def applicable(cls, _) -> bool:
        return False

    def validate(self):
        super().validate()

    def __init__(self, readers: List[Reader]):
        self.readers = readers

    def __enter__(self):
        for reader in self.readers:
            reader.__enter__()
        return self

    def __exit__(self, *args):
        for reader in self.readers:
            reader.__exit__(*args)

    # def steps(self):
