from contextlib import contextmanager


class Config:
    """Configuration object that should be shared between reader and
    writer.  It is initialized by the main function, but may have its
    attributes manipulated by either reader or writer depending on
    their needs.
    """

    basis = ()
    geometry = None
    nvis = 1
    mode = 'binary'
    last = False
    endianness = 'native'

    def __init__(self):
        self._required_keys = set()

    def assign(self, key, value):
        if key in self._required_keys:
            assert value == getattr(self, key)
        else:
            setattr(self, key, value)

    def require(self, **kwargs):
        for key, value in kwargs.items():
            self.assign(key, value)
            self._required_keys.add(key)

    @contextmanager
    def __call__(self, **kwargs):
        prev = dict(self.__dict__)
        prev_required = set(self._required_keys)
        for key, value in kwargs.items():
            self.assign(key, value)
        yield
        self.__dict__ = prev
        self._required_keys = prev_required


config = Config()
