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

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self._required_keys = set()

    def require(self, **kwargs):
        for key, value in kwargs.items():
            if key in self._required_keys:
                assert value == getattr(self, key)
            else:
                self._required_keys.add(key)
                setattr(self, key, value)
