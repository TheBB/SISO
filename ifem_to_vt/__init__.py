from contextlib import contextmanager


class Config:
    """Configuration object that should be shared between reader and
    writer.  It is initialized by the main function, but may have its
    attributes manipulated by either reader or writer depending on
    their needs.
    """


    # Number of subdivisions for additional resolution.
    nvis = 1

    # Whether the data set contains multiple time steps.
    multiple_timesteps = True

    # Whether to copy only the final time step.
    only_final_timestep = False

    # Output mode. Used by the VTK and VTF writers.
    output_mode = 'binary'

    # Input endianness indicator. Used by the SIMRA reader.
    input_endianness = 'native'

    # List of basis objects to copy to output. Used by the IFEM
    # reader.
    only_bases = ()

    # Which basis should be used to represent the geometry. Used by
    # the IFEM reader.
    geometry_basis = None


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
        try:
            yield
        finally:
            self.__dict__ = prev
            self._required_keys = prev_required


config = Config()
