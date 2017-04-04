import h5py


class Reader:

    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.h5 = h5py.File(self.filename, 'r')
        return self

    def __exit__(self, type_, value, backtrace):
        self.h5.close()

    @property
    def ntimes(self):
        return len(self.h5)
