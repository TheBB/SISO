class Writer:

    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        return self

    def __exit__(self, type_, value, backtrace):
        pass
