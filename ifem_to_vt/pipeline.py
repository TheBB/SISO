from . import config
from .reader import Reader
from .writer import Writer


def pipeline(reader: Reader, writer: Writer):
    """Main driver for moving data from reader to writer."""
    reader.write(writer)
