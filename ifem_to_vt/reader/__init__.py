import importlib
from os.path import splitext


def get_reader(filename, config):
    _, fmt = splitext(filename)
    fmt = fmt[1:].lower()
    try:
        module = importlib.import_module('ifem_to_vt.reader.{}'.format(fmt))
        if hasattr(module, 'get_reader'):
            return module.get_reader(filename, config)
        else:
            return module.Reader(filename, config)
    except ModuleNotFoundError:
        raise ValueError('Unsupported format: {}'.format(fmt))
