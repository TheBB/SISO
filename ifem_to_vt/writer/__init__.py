import importlib


def get_writer(fmt):
    try:
        module = importlib.import_module('ifem_to_vt.writer.{}'.format(fmt))
        return module.Writer
    except ModuleNotFoundError:
        raise ValueError('Unsupported format: {}'.format(fmt))
