import importlib


def get_writer(fmt):
    try:
        module = importlib.import_module('ifem_to_vt.writer.{}'.format(fmt))
        return lambda fn: module.Writer(fn)
    except ModuleNotFoundError:
        raise ValueError('Unsupported format: {}'.format(fmt))
