import importlib


def get_writer(fmt):
    try:
        module = importlib.import_module('ifem_to_vt.writer.{}'.format(fmt))
        return lambda fn, config: module.Writer(fn, config)
    except ModuleNotFoundError:
        raise ValueError('Unsupported format: {}'.format(fmt))
