import importlib


def get_writer(fmt, *args, **kwargs):
    try:
        module = importlib.import_module('ifem_to_vt.writer.{}'.format(fmt))
        return lambda fn: module.Writer(fn, *args, **kwargs)
    except ModuleNotFoundError:
        raise ValueError('Unsupported format: {}'.format(fmt))
