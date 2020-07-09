from functools import wraps
import click
from os.path import splitext, basename
import sys
import warnings

import treelog as log

from . import Config
from ifem_to_vt.reader import get_reader
from ifem_to_vt.writer import get_writer


def suppress_warnings(func):
    @wraps(func)
    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'Conversion of the second argument of issubdtype')
            return func(*args, **kwargs)
    return inner


class RichOutputLog(log.RichOutputLog):

    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def write(self, text, level):
        message = ''.join([self._cmap[level.value], text, '\033[0m\n', self._current])
        click.echo(message, file=self.stream, nl=False)


@click.command()
@click.option('--basis', '-b', multiple=True, help='Include fields in this basis.')
@click.option('--geometry', '-g', default=None, help='Use this basis to provide geometry.')
@click.option('--nvis', '-n', default=1, help='Extra sampling points per element.')
@click.option('--fmt', '-f', type=click.Choice(['vtf', 'vtk', 'vtu', 'pvd', 'nc']), required=False, help='Output format.')
@click.option('--mode', '-m', type=click.Choice(['binary', 'ascii', 'appended']), default='binary', help='Output mode.')
@click.option('--last', is_flag=True, help='Read only the last step.')
@click.option('--endianness', type=click.Choice(['native', 'little', 'big']), default='native')
@click.option('--debug', 'verbosity', flag_value='debug')
@click.option('--info', 'verbosity', flag_value='info', default=True)
@click.option('--user', 'verbosity', flag_value='user')
@click.option('--warning', 'verbosity', flag_value='warning')
@click.option('--error', 'verbosity', flag_value='error')
@click.option('--rich/--no-rich', default=True)
@click.argument('infile', type=str, required=True)
@click.argument('outfile', type=str, required=False)
@suppress_warnings
def convert(verbosity, rich, infile, fmt, outfile, **kwargs):

    # Set up logging
    if rich:
        logger = RichOutputLog(sys.stdout)
    else:
        logger = log.TeeLog(
            log.FilterLog(log.StdoutLog(), maxlevel=log.proto.Level.user),
            log.FilterLog(log.StderrLog(), minlevel=log.proto.Level.warning),
        )
    log.current = log.FilterLog(logger, minlevel=getattr(log.proto.Level, verbosity))

    # Determine filename of output
    if outfile and not fmt:
        _, fmt = splitext(outfile)
        fmt = fmt[1:].lower()
    elif not outfile:
        fmt = fmt or 'vtu'
        filename = basename(infile)
        base, _ = splitext(filename)
        outfile = '{}.{}'.format(base, fmt)

    try:
        Writer = get_writer(fmt)
    except ValueError as e:
        log.error(e)
        sys.exit(1)

    config = Config(**kwargs)
    with get_reader(infile, config) as r, Writer(outfile, config) as w:
        r.write(w)


if __name__ == '__main__':
    convert()
