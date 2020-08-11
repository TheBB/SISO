from functools import wraps
from pathlib import Path
import sys
import traceback
import warnings

import click
import treelog as log

from . import config
from ifem_to_vt.reader import Reader
from ifem_to_vt.writer import Writer


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
@click.option('--fmt', '-f', type=click.Choice(['vtf', 'vtk', 'vtu', 'pvd', 'nc']), required=False, help='Output format.')

# Options that are forwarded to config
@click.option('--basis', '-b', 'only_bases', multiple=True, help='Include fields in this basis.')
@click.option('--geometry', '-g', 'geometry_basis', default=None, help='Use this basis to provide geometry.')
@click.option('--volumetric', 'volumetric', flag_value='volumetric', help='Only include volumetric fields.')
@click.option('--planar', 'volumetric', flag_value='planar', help='Only include planar (surface) fields.')
@click.option('--extrude', 'volumetric', flag_value='extrude', help='Extrude planar (surface) fields.')
@click.option('--nvis', '-n', 'nvis', default=1, help='Extra sampling points per element.')
@click.option('--last', 'only_final_timestep', is_flag=True, help='Read only the last step.')
@click.option('--endianness', 'input_endianness', type=click.Choice(['native', 'little', 'big']), default='native')
@click.option('--mode', '-m', 'output_mode', type=click.Choice(['binary', 'ascii', 'appended']),
              default='binary', help='Output mode.')

# Logging and verbosity
@click.option('--debug', 'verbosity', flag_value='debug')
@click.option('--info', 'verbosity', flag_value='info', default=True)
@click.option('--user', 'verbosity', flag_value='user')
@click.option('--warning', 'verbosity', flag_value='warning')
@click.option('--error', 'verbosity', flag_value='error')
@click.option('--rich/--no-rich', default=True)

# Filenames
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

    # Convert to pathlib
    infile = Path(infile)
    outfile = Path(outfile) if outfile else None

    # Determine filename of output
    if outfile and not fmt:
        fmt = outfile.suffix[1:].lower()
    elif not outfile:
        fmt = fmt or 'pvd'
        outfile = Path(infile.name).with_suffix(f'.{fmt}')

    try:
        # The config can influence the choice of readers or writers,
        # so apply it first
        with config(**kwargs):
            ReaderClass = Reader.find_applicable(infile)
            WriterClass = Writer.find_applicable(fmt)
            with ReaderClass(infile) as r, WriterClass(outfile) as w:
                r.write(w)
    except Exception as e:
        if verbosity == 'debug':
            # In debug mode, allow exceptions to filter through in raw form
            traceback.print_exc()
        else:
            log.error(f"Error: {e}")
            log.error("More information may be available with --debug")
        sys.exit(1)


if __name__ == '__main__':
    convert()
