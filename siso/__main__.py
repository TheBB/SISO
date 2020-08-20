from functools import wraps
from pathlib import Path
import sys
import traceback
import warnings

import click
import treelog as log

from . import config, ConfigSource
from .pipeline import pipeline
from .util import split_commas
from .reader import Reader
from .writer import Writer


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


class Option(click.Option):
    """A custom option class that tracks which options have been
    explicitly set by the user, and stores them in a special attribute
    attached to the context object."""

    def process_value(self, ctx, value):
        if value is not None:
            if not hasattr(ctx, 'explicit_options'):
                ctx.explicit_options = set()
            ctx.explicit_options.add(self.name)
        return super().process_value(ctx, value)


def tracked_option(*args, **kwargs):
    return click.option(*args, **kwargs, cls=Option)


@click.command()
@click.option('--fmt', '-f', type=click.Choice(['vtf', 'vtk', 'vtu', 'pvd', 'nc']), required=False, help='Output format.')

# Options that are forwarded to config
@tracked_option('--periodic/--no-periodic', help='Hint that the data may be periodic.', default=False)
@tracked_option('--basis', '-b', 'only_bases', multiple=True, help='Include fields in this basis.')
@tracked_option('--geometry', '-g', 'geometry_basis', default=None, help='Use this basis to provide geometry.')
@tracked_option('--nvis', '-n', 'nvis', default=1, help='Extra sampling points per element.')
@tracked_option('--last', 'only_final_timestep', is_flag=True, help='Read only the last step.')
@tracked_option('--endianness', 'input_endianness', type=click.Choice(['native', 'little', 'big']), default='native')
@tracked_option('--mode', '-m', 'output_mode', type=click.Choice(['binary', 'ascii', 'appended']),
                default='binary', help='Output mode.')

@tracked_option('--no-fields', 'field_filter', is_flag=True, flag_value=())
@tracked_option('--filter', '-l', 'field_filter', multiple=True, help='List of fields to include.')

@tracked_option('--volumetric', 'volumetric', flag_value='volumetric', help='Only include volumetric fields.', default=True)
@tracked_option('--planar', 'volumetric', flag_value='planar', help='Only include planar (surface) fields.')
@tracked_option('--extrude', 'volumetric', flag_value='extrude', help='Extrude planar (surface) fields.')

@tracked_option('--local', 'mapping', flag_value='local', help='Local (cartesian) mapping.', default=True)
@tracked_option('--global', 'mapping', flag_value='global', help='Global (spherical) mapping.')

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

@click.pass_context

@suppress_warnings
def convert(ctx, verbosity, rich, infile, fmt, outfile, **kwargs):
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

    # Handle default values of multi-valued options that should be
    # distinguished from empty, as well as comma splitting
    for k in ['field_filter', 'only_bases']:
        kwargs[k] = tuple(split_commas(kwargs[k]))
    explicit_options = getattr(ctx, 'explicit_options', set())
    if 'field_filter' not in explicit_options:
        kwargs['field_filter'] = None

    try:
        # The config can influence the choice of readers or writers,
        # so apply it first.  Since kwargs may include options that
        # are not explicity set by the user, we set the source to
        # Default, and later use the upgrade_source method.
        with config(source=ConfigSource.Default, **kwargs):
            for option in explicit_options:
                config.upgrade_source(option, ConfigSource.User)
            ReaderClass = Reader.find_applicable(infile)
            WriterClass = Writer.find_applicable(fmt)
            with ReaderClass(infile) as reader, WriterClass(outfile) as writer:
                reader.validate()
                writer.validate()
                pipeline(reader, writer)

    except Exception as e:
        if verbosity == 'debug':
            # In debug mode, allow exceptions to filter through in raw form
            traceback.print_exc()
        else:
            log.error(f"Error: {e}")
            log.error("More information may be available with --debug")
        sys.exit(1)


def deprecated():
    print("ifem-to-vt is deprecated, please launch with 'siso'\n\n", file=sys.stderr)
    convert()


if __name__ == '__main__':
    convert()
