import logging
from pathlib import Path
import sys

import click
from rich.logging import RichHandler

from .api import Source
from .field import Field, FieldType
from .multisource import MultiSource
from .reader import find_reader
from . import filters, util

from .writer import OutputFormat, find_writer
from .writer.api import OutputMode, WriterSettings

from typing import Optional, Tuple, List, Sequence


class Enum(click.Choice):
    def __init__(self, enum, case_sensitive: bool = False):
        self._enum = enum
        super().__init__(choices=[item.value for item in enum], case_sensitive=case_sensitive)

    def convert(self, value, param, ctx):
        name = super().convert(value, param, ctx)
        return self._enum(name)


def get_source(inpath: Sequence[Path]) -> Source:
    if len(inpath) == 1:
        source = find_reader(inpath[0])
        if not source:
            logging.critical(f'Unable to determine type of {inpath[0]}')
            sys.exit(2)
        return source
    else:
        sources: List[Source] = []
        for path in inpath:
            source = find_reader(path)
            if source is None:
                logging.critical(f'Unable to determine type of {path}')
                sys.exit(2)
            sources.append(source)
        return MultiSource(sources)


@click.command()

@click.option('--mode', '-m', 'output_mode', type=Enum(OutputMode))
@click.option('--unstructured', 'require_unstructured', is_flag=True)

# Logging and verbosity
@click.option('--debug', 'verbosity', flag_value='debug')
@click.option('--info', 'verbosity', flag_value='info', default=True)
@click.option('--warning', 'verbosity', flag_value='warning')
@click.option('--error', 'verbosity', flag_value='error')
@click.option('--critical', 'verbosity', flag_value='critical')
@click.option('--rich/--no-rich', default=True)

# Input and output
@click.option('-o', 'outpath', type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path))
@click.option('--fmt', '-f', type=Enum(OutputFormat))
@click.argument('inpath', nargs=-1, type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True, path_type=Path))

def main(
    verbosity: str,
    rich: bool,

    output_mode: Optional[OutputMode],
    require_unstructured: bool,

    inpath: Tuple[Path, ...],
    outpath: Optional[Path],
    fmt: Optional[OutputFormat],
):
    # Configure logging
    if rich:
        handlers = [RichHandler(show_path=False)]
        log_format = '{message}'
    else:
        handlers = None
        log_format = '{asctime} {levelname: <8} {message}'
    logging.basicConfig(
        level=verbosity.upper(),
        style='{',
        format=log_format,
        datefmt='[%X]',
        handlers=handlers
    )

    # Assert that there are inputs
    if not inpath:
        logging.critical('No inputs given')
        sys.exit(1)
    assert len(inpath) > 0

    # Resolve potential mismatches between output and format
    if outpath and not fmt:
        fmt = OutputFormat(outpath.suffix[1:].lower())
    elif not outpath:
        fmt = fmt or OutputFormat.Pvd
        outpath = Path(inpath[0].name).with_suffix(fmt.default_suffix())
    assert fmt
    assert outpath

    # Construct source and sink objects
    source = get_source(inpath)
    if not source:
        sys.exit(2)

    sink = find_writer(fmt, outpath)
    if not sink:
        sys.exit(3)

    with source:
        sink.configure(WriterSettings(
            output_mode=output_mode,
        ))

        if not source.properties.globally_keyed:
            source = filters.KeyZones(source)

        if not source.properties.tesselated:
            if sink.properties.require_tesselated or sink.properties.require_single_zone or require_unstructured:
                source = filters.Tesselate(source)

        if not source.properties.single_zoned:
            if sink.properties.require_single_zone:
                source = filters.ZoneMerge(source)

        if source.properties.split_fields:
            pass

        if source.properties.recombine_fields:
            pass

        if require_unstructured:
            source = filters.ForceUnstructured(source)

        geometries: List[Field] = []
        fields: List[Field] = []
        for field in source.fields():
            if field.type == FieldType.Geometry:
                geometries.append(field)
            else:
                fields.append(field)

        for field in fields:
            logging.debug(
                f"Discovered field '{field.name}' with "
                f"{util.pluralize(field.ncomps, 'component', 'components')}"
            )
        logging.debug(f'Using {geometries[0].name} as geometry')

        with sink:
            sink.consume(source, geometries[0], fields)
