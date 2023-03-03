from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, cast

import click
from rich.console import Console
from rich.logging import RichHandler

from . import coords, filters, util
from .api import CoordinateSystem, Dimensionality, Endianness, Field, ReaderSettings, Source, Staggering
from .multisource import MultiSource
from .reader import FindReaderSettings, find_reader
from .writer import OutputFormat, find_writer
from .writer.api import OutputMode, WriterSettings


class Enum(click.Choice):
    def __init__(self, enum, case_sensitive: bool = False):
        self._enum = enum
        super().__init__(choices=[item.value for item in enum], case_sensitive=case_sensitive)

    def convert(self, value, param, ctx):
        name = super().convert(value, param, ctx)
        return self._enum(name)


class Coords(click.ParamType):
    name = "coords"

    def convert(self, value, param, ctx):
        if value is None or value is False:
            return None
        if isinstance(value, CoordinateSystem):
            return value
        return coords.find_system(value)


def find_source(inpath: Sequence[Path], settings: FindReaderSettings) -> Source:
    if len(inpath) == 1:
        source = find_reader(inpath[0], settings)
        if not source:
            logging.critical(f"Unable to determine type of {inpath[0]}")
            sys.exit(2)
        return source
    else:
        sources: List[Source] = []
        for path in inpath:
            source = find_reader(path, settings)
            if source is None:
                logging.critical(f"Unable to determine type of {path}")
                sys.exit(2)
            sources.append(source)
        return MultiSource(sources)


@click.command()

# Pipeline options
@click.option("--unstructured", "require_unstructured", is_flag=True)
@click.option("--decompose/--no-decompose", default=True)
@click.option("--periodic", is_flag=True)
@click.option("--eigenmodes-are-displacement", "--ead", "eigenmodes_are_displacement", is_flag=True)
@click.option("--out-coords", default=coords.Generic(), type=Coords())
@click.option("--coords", "out_coords", default=coords.Generic(), type=Coords())
@click.option("--in-coords", default=None)

# Writer options
@click.option("--mode", "-m", "output_mode", type=Enum(OutputMode))

# Reader options
@click.option("--in-endianness", type=Enum(Endianness), default="native")
@click.option(
    "--volumetric",
    "dimensionality",
    flag_value=Dimensionality.Volumetric,
    default=True,
    type=click.UNPROCESSED,
)
@click.option("--planar", "dimensionality", flag_value=Dimensionality.Planar, type=click.UNPROCESSED)
@click.option("--extrude", "dimensionality", flag_value=Dimensionality.Extrude, type=click.UNPROCESSED)
@click.option("--staggering", type=Enum(Staggering), default="inner")

# Logging, verbosity and testing
@click.option("--verify-strict/--no-verify-strict", default=False)
@click.option("--debug", "verbosity", flag_value="debug")
@click.option("--info", "verbosity", flag_value="info", default=True)
@click.option("--warning", "verbosity", flag_value="warning")
@click.option("--error", "verbosity", flag_value="error")
@click.option("--critical", "verbosity", flag_value="critical")
@click.option("--rich/--no-rich", default=True)

# Input and output
@click.option("-o", "outpath", type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path))
@click.option("--fmt", "-f", type=Enum(OutputFormat))
@click.argument(
    "inpath",
    nargs=-1,
    type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True, path_type=Path),
)
def main(
    # Pipeline options
    require_unstructured: bool,
    decompose: bool,
    periodic: bool,
    eigenmodes_are_displacement: bool,
    out_coords: CoordinateSystem,
    in_coords: Optional[str],
    # Writer options
    output_mode: Optional[OutputMode],
    # Reader options
    in_endianness: Endianness,
    dimensionality: Dimensionality,
    staggering: Staggering,
    # Logging, verbosity and testing
    verify_strict: bool,
    verbosity: str,
    rich: bool,
    # Input and output
    inpath: Tuple[Path, ...],
    outpath: Optional[Path],
    fmt: Optional[OutputFormat],
) -> None:
    # Configure logging
    color_system: Optional[Literal["auto"]] = "auto" if rich else None
    logging.basicConfig(
        level=verbosity.upper(),
        style="{",
        format="{message}",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                show_path=False,
                console=Console(color_system=color_system),
            )
        ],
    )

    # Assert that there are inputs
    if not inpath:
        logging.critical("No inputs given")
        sys.exit(1)
    assert len(inpath) > 0

    # Resolve potential mismatches between output and format
    if outpath and not fmt:
        fmt = OutputFormat(outpath.suffix[1:].casefold())
    elif not outpath:
        fmt = fmt or OutputFormat.Pvd
        outpath = Path(inpath[0].name).with_suffix(fmt.default_suffix())
    assert fmt
    assert outpath

    # Construct source and sink objects
    source = find_source(
        inpath,
        FindReaderSettings(
            endianness=in_endianness,
        ),
    )
    if not source:
        sys.exit(2)
    source.configure(
        ReaderSettings(
            endianness=in_endianness,
            dimensionality=dimensionality,
            staggering=staggering,
            periodic=periodic,
        )
    )

    sink = find_writer(fmt, outpath)
    if not sink:
        sys.exit(3)
    sink.configure(
        WriterSettings(
            output_mode=output_mode,
        )
    )

    with source:
        in_props = source.properties
        out_props = sink.properties

        if verify_strict:
            source = filters.Strict(source)

        if not in_props.globally_keyed:
            source = filters.KeyZones(source)

        if not in_props.tesselated:
            if out_props.require_tesselated or out_props.require_single_zone or require_unstructured:
                source = filters.Tesselate(source)

        if not in_props.single_zoned:
            if out_props.require_single_zone:
                source = filters.ZoneMerge(source)

        if in_props.split_fields:
            source = filters.Split(source, in_props.split_fields)

        if in_props.recombine_fields:
            source = filters.Recombine(source, in_props.recombine_fields)

        if decompose:
            source = filters.Decompose(source)

        if require_unstructured:
            source = filters.ForceUnstructured(source)

        if eigenmodes_are_displacement:
            source = filters.EigenDisp(source)

        if verify_strict:
            source = filters.Strict(source)

        geometries: List[Field] = []
        fields: List[Field] = []
        for field in source.fields():
            if field.is_geometry:
                geometries.append(field)
            else:
                fields.append(field)

        for field in fields:
            logging.debug(
                f"Discovered field '{field.name}' with "
                f"{util.pluralize(field.ncomps, 'component', 'components')}"
            )

        for geometry in geometries:
            logging.debug(f"Discovered geometry '{geometry.name}' with coordinate system {geometry.coords}")

        if in_coords:
            geometries = [geometry for geometry in geometries if geometry.fits_system_name(in_coords)]
            names = ", ".join(f"'{geometry.name}'" for geometry in geometries)
            logging.debug(f"Retaining {names}")

        result = coords.optimal_system([geometry.coords for geometry in geometries], out_coords)
        if result is None:
            logging.critical(f"Unable to determine a coordinate system conversion path to {out_coords}")
            logging.critical("These source coordinate systems were considered:")
            for geometry in geometries:
                logging.critical(f"- {geometry.coords} (field '{geometry.name}')")
            sys.exit(3)

        i, path = result
        geometry = geometries[i]
        logging.info(f"Using '{geometry.name}' as geometry")
        source.use_geometry(geometry)

        if path:
            logging.debug("Coordinate conversion path:")
            str_path = " -> ".join(str(system) for system in path)
            logging.debug(str_path)
            source = filters.CoordTransform(source, path)

        with sink:
            sink.consume(source, geometry, fields)
