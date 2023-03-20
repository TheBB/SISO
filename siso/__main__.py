from __future__ import annotations

import logging
import sys
from itertools import chain
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple

import click
from click_option_group import MutuallyExclusiveOptionGroup, optgroup
from rich.console import Console
from rich.logging import RichHandler

from . import coord, filter, util
from .api import CoordinateSystem, Dimensionality, Endianness, Rationality, ReaderSettings, Source, Staggering
from .instrument import Instrumenter
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
            return coord.Generic()
        if isinstance(value, CoordinateSystem):
            return value
        return coord.find_system(value)


class SliceType(click.ParamType):
    name = "slice"

    def convert(self, value, param, ctx):
        if value is None or isinstance(value, tuple):
            return value
        try:
            args = value.split(":")
            assert 1 <= len(args) <= 3
            return tuple(int(arg) if arg else None for arg in args)
        except (AssertionError, ValueError):
            self.fail(f"{value!r} is not valid slice syntax", param, ctx)


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

# Output
@optgroup.group("Output")
@optgroup.option(
    "-o", "outpath", type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path)
)
@optgroup.option("--mode", "-m", "output_mode", type=Enum(OutputMode))
@optgroup.option("--fmt", "-f", type=Enum(OutputFormat))

# Output coordinate systems
@optgroup.group("Output coordinate systems", cls=MutuallyExclusiveOptionGroup)
@optgroup.option("--out-coords", "--coords", "out_coords", default=coord.Generic(), type=Coords())
@optgroup.option("--geocentric", "out_coords", flag_value=coord.Geocentric(), type=Coords())

# Input coordinate systems
@optgroup.group("Input coordinate systems")
@optgroup.option("--in-coords", default=None)

# Time slicing
@optgroup.group("Time slicing", cls=MutuallyExclusiveOptionGroup)
@optgroup.option("--times", "timestep_slice", default=None, type=SliceType())
@optgroup.option("--time", "timestep_index", default=None, type=int)
@optgroup.option("--last", "only_final_timestep", is_flag=True)

# Field filtering
@optgroup.group("Field filtering", cls=MutuallyExclusiveOptionGroup)
@optgroup.option("--no-fields", is_flag=True)
@optgroup.option("--filter", "-l", "field_filter", multiple=True, default=None)

# Endianness
@optgroup.group("Endianness")
@optgroup.option("--in-endianness", type=Enum(Endianness), default="native")
@optgroup.option("--out-endianness", type=Enum(Endianness), default="native")

# Reader options
@optgroup.group("Input processing")
@optgroup.option("--staggering", type=Enum(Staggering), default="inner")
@optgroup.option("--periodic", is_flag=True)
@optgroup.option("--nvis", "-n", default=1)

# Dimensionality
@optgroup.group("Dimensionality", cls=MutuallyExclusiveOptionGroup)
@optgroup.option(
    "--volumetric",
    "dimensionality",
    flag_value=Dimensionality.Volumetric,
    default=True,
    type=click.UNPROCESSED,
)
@optgroup.option("--planar", "dimensionality", flag_value=Dimensionality.Planar, type=click.UNPROCESSED)
@optgroup.option("--extrude", "dimensionality", flag_value=Dimensionality.Extrude, type=click.UNPROCESSED)

# Rationality
@optgroup.group("Rationality")
@optgroup.option("--rational", "rationality", flag_value=Rationality.Always, type=click.UNPROCESSED)
@optgroup.option("--non-rational", "rationality", flag_value=Rationality.Never, type=click.UNPROCESSED)

# Miscellaneous options
@optgroup.group("Miscellaneous")
@optgroup.option("--unstructured", "require_unstructured", is_flag=True)
@optgroup.option("--decompose/--no-decompose", default=True)
@optgroup.option("--eigenmodes-are-displacement", "--ead", "eigenmodes_are_displacement", is_flag=True)
@optgroup.option(
    "--mesh",
    "mesh_filename",
    type=click.Path(exists=True, file_okay=True, readable=True, path_type=Path),
)
@optgroup.option("--basis", "-b", "basis_filter", multiple=True, default=None)

# Verbosity options
@optgroup.group("Verbosity", cls=MutuallyExclusiveOptionGroup)
@optgroup.option("--debug", "verbosity", flag_value="debug")
@optgroup.option("--info", "verbosity", flag_value="info", default=True)
@optgroup.option("--warning", "verbosity", flag_value="warning")
@optgroup.option("--error", "verbosity", flag_value="error")
@optgroup.option("--critical", "verbosity", flag_value="critical")

# Colors
@optgroup.group("Log formatting")
@optgroup.option("--rich/--no-rich", default=True)

# Debugging options
@optgroup.group("Debugging")
@optgroup.option("--verify-strict", is_flag=True)
@optgroup.option("--instrument", is_flag=True)

# Input
@click.argument(
    "inpath",
    nargs=-1,
    type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True, path_type=Path),
    required=True,
)
def main(
    # Pipeline options
    require_unstructured: bool,
    decompose: bool,
    periodic: bool,
    eigenmodes_are_displacement: bool,
    out_coords: CoordinateSystem,
    in_coords: Optional[str],
    timestep_slice: Tuple[Optional[int]],
    timestep_index: Optional[int],
    only_final_timestep: bool,
    nvis: int,
    no_fields: bool,
    field_filter: Tuple[str],
    # Writer options
    output_mode: Optional[OutputMode],
    out_endianness: Endianness,
    # Reader options
    in_endianness: Endianness,
    dimensionality: Dimensionality,
    staggering: Staggering,
    rationality: Optional[Rationality],
    mesh_filename: Optional[Path],
    basis_filter: Tuple[str],
    # Logging, verbosity and testing
    verify_strict: bool,
    instrument: bool,
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

    # Resolve potential mismatches between output and format
    if outpath and not fmt:
        suffix = outpath.suffix[1:].casefold()
        if suffix == "dat":
            logging.warning("Interpreting .dat filetype as SIMRA Mesh file")
            logging.warning("Note: the .dat extension is overloaded - don't rely on this behavior")
            logging.warning("Prefer using '-f simra'")
            fmt = OutputFormat.Simra
        else:
            fmt = OutputFormat(suffix)
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
            mesh_filename=mesh_filename,
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
            mesh_filename=mesh_filename,
            rationality=rationality,
        )
    )

    sink = find_writer(fmt, outpath)
    if not sink:
        sys.exit(3)
    sink.configure(
        WriterSettings(
            output_mode=output_mode,
            endianness=out_endianness,
        )
    )

    with source:
        # in_props = source.properties
        out_props = sink.properties

        if verify_strict:
            source = filter.Strict(source)
            logging.debug("Attaching Strict (--verify-strict)")

        if not source.properties.globally_keyed:
            logging.debug("Attaching KeyZones (source is not globally keyed)")
            source = filter.KeyZones(source)

        if basis_filter:
            logging.debug("Attaching BasisFilter (--basis)")
            allowed_bases = set(
                chain.from_iterable(map(str.casefold, basis_name.split(",")) for basis_name in basis_filter)
            )
            source = filter.BasisFilter(source, allowed_bases)

        if nvis > 1:
            logging.debug("Attaching Discretize (--nvis)")
            source = filter.Discretize(source, nvis)

        if out_props.require_single_basis and not source.properties.single_basis:
            logging.debug("Attaching BasisMerge (sink requires single basis)")
            source = filter.BasisMerge(source)

        if not source.properties.discrete_topology:
            if out_props.require_discrete_topology:
                logging.debug("Attaching Discretize (sink requires discrete)")
                source = filter.Discretize(source, 1)
            elif out_props.require_single_zone:
                logging.debug("Attaching Discretize (sink requires single zone)")
                source = filter.Discretize(source, 1)
            elif require_unstructured:
                logging.debug("Attaching Discretize (--unstructured)")
                source = filter.Discretize(source, 1)

        if not source.properties.single_zoned and out_props.require_single_zone:
            logging.debug("Attaching ZoneMerge (sink requires single zone)")
            source = filter.ZoneMerge(source)

        if source.properties.split_fields:
            logging.debug("Attaching Split (source recommendation)")
            source = filter.Split(source, source.properties.split_fields)

        if source.properties.recombine_fields:
            logging.debug("Attaching Recombine (source recommendation)")
            source = filter.Recombine(source, source.properties.recombine_fields)

        if decompose:
            logging.debug("Attaching Decompose (--decompose)")
            source = filter.Decompose(source)

        if require_unstructured:
            logging.debug("Attaching ForceUnstructured (--unstructured)")
            source = filter.ForceUnstructured(source)

        if eigenmodes_are_displacement:
            logging.debug("Attaching EigenDisp (--eigenmodes-are-displacement)")
            source = filter.EigenDisp(source)

        if timestep_slice is not None:
            logging.debug("Attaching StepSlice (--times)")
            source = filter.StepSlice(source, timestep_slice)
        elif timestep_index is not None:
            logging.debug("Attaching StepSlice (--time)")
            source = filter.StepSlice(source, (timestep_index, timestep_index + 1))
        elif only_final_timestep:
            logging.debug("Attaching LastTime (--last)")
            source = filter.LastTime(source)

        if no_fields:
            logging.debug("Attaching FieldFilter (--no-fields)")
            source = filter.FieldFilter(source, set())
        elif field_filter:
            logging.debug("Attaching FieldFilter (--filter)")
            allowed_fields = set(
                chain.from_iterable(map(str.casefold, field_name.split(",")) for field_name in field_filter)
            )
            source = filter.FieldFilter(source, allowed_fields)

        if verify_strict:
            logging.debug("Attaching Strict (--verify-strict)")
            source = filter.Strict(source)

        assert not (out_props.require_instantaneous and not source.properties.instantaneous)

        for basis in source.bases():
            for field in source.fields(basis):
                logging.debug(
                    f"Discovered field '{field.name}' with "
                    f"{util.pluralize(field.ncomps, 'component', 'components')}"
                )

        geometries = [geometry for basis in source.bases() for geometry in source.geometries(basis)]

        if in_coords:
            geometries = [geometry for geometry in geometries if geometry.fits_system_name(in_coords)]
            names = ", ".join(f"'{geometry.name}'" for geometry in geometries)
            logging.debug(f"Retaining {names}")

        result = coord.optimal_system([geometry.coords for geometry in geometries], out_coords)
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
            logging.debug("Attaching CoordTransform")
            source = filter.CoordTransform(source, path)

        instrumenter: Optional[Instrumenter] = None
        if instrument:
            instrumenter = Instrumenter(source)

        with sink:
            sink.consume(source, geometry)

        if instrument:
            assert instrumenter
            instrumenter.report()
