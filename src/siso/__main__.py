from __future__ import annotations

import enum
import logging
import sys
import traceback
from functools import partial, wraps
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import click
from click_option_group import MutuallyExclusiveOptionGroup, optgroup
from rich.console import Console
from rich.logging import RichHandler

from . import api, coord, filter, util
from .api import CoordinateSystem, Dimensionality, Endianness, Rationality, ReaderSettings, Source, Staggering
from .instrument import Instrumenter
from .multisource import MultiSource
from .reader import FindReaderSettings, find_reader
from .writer import OutputFormat, find_writer
from .writer.api import OutputMode, WriterSettings

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def coord_callback(
    ctx: click.Context,
    param: click.Parameter,
    value: Any,
    constructor: Callable[[Any], CoordinateSystem],
) -> CoordinateSystem | None:
    """Callback used for converting a CLI argument to a coordinate system and
    assigning it to the 'out_coords' parameter being passed to the main
    function.

    This is used for the multiple different options for specifying out_coords in
    slightly different ways.
    """
    if not value:
        return None
    system = constructor(value)
    ctx.params["out_coords"] = system
    return system


def defaults(**def_kwargs: Any) -> Callable[[Callable], Callable]:
    """Utility decorator for assigning default values to the main function after
    the CLI arguments have been processed but right before execution.

    This is used to provide defaults for parameters which we can't do with
    click, for whatever reason.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def inner(**in_kwargs: Any) -> Any:
            for k, v in def_kwargs.items():
                if in_kwargs.get(k) is None:
                    in_kwargs[k] = v
            return func(**in_kwargs)

        return inner

    return decorator


def catch(func: Callable) -> Callable:
    @wraps(func)
    def inner(**kwargs: Any) -> Any:
        try:
            return func(**kwargs)
        except api.BadInput as e:
            logging.error(f"Bad input: {e.show()}")
            sys.exit(5)
        except api.Unexpected as e:
            logging.error(f"Unexpected: {e.show()}")
            sys.exit(6)
        except api.Unsupported as e:
            logging.error(f"Unsupported: {e.show()}")
            logging.debug(traceback.format_exc())
            sys.exit(7)

    return inner


class Enum[E: enum.Enum](click.Choice):
    """Parameter type for selecting one choice from an enum."""

    _enum: type[E]

    def __init__(self, enum: type[E], case_sensitive: bool = False):
        self._enum = enum
        super().__init__(choices=[item.value for item in enum], case_sensitive=case_sensitive)

    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> E:
        name = super().convert(value, param, ctx)
        return self._enum(name)


class SliceType(click.ParamType):
    """Parameter type for parsing a range using Python syntax."""

    name = "[START:]STOP[:STEP]"

    def convert(
        self, value: Any, param: click.Parameter | None, ctx: click.Context | None
    ) -> tuple[int | None, ...] | None:
        if value is None or isinstance(value, tuple):
            return value
        try:
            args = value.split(":")
            assert 1 <= len(args) <= 3
            return tuple(int(arg) if arg else None for arg in args)
        except (AssertionError, ValueError):
            self.fail(f"{value!r} is not valid slice syntax", param, ctx)


def find_source(inpath: Sequence[Path], settings: FindReaderSettings) -> Source:
    """Construct a suitable source object from one or more input paths."""
    if len(inpath) == 1:
        # If there's only one input file, find a suitable reader for it.
        source = find_reader(inpath[0], settings)
        if not source:
            logging.critical(f"Unable to determine type of {inpath[0]}")
            sys.exit(2)
        return source

    # If there's multiple input files, find readers for each and bundle them
    # in a MultiSource object.
    sources: list[Source] = []
    for path in inpath:
        source = find_reader(path, settings)
        if source is None:
            logging.critical(f"Unable to determine type of {path}")
            sys.exit(2)
        sources.append(source)
    return MultiSource(sources)


# Main entry-point for the Siso application.
#
# We use click to parse CLI arguments. Each option should be part of a group.
# Some groups are mutually exclusive, others are just based on topic.
@click.command(
    name="Siso",
    help="Convert between various scientific data formats.",
)
# Output options
@optgroup.group("Output")
@optgroup.option(
    "-o",
    "outpath",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
    help="Path of the file or directory to output to.",
)
@optgroup.option(
    "--fmt",
    "-f",
    type=Enum(OutputFormat),
    help=(
        "Format of output. "
        "If this is not provided, will be derived from the file name of the output if provided. "
        "Default is PVD."
    ),
)
@optgroup.option(
    "--mode",
    "-m",
    "output_mode",
    type=Enum(OutputMode),
    help="Mode of output, for those formats which support them.",
)
# Output coordinate systems
#
# It's not possible to assign all of these options to the 'out_coords'
# parameter. They would get in each other's way. Instead we use
# expose_value=False to hide the parsed value from the main function, and
# instead assign the value to the correct parameter using a callback function.
@optgroup.group("Output coordinate systems", cls=MutuallyExclusiveOptionGroup)
# This is the most general option
@optgroup.option(
    "--out-coords",
    "--coords",
    expose_value=False,
    callback=partial(coord_callback, constructor=lambda x: coord.find_system(x)),
    help=(
        "Coordinate system of output. "
        "For simpler usage, use one of the quick options below instead. "
        "Common values are 'geocentric', 'geodetic' (WGS84), 'geodetic:sphere', "
        "'utm:33n' for UTM coordinates in a specific zone (with latitude band), or "
        "'utm:33north' or 'utm:33south' for UTM coordinates in a zone restricted to a hemisphere. "
        "Note: 'utm:33s' will be interpreted as zone 33S, which is north of the equator."
    ),
)
# Each of these options are quick and easy versions for various more specific
# coordinate systems.
@optgroup.option(
    "--geocentric",
    expose_value=False,
    is_flag=True,
    callback=partial(coord_callback, constructor=lambda x: coord.Geocentric()),
    help="Quick option for geocentric output coordinates. Equivalent to '--coords geocentric'.",
)
@optgroup.option(
    "--geodetic",
    expose_value=False,
    type=click.Choice(["WGS84", "GRS80", "WGS72", "sphere"], case_sensitive=False),
    callback=partial(coord_callback, constructor=lambda x: coord.Geodetic.make((x,))),
    help="Quick option for geodetic longitude and latitude output coordinates with specific datum.",
)
@optgroup.option(
    "--wgs84",
    expose_value=False,
    is_flag=True,
    callback=partial(coord_callback, constructor=lambda x: coord.Geodetic(coord.Wgs84())),
    help="Quick option for geodetic longitude and latitude output coordinates with WGS84 reference geoid.",
)
@optgroup.option(
    "--utm",
    expose_value=False,
    type=click.Tuple([click.IntRange(1, 60), click.Choice(["north", "south"], case_sensitive=False)]),
    callback=partial(coord_callback, constructor=lambda x: coord.Utm(x[0], x[1] == "north")),
    help="Quick option for UTM output coordinates with zone number and hemisphere.",
    metavar="ZONE [north|south]",
)
# Input coordinate systems
@optgroup.group("Input coordinate systems")
@optgroup.option(
    "--in-coords",
    default=None,
    help=(
        "Specify which input coordinate system(s) to use, "
        "if there are multiple that can convert to the provided output coordinate system."
    ),
)
# Time slicing
@optgroup.group("Time slicing", cls=MutuallyExclusiveOptionGroup)
@optgroup.option(
    "--times",
    "timestep_slice",
    default=None,
    type=SliceType(),
    help="Specify a subset of timesteps to extract. Slices are closed on the left and open on the right.",
)
@optgroup.option(
    "--time",
    "timestep_index",
    default=None,
    type=int,
    help="Specify a specific timestep to extract (zero-indexed).",
)
@optgroup.option(
    "--last",
    "only_final_timestep",
    is_flag=True,
    help="Only extract the last timestep.",
)
# Field filtering
@optgroup.group("Field filtering", cls=MutuallyExclusiveOptionGroup)
@optgroup.option(
    "--no-fields",
    is_flag=True,
    help="Don't extract any fields, only the geometry.",
)
@optgroup.option(
    "--filter",
    "-l",
    "field_filter",
    multiple=True,
    default=None,
    help=(
        "Specify which fields to extract. "
        "This option can be provided multiple times, or you can supply a comma-separated list of field names."
    ),
    metavar="NAME[,NAME]*",
)
@optgroup.option(
    "--exclude",
    "field_exclude",
    multiple=True,
    default=None,
    help=(
        "Specify which fields to exclude. "
        "This option can be provided multiple times, or you can supply a comma-separated list of field names."
    ),
    metavar="NAME[,NAME]*",
)
# Endianness
@optgroup.group("Endianness")
@optgroup.option(
    "--in-endianness",
    type=Enum(Endianness),
    default="native",
    help=(
        "Override the assumed endianness of the input. Useful for raw data dump formats with little metadata."
    ),
)
@optgroup.option(
    "--out-endianness",
    type=Enum(Endianness),
    default="native",
    help="Override the endianness of the output.",
)
# Reader options
@optgroup.group("Input processing")
@optgroup.option("--staggering", type=Enum(Staggering), default="inner")
@optgroup.option(
    "--periodic",
    is_flag=True,
    help="Stitch together periodic geometries.",
)
@optgroup.option(
    "--nvis",
    "-n",
    default=1,
    help="Number of subdivisions to use when sampling superlinear geometries.",
)
# Dimensionality
@optgroup.group("Dimensionality", cls=MutuallyExclusiveOptionGroup)
@optgroup.option(
    "--volumetric",
    "dimensionality",
    flag_value=Dimensionality.Volumetric,
    default=True,
    type=click.UNPROCESSED,
    help="Extract volumetric data and fields only. (Default.)",
)
@optgroup.option(
    "--planar",
    "dimensionality",
    flag_value=Dimensionality.Planar,
    type=click.UNPROCESSED,
    help="Extract planar data and fields only.",
)
@optgroup.option(
    "--extrude",
    "dimensionality",
    flag_value=Dimensionality.Extrude,
    type=click.UNPROCESSED,
    help="Extract volumetric data, and extrude planar data so that it becomes volumetric.",
)
# Rationality
@optgroup.group("Rationality")
@optgroup.option(
    "--rational",
    "rationality",
    flag_value=Rationality.Always,
    type=click.UNPROCESSED,
    help="Assume ambiguous spline objects are always rational.",
)
@optgroup.option(
    "--non-rational",
    "rationality",
    flag_value=Rationality.Never,
    type=click.UNPROCESSED,
    help="Assume ambiguous spline objects are never rational.",
)
# Miscellaneous options
@optgroup.group("Miscellaneous")
@optgroup.option(
    "--unstructured",
    "require_unstructured",
    is_flag=True,
    help="Force output of unstructured grids, even if the output format supports structured.",
)
@optgroup.option(
    "--decompose/--no-decompose",
    default=True,
    help="Decompose vector fields into scalar components.",
)
@optgroup.option(
    "--eigenmodes-are-displacement",
    "--ead",
    "eigenmodes_are_displacement",
    is_flag=True,
    help="Interpret eigenmodes as displacement fields.",
)
@optgroup.option(
    "--mesh",
    "mesh_filename",
    type=click.Path(exists=True, file_okay=True, readable=True, path_type=Path),
    help="Override path to mesh file, for input formats where data and mesh are separate.",
)
@optgroup.option(
    "--basis",
    "-b",
    "basis_filter",
    multiple=True,
    default=None,
    metavar="NAME[,NAME]*",
    help=(
        "Specify which bases to extract. "
        "This option can be provided multiple times, or you can supply a comma-separated list of basis names."
    ),
)
# Verbosity
@optgroup.group("Verbosity", cls=MutuallyExclusiveOptionGroup)
@optgroup.option("--debug", "verbosity", flag_value="debug", help="Print debug messages.")
@optgroup.option("--info", "verbosity", flag_value="info", default=True, help="Print normal information.")
@optgroup.option("--warning", "verbosity", flag_value="warning", help="Only print warnings or errors.")
@optgroup.option("--error", "verbosity", flag_value="error", help="Only print errors.")
@optgroup.option("--critical", "verbosity", flag_value="critical", help="Only print critical errors.")
# Colors
@optgroup.group("Log formatting")
@optgroup.option("--rich/--no-rich", default=True, help="Use rich output formatting.")
# Debugging
@optgroup.group("Debugging")
@optgroup.option("--verify-strict", is_flag=True, help="Add extra assertions for debugging purposes.")
@optgroup.option("--instrument", is_flag=True, help="Add instrumentation for profiling purposes.")
# Input
@click.argument(
    "inpath",
    nargs=-1,
    type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True, path_type=Path),
    required=True,
    metavar="INPUT...",
)
# Final step befor execution: apply defaults that are not provided by the click machinery.
@defaults(
    out_coords=coord.Generic(),
)
# Catch Siso errors and log them to stdout.
@catch

# Main entry-point
def main(
    # Pipeline options
    require_unstructured: bool,
    decompose: bool,
    periodic: bool,
    eigenmodes_are_displacement: bool,
    out_coords: CoordinateSystem,
    in_coords: str | None,
    timestep_slice: tuple[int | None],
    timestep_index: int | None,
    only_final_timestep: bool,
    nvis: int,
    no_fields: bool,
    field_filter: tuple[str],
    field_exclude: tuple[str],
    # Writer options
    output_mode: OutputMode | None,
    out_endianness: Endianness,
    # Reader options
    in_endianness: Endianness,
    dimensionality: Dimensionality,
    staggering: Staggering,
    rationality: Rationality | None,
    mesh_filename: Path | None,
    basis_filter: tuple[str],
    # Logging, verbosity and testing
    verify_strict: bool,
    instrument: bool,
    verbosity: str,
    rich: bool,
    # Input and output
    inpath: tuple[Path, ...],
    outpath: Path | None,
    fmt: OutputFormat | None,
) -> None:
    # Configure logging
    color_system: Literal["auto"] | None = "auto" if rich else None
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
        # Output path is given but format is not: use the output path suffix to
        # determine the format.
        suffix = outpath.suffix[1:].casefold()
        if suffix == "dat":
            logging.warning("Interpreting .dat filetype as SIMRA Mesh file")
            logging.warning("Note: the .dat extension is overloaded - don't rely on this behavior")
            logging.warning("Prefer using '-f simra'")
            fmt = OutputFormat.Simra
        else:
            fmt = OutputFormat(suffix)
    elif not outpath:
        # Output path is not given. Set format to PVD if not explicitly
        # provided, and use it to create a default output path.
        fmt = fmt or OutputFormat.Pvd
        outpath = Path(inpath[0].name).with_suffix(fmt.default_suffix())

    # Hint to the type checker that these are non-null
    assert fmt
    assert outpath

    # Construct source object for reading input
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

    # Construct sink object for writing output
    sink = find_writer(fmt, outpath)
    if not sink:
        sys.exit(3)
    sink.configure(
        WriterSettings(
            output_mode=output_mode,
            endianness=out_endianness,
        )
    )

    # Apply filters according to the CLI options provided, as well as the known
    # properties of the source and the sink. This produces a 'stack' of source
    # objects, the bottom reading from the input on disk, and each successive
    # one transforming the data from the stage below in some way.
    #
    # The sequence of filters can be crucial! Please read the documentation in
    # siso.api for some of the terms used.
    #
    # In some cases the properties of the source are not fully known before
    # opening the file, thus we trigger the __enter__ method already here.
    with source:
        out_props = sink.properties

        # Debugging: if strict verification is required, add the filter as early
        # as possible.
        if verify_strict:
            source = filter.Strict(source)
            logging.debug("Attaching Strict (--verify-strict)")

        # If the source does not have zones with global keys, insert the
        # KeyZones filter. Many operations later require this.
        if not source.properties.globally_keyed:
            logging.debug("Attaching KeyZones (source is not globally keyed)")
            source = filter.KeyZones(source)

        # Filter out irrelevant bases if the --basis option is provided.
        if basis_filter:
            logging.debug("Attaching BasisFilter (--basis)")
            allowed_bases = set(
                chain.from_iterable(map(str.casefold, basis_name.split(",")) for basis_name in basis_filter)
            )
            source = filter.BasisFilter(source, allowed_bases)

        # Discretizing superlinear geometries happens before basis merging. This
        # is because basis merging currently discretizes geometries anyway.
        # There's no reason why it MUST do so. If we fix the BasisMerge filter
        # so that it preserves superlinearity of bases, we should remove this
        # step, and instead apply the nvis parameter after basis merging.
        if nvis > 1:
            logging.debug("Attaching Discretize (--nvis)")
            source = filter.Discretize(source, nvis)

        # If the sink can only handle a single basis, we must merge the bases
        # into one. This step also has the side-effect of discretizing the
        # geometry, which is why it must come after handling of nvis.
        if out_props.require_single_basis and not source.properties.single_basis:
            logging.debug("Attaching BasisMerge (sink requires single basis)")
            source = filter.BasisMerge(source)

        # Apply discretization for any number of other reasons except nvis,
        # which has been handled earlier in the pipeline.
        if not source.properties.discrete_topology:
            if out_props.require_discrete_topology:
                logging.debug("Attaching Discretize (sink requires discrete)")
                source = filter.Discretize(source, 1)
            elif out_props.require_single_zone:
                # The zone merge filter only works for discrete bases, so apply
                # it automatically.
                logging.debug("Attaching Discretize (sink requires single zone)")
                source = filter.Discretize(source, 1)
            elif require_unstructured:
                logging.debug("Attaching Discretize (--unstructured)")
                source = filter.Discretize(source, 1)

        # If the sink cannot handle more than one zone, apply the zone merge filter.
        if not source.properties.single_zoned and out_props.require_single_zone:
            logging.debug("Attaching ZoneMerge (sink requires single zone)")
            source = filter.ZoneMerge(source)  # type: ignore[arg-type]

        # If the source recommends splitting some fields, do that now.
        if source.properties.split_fields:
            logging.debug("Attaching Split (source recommendation)")
            source = filter.Split(source, source.properties.split_fields)

        # If the source recommends recombining some fields, do that now.
        if source.properties.recombine_fields:
            logging.debug("Attaching Recombine (source recommendation)")
            source = filter.Recombine(source, source.properties.recombine_fields)

        # Decompose vector fields into their scalar components.
        if decompose:
            logging.debug("Attaching Decompose (--decompose)")
            source = filter.Decompose(source)

        # Force structured topologies to become unstructured. This requires
        # discrete topologies.
        if require_unstructured:
            logging.debug("Attaching ForceUnstructured (--unstructured)")
            source = filter.ForceUnstructured(source)  # type: ignore[arg-type]

        # Convert eigenmode fields to displacements.
        if eigenmodes_are_displacement:
            logging.debug("Attaching EigenDisp (--eigenmodes-are-displacement)")
            source = filter.EigenDisp(source)

        # Apply timestep slicing if necessary.
        if timestep_slice is not None:
            logging.debug("Attaching StepSlice (--times)")
            source = filter.StepSlice(source, timestep_slice)
        elif timestep_index is not None:
            logging.debug("Attaching StepSlice (--time)")
            source = filter.StepSlice(
                source,
                (timestep_index, timestep_index + 1),
                explicit_instantaneous=True,
            )
        elif only_final_timestep:
            logging.debug("Attaching LastTime (--last)")
            source = filter.LastTime(source)

        # At this point, abort execution if the source has (or may have)
        # multiple steps, but the output requires only one.
        assert not (out_props.require_instantaneous and not source.properties.instantaneous)

        # Apply field filtering if necessary.
        if no_fields:
            logging.debug("Attaching FieldFilter (--no-fields)")
            source = filter.FieldFilter(source, None, "all")
        elif field_filter or field_exclude:
            logging.debug("Attaching FieldFilter (--filter, --exclude)")
            allowed_fields = (
                set(
                    chain.from_iterable(
                        map(str.casefold, field_name.split(",")) for field_name in field_filter
                    )
                )
                if field_filter
                else None
            )
            excluded_fields = (
                set(
                    chain.from_iterable(
                        map(str.casefold, field_name.split(",")) for field_name in field_exclude
                    )
                )
                if field_exclude
                else None
            )
            source = filter.FieldFilter(source, allowed_fields, excluded_fields)

        # Apply another strict verification filter.
        if verify_strict:
            logging.debug("Attaching Strict (--verify-strict)")
            source = filter.Strict(source)

        # Print the names of all the discovered fields to the debug log
        for basis in source.bases():
            for field in source.fields(basis):
                logging.debug(
                    f"Discovered field '{field.name}' with "
                    f"{util.pluralize(field.num_comps, 'component', 'components')} "
                    f"(basis '{basis.name}')"
                )

        # Choose a geometry to use. This requires filtering based on the
        # coordinate system options. First, get a list of all available geometry
        # fields.
        geometries = [geometry for basis in source.bases() for geometry in source.geometries(basis)]

        # The --in-coords option filters the geometry list, so apply that here.
        if in_coords:
            geometries = [geometry for geometry in geometries if geometry.fits_system_name(in_coords)]
            names = ", ".join(f"'{geometry.name}'" for geometry in geometries)
            logging.debug(f"Retaining {names}")

        # At this point, find the cheapest coordinate conversion path from a
        # source geometry to the output coordinate system.
        result = coord.optimal_system([geometry.coords for geometry in geometries], out_coords)
        if result is None:
            logging.critical(f"Unable to determine a coordinate system conversion path to {out_coords}")
            logging.critical("These source coordinate systems were considered:")
            for geometry in geometries:
                logging.critical(f"- {geometry.coords} (field '{geometry.name}')")
            sys.exit(4)

        # Pick a geometry field and notify the source stack that we are about to use it.
        i, path = result
        geometry = geometries[i]
        logging.info(f"Using '{geometry.name}' as geometry")
        source.use_geometry(geometry)

        # If the coordinate conversion path is nontrivial, apply it now.
        if path:
            logging.debug("Coordinate conversion path:")
            str_path = " -> ".join(str(system) for system in path)
            logging.debug(str_path)
            logging.debug("Attaching CoordTransform")
            source = filter.CoordTransform(source, path)

        # Debugging option: apply an instrumentation object for profiling if required.
        instrumenter: Instrumenter | None = None
        if instrument:
            instrumenter = Instrumenter(source)

        # Open the sink object and ask it to consume data from the source stack.
        with sink:
            sink.consume(source, geometry)

        # Print instrumentation data.
        if instrument:
            assert instrumenter
            instrumenter.report()
