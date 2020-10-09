from operator import attrgetter
import treelog as log

from typing import Iterable, List, Tuple

from . import config
from .filters import Source, LastStepFilter, TesselatorFilter, MergeTopologiesFilter
from .coords import Coords, Converter, graph
from .fields import Field, ComponentField
from .reader import Reader
from .writer import Writer


def discover_decompositions(fields: List[Field]) -> Iterable[Field]:
    for field in fields:
        yield field
        for subfield in field.decompositions():
            log.debug(f"Discovered decomposed scalar field '{subfield.name}'")
            yield subfield


def discover_fields(reader: Reader) -> Tuple[List[Field], List[Field]]:
    geometries, fields = [], []
    for field in reader.fields():
        if field.is_geometry:
            geometries.append(field)
            continue
        if config.field_filter is not None and field.name not in config.field_filter:
            continue
        fields.append(field)

    for field in fields:
        log.debug(f"Discovered field '{field.name}' with {field.ncomps} component(s)")

    fields = sorted(fields, key=attrgetter('name'))
    fields = sorted(fields, key=attrgetter('cells'))
    fields = list(discover_decompositions(fields))

    for field in geometries:
        log.debug(f"Discovered geometry '{field.name}' with coordinates {field.coords}")

    return geometries, fields


def pick_geometry(geometries: List[Field]) -> Tuple[Field, Converter]:
    if not geometries:
        raise TypeError("No geometry found, don't know what to do")

    # Find the geometry that can most easily be converted to the target
    index, converter = graph.optimal_source(config.coords, map(attrgetter('coords'), geometries))
    return geometries[index], converter


def pipeline(reader: Source, writer: Writer):
    """Main driver for moving data from reader to writer."""

    # TODO: Streamline filter application
    if config.only_final_timestep:
        reader = LastStepFilter(reader)
    reader = TesselatorFilter(reader)

    if writer.writer_name != 'VTF':
        reader = MergeTopologiesFilter(reader)

    geometries, fields = discover_fields(reader)
    geometry, converter = pick_geometry(geometries)
    log.debug(f"Using '{geometry.name}' as geometry input")

    if not converter.is_trivial and any(f.is_vector for f in fields):
        log.warning(f"Nontrivial coordinate transformations detected")
        trivial = False
        geometry_nodes = dict()
    else:
        trivial = True

    first = True
    for stepid, stepdata in log.iter.plain('Step', reader.steps()):
        with writer.step(stepdata) as step:
            with step.geometry(geometry) as geom:
                for patch, data in geometry.patches(stepid, force=first):
                    if not trivial:
                        geometry_nodes[patch.key] = data
                    data = converter.points(geometry.coords, config.coords, data)
                    geom(patch, data)

            for field in fields:
                with step.field(field) as fld:
                    for patch, data in field.patches(stepid, force=first, coords=geometry.coords):
                        if field.is_vector and trivial:
                            data = converter.vectors(geometry.coords, config.coords, data)
                        elif field.is_vector:
                            data = converter.vectors(geometry.coords, config.coords, data, nodes=geometry_nodes[patch.key])
                        fld(patch, data)

            first = False
