from operator import attrgetter
import treelog as log

from typing import TypeVar, Iterable, List, Tuple

from . import config
from .coords import Coords, Converter, graph
from .fields import Field, ComponentField
from .reader import Reader
from .writer import Writer


T = TypeVar('T')
def last(iterable: Iterable[T]) -> Iterable[T]:
    """Yield only the last element in an iterable."""
    for x in iterable:
        pass
    yield x


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


def pipeline(reader: Reader, writer: Writer):
    """Main driver for moving data from reader to writer."""

    if config.only_final_timestep:
        config.require(multiple_timesteps=False)

    steps = reader.steps()
    if config.only_final_timestep:
        steps = last(steps)

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
    for stepid, stepdata in log.iter.plain('Step', steps):
        writer.add_step(**stepdata)

        for patch, data in geometry.patches(stepid, force=first):
            if not trivial:
                geometry_nodes[patch.key] = data
            data = converter.points(geometry.coords, config.coords, data)
            writer.update_geometry(geometry, patch, data)
        writer.finalize_geometry()

        for field in fields:
            for patch, data in field.patches(stepid, force=first, coords=geometry.coords):
                if field.is_vector and trivial:
                    data = converter.vectors(geometry.coords, config.coords, data)
                elif field.is_vector:
                    data = converter.vectors(geometry.coords, config.coords, data, nodes=geometry_nodes[patch.key])
                writer.update_field(field, patch, data)

        writer.finalize_step()
        first = False
