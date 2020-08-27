from operator import attrgetter
import treelog as log

from typing import TypeVar, Iterable, List, Tuple

from . import config
from .reader import Reader
from .writer import Writer
from .fields import Field, ComponentField


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
    return geometries, list(discover_decompositions(fields))


def pipeline(reader: Reader, writer: Writer):
    """Main driver for moving data from reader to writer."""

    if config.only_final_timestep:
        config.require(multiple_timesteps=False)

    steps = reader.steps()
    if config.only_final_timestep:
        steps = last(steps)

    geometries, fields = discover_fields(reader)

    if not geometries:
        raise TypeError("No geometry found, don't know what to do")
    geometry = geometries[0]

    first = True
    for stepid, stepdata in log.iter.plain('Step', steps):
        writer.add_step(**stepdata)

        for patch, data in geometry.patches(stepid, force=first):
            writer.update_geometry(geometry, patch, data)
            # log.debug(f"Updating geometry {patch.key}")
        writer.finalize_geometry()

        for field in fields:
            for patch, data in field.patches(stepid, force=first):
                writer.update_field(field, patch, data)

        writer.finalize_step()
        first = False
