import treelog as log

from typing import TypeVar, Iterable, List

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


def discover_fields(reader: Reader) -> List[Field]:
    fields: List[Field] = []
    for field in reader.fields():
        fields.append(field)
        log.debug(f"Discovered field '{field.name}' with {field.ncomps} component(s)")

        for subfield in field.decompositions():
            fields.append(subfield)
            log.debug(f"Discovered decomposed scalar field '{subfield.name}'")

    return fields


def pipeline(reader: Reader, writer: Writer):
    """Main driver for moving data from reader to writer."""

    if config.only_final_timestep:
        config.require(multiple_timesteps=False)

    steps = reader.steps()
    if config.only_final_timestep:
        steps = last(steps)

    fields = discover_fields(reader)

    first = True
    for stepid, stepdata in log.iter.plain('Step', steps):
        writer.add_step(**stepdata)

        for patch in reader.geometry(stepid, force=first):
            writer.update_geometry(patch)
        writer.finalize_geometry()

        for field in fields:
            for patch in field.patches(stepid, force=first):
                writer.update_field(patch)

        writer.finalize_step()
        first = False
