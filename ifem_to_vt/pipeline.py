import treelog as log

from typing import TypeVar, Iterable

from . import config
from .reader import Reader
from .writer import Writer


T = TypeVar('T')
def last(iterable: Iterable[T]) -> Iterable[T]:
    """Yield only the last element in an iterable."""
    for x in iterable:
        pass
    yield x


def pipeline(reader: Reader, writer: Writer):
    """Main driver for moving data from reader to writer."""

    first = True

    steps = reader.steps()
    if config.only_final_timestep:
        steps = last(steps)

    for stepid, stepdata in log.iter.plain('Step', steps):
        writer.add_step(**stepdata)

        for patch in reader.geometry(stepid, force=first):
            writer.update_geometry(patch)
        writer.finalize_geometry()

        for field in reader.fields():
            for patch in field.patches(stepid, force=first):
                writer.update_field(patch)
                for index, subname in field.decompositions():
                    writer.update_field(patch.pick_component(index, subname))

        writer.finalize_step()
        first = False
