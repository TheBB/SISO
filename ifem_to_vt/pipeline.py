import treelog as log

from . import config
from .reader import Reader
from .writer import Writer


def pipeline(reader: Reader, writer: Writer):
    """Main driver for moving data from reader to writer."""

    first = True
    for stepid, stepdata in log.iter.plain('Step', reader.steps()):
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
