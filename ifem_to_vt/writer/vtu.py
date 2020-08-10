import vtk

from .vtk import VTKWriter
from .. import config


class VTUWriter(VTKWriter):

    writer_name = "VTU"

    @classmethod
    def applicable(cls, fmt: str) -> bool:
        return fmt == 'vtu'

    def nan_filter(self, results):
        return results

    def validate_mode(self):
        if not config.output_mode in ('appended', 'ascii', 'binary'):
            raise ValueError("VTU format does not support '{}' mode".format(self.config.output_mode))

    def get_writer(self):
        writer = vtk.vtkXMLUnstructuredGridWriter()
        if config.output_mode == 'appended':
            writer.SetDataModeToAppended()
        elif config.output_mode == 'ascii':
            writer.SetDataModeToAscii()
        elif config.output_mode == 'binary':
            writer.SetDataModeToBinary()
        return writer
