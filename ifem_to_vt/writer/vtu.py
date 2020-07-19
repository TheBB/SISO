import vtk
from .vtk import AbstractVTKWriter

from .. import config


class Writer(AbstractVTKWriter):

    def nan_filter(self, results):
        return results

    def validate_mode(self):
        if not config.output_mode in ('appended', 'ascii', 'binary'):
            raise ValueError("VTU format does not support '{}' mode".format(self.config.output_mode))

    def writer(self):
        writer = vtk.vtkXMLUnstructuredGridWriter()
        if config.output_mode == 'appended':
            writer.SetDataModeToAppended()
        elif config.output_mode == 'ascii':
            writer.SetDataModeToAscii()
        elif config.output_mode == 'binary':
            writer.SetDataModeToBinary()
        return writer
