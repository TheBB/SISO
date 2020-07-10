import vtk
from .vtk import AbstractVTKWriter

from .. import config


class Writer(AbstractVTKWriter):

    def nan_filter(self, results):
        return results

    def validate_mode(self):
        if not config.mode in ('appended', 'ascii', 'binary'):
            raise ValueError("VTU format does not support '{}' mode".format(self.config.mode))

    def writer(self):
        writer = vtk.vtkXMLUnstructuredGridWriter()
        if config.mode == 'appended':
            writer.SetDataModeToAppended()
        elif config.mode == 'ascii':
            writer.SetDataModeToAscii()
        elif config.mode == 'binary':
            writer.SetDataModeToBinary()
        return writer
