import vtk
from .vtk import AbstractVTKWriter


class Writer(AbstractVTKWriter):

    def validate_mode(self):
        if not self.config.mode in ('appended', 'ascii', 'binary'):
            raise ValueError("VTU format does not support '{}' mode".format(self.config.mode))

    def writer(self):
        writer = vtk.vtkXMLUnstructuredGridWriter()
        if self.config.mode == 'appended':
            writer.SetDataModeToAppended()
        elif self.config.mode == 'ascii':
            writer.SetDataModeToAscii()
        elif self.config.mode == 'binary':
            writer.SetDataModeToBinary()
        return writer
