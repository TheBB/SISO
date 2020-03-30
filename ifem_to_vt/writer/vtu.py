import vtk
from .vtk import AbstractVTKWriter


class Writer(AbstractVTKWriter):

    def validate_mode(self):
        if not self.mode in ('appended', 'ascii', 'binary'):
            raise ValueError("VTK format does not support '{}' mode".format(self.mode))

    def writer(self):
        writer = vtk.vtkXMLUnstructuredGridWriter()
        if self.mode == 'appended':
            writer.SetDataModeToAppended()
        elif self.mode == 'ascii':
            writer.SetDataModeToAscii()
        elif self.mode == 'binary':
            writer.SetDataModeToBinary()
        return writer
