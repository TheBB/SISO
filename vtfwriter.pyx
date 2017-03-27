# distutils: language = c++

cimport numpy as np
import numpy as np
import ctypes


cdef extern from 'VTFAPI.h':

    cdef const int VTFA_BEAMS
    cdef const int VTFA_QUADS
    cdef const int VTFA_HEXAHEDRONS

    cdef const int VTFA_DIM_SCALAR
    cdef const int VTFA_DIM_VECTOR

    cdef const int VTFA_RESMAP_NODE
    cdef const int VTFA_RESMAP_ELEMENT

    cdef cppclass VTFAFile 'VTFAFile':
        VTFAFile()
        int CreateVTFFile(const char*, int, int)
        int OpenVTFFile(const char*)
        int CloseFile()
        int WriteBlock(VTFABlock*)

    cdef cppclass VTFABlock 'VTFABlock':
        int GetBlockID()
        int GetBlockType()

    cdef cppclass VTFANodeBlock 'VTFANodeBlock' (VTFABlock):
        VTFANodeBlock(int, int)
        int SetNodes(const float*, int)

    cdef cppclass VTFAElementBlock 'VTFAElementBlock' (VTFABlock):
        VTFAElementBlock(int, int, int)
        void SetNodeBlockID(int)
        void SetPartName(const char*)
        void SetPartID(int)
        int AddElements(int, const int*, int)

    cdef cppclass VTFAGeometryBlock 'VTFAGeometryBlock' (VTFABlock):
        VTFAGeometryBlock()
        int SetGeometryElementBlocks(const int *, int)

    cdef cppclass VTFAResultBlock 'VTFAResultBlock' (VTFABlock):
        VTFAResultBlock(int, int, int, int)
        int SetMapToBlockID(int)
        int SetResults1D(const float*, int)
        int SetResults3D(const float*, int)
        int GetDimension()

    cdef cppclass VTFAScalarBlock 'VTFAScalarBlock' (VTFABlock):
        VTFAScalarBlock(int)
        void SetName(const char*)
        void SetResultBlocks(const int*, int, int)

    cdef cppclass VTFAStateInfoBlock 'VTFAStateInfoBlock' (VTFABlock):
        VTFAStateInfoBlock()
        int SetStepData(int, const char*, float, int)


cdef class File:

    cdef VTFAFile* vtf
    cdef bytes filename
    cdef str mode

    def __init__(self, filename, mode='r'):
        self.filename = filename.encode()
        self.mode = mode

    def __enter__(self):
        self.vtf = new VTFAFile()
        if 'w' in self.mode:
            self.vtf.CreateVTFFile(self.filename, 'b' in self.mode, 0)
        else:
            self.vtf.OpenVTFFile(self.filename)
        return self

    def __exit__(self, tp, value, backtrace):
        self.vtf.CloseFile()

    def WriteBlock(self, Block block):
        self.vtf.WriteBlock(block._vtf)


cdef class Block:

    cdef VTFABlock* _vtf

    def GetBlockID(self):
        return self._vtf.GetBlockID()

    def GetBlockType(self):
        return self._vtf.GetBlockType()


cdef class NodeBlock(Block):

    def __init__(self, blockid):
        self._vtf = new VTFANodeBlock(blockid, 0)

    cdef VTFANodeBlock* vtf(self):
        return <VTFANodeBlock*> self._vtf

    def SetNodes(self, nodes):
        cdef np.ndarray[float] data = np.ascontiguousarray(nodes, dtype=ctypes.c_float)
        self.vtf().SetNodes(&data[0], len(nodes) / 3)


cdef class ElementBlock(Block):

    def __init__(self, blockid):
        self._vtf = new VTFAElementBlock(blockid, 0, 0)

    cdef VTFAElementBlock* vtf(self):
        return <VTFAElementBlock*> self._vtf

    def SetNodeBlockID(self, i):
        self.vtf().SetNodeBlockID(i)

    def SetPartName(self, name):
        self.vtf().SetPartName(name.encode())

    def SetPartID(self, i):
        self.vtf().SetPartID(i)

    def AddElements(self, elements, dim):
        cdef np.ndarray[int] data = np.ascontiguousarray(elements, dtype=ctypes.c_int)
        if dim == 1:
            self.vtf().AddElements(VTFA_BEAMS, &data[0], len(elements) / 2)
        elif dim == 2:
            self.vtf().AddElements(VTFA_QUADS, &data[0], len(elements) / 4)
        else:
            self.vtf().AddElements(VTFA_HEXAHEDRONS, &data[0], len(elements) / 8)


cdef class GeometryBlock(Block):

    def __init__(self):
        self._vtf = new VTFAGeometryBlock()

    cdef VTFAGeometryBlock* vtf(self):
        return <VTFAGeometryBlock*> self._vtf

    def SetGeometryElementBlocks(self, blocks):
        cdef np.ndarray[int] data = np.ascontiguousarray(blocks, dtype=ctypes.c_int)
        self.vtf().SetGeometryElementBlocks(&data[0], len(blocks))


cdef class ResultBlock(Block):

    def __init__(self, blockid, vector=False, cells=False):
        self._vtf = new VTFAResultBlock(
            blockid,
            VTFA_DIM_VECTOR if vector else VTFA_DIM_SCALAR,
            VTFA_RESMAP_ELEMENT if cells else VTFA_RESMAP_NODE,
            0
        )

    cdef VTFAResultBlock* vtf(self):
        return <VTFAResultBlock*> self._vtf

    def SetMapToBlockID(self, i):
        self.vtf().SetMapToBlockID(i)

    def SetResults(self, results):
        cdef np.ndarray[float] data = np.ascontiguousarray(results, dtype=ctypes.c_float)
        if self.GetDimension() == VTFA_DIM_SCALAR:
            self.vtf().SetResults1D(&data[0], len(results))
        elif self.GetDimension() == VTFA_DIM_SCALAR:
            self.vtf().SetResults3D(&data[0], len(results) / 3)

    def GetDimension(self):
        return self.vtf().GetDimension()


cdef class ScalarBlock(Block):

    def __init__(self, blockid):
        self._vtf = new VTFAScalarBlock(blockid)

    cdef VTFAScalarBlock* vtf(self):
        return <VTFAScalarBlock*> self._vtf

    def SetName(self, name):
        self.vtf().SetName(name.encode())

    def SetResultBlocks(self, blocks, step):
        cdef np.ndarray[int] data = np.ascontiguousarray(blocks, dtype=ctypes.c_int)
        self.vtf().SetResultBlocks(&data[0], len(blocks), step)


cdef class StateInfoBlock(Block):

    def __init__(self):
        self._vtf = new VTFAStateInfoBlock()

    cdef VTFAStateInfoBlock* vtf(self):
        return <VTFAStateInfoBlock*> self._vtf

    def SetStepData(self, step, name, time):
        self.vtf().SetStepData(step, name.encode(), time, 0)
