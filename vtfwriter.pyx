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
        int AddGeometryElementBlock(int)
        int AddGeometryElementBlock(int, int)

    cdef cppclass VTFAResultBlock 'VTFAResultBlock' (VTFABlock):
        VTFAResultBlock(int, int, int, int)
        int SetMapToBlockID(int)
        int SetResults1D(const float*, int)
        int SetResults3D(const float*, int)
        int GetDimension()

    cdef cppclass VTFAScalarBlock 'VTFAScalarBlock' (VTFABlock):
        VTFAScalarBlock(int)
        void SetName(const char*)
        int AddResultBlock(int, int)

    cdef cppclass VTFAVectorBlock 'VTFAVectorBlock' (VTFABlock):
        VTFAVectorBlock(int)
        void SetName(const char*)
        int AddResultBlock(int, int)

    cdef cppclass VTFADisplacementBlock 'VTFADisplacementBlock' (VTFABlock):
        VTFADisplacementBlock(int)
        void SetName(const char*)
        void SetRelativeDisplacementResults(int)
        int AddResultBlock(int, int)

    cdef cppclass VTFAStateInfoBlock 'VTFAStateInfoBlock' (VTFABlock):
        VTFAStateInfoBlock()
        int SetStepData(int, const char*, float, int)


cdef class File:

    cdef VTFAFile* vtf
    cdef bytes filename
    cdef str mode
    cdef int blockid

    def __init__(self, filename, mode='r'):
        self.filename = filename.encode()
        self.mode = mode
        self.blockid = 1

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

    def NodeBlock(self, *args, **kwargs):
        blk = NodeBlock(self, self.blockid, *args, **kwargs)
        self.blockid += 1
        return blk

    def ElementBlock(self, *args, **kwargs):
        blk = ElementBlock(self, self.blockid, *args, **kwargs)
        self.blockid += 1
        return blk

    def GeometryBlock(self, *args, **kwargs):
        return GeometryBlock(self, *args, **kwargs)

    def ResultBlock(self, *args, **kwargs):
        blk = ResultBlock(self, self.blockid, *args, **kwargs)
        self.blockid += 1
        return blk

    def ScalarBlock(self, *args, **kwargs):
        blk = ScalarBlock(self, self.blockid, *args, **kwargs)
        self.blockid += 1
        return blk

    def VectorBlock(self, *args, **kwargs):
        blk = VectorBlock(self, self.blockid, *args, **kwargs)
        self.blockid += 1
        return blk

    def DisplacementBlock(self, *args, **kwargs):
        blk = DisplacementBlock(self, self.blockid, *args, **kwargs)
        self.blockid += 1
        return blk

    def StateInfoBlock(self, *args, **kwargs):
        return StateInfoBlock(self, *args, **kwargs)


cdef class Block:

    cdef VTFABlock* _vtf
    cdef File parent

    def __enter__(self):
        return self

    def __exit__(self, type, value, backtrace):
        self.parent.WriteBlock(self)

    def GetBlockID(self):
        return self._vtf.GetBlockID()

    def GetBlockType(self):
        return self._vtf.GetBlockType()


cdef class NodeBlock(Block):

    def __init__(self, parent, blockid):
        self.parent = parent
        self._vtf = new VTFANodeBlock(blockid, 0)

    cdef VTFANodeBlock* vtf(self):
        return <VTFANodeBlock*> self._vtf

    def SetNodes(self, nodes):
        cdef np.ndarray[float] data = np.ascontiguousarray(nodes, dtype=ctypes.c_float)
        self.vtf().SetNodes(&data[0], len(nodes) / 3)


cdef class ElementBlock(Block):

    def __init__(self, parent, blockid):
        self.parent = parent
        self._vtf = new VTFAElementBlock(blockid, 0, 0)

    cdef VTFAElementBlock* vtf(self):
        return <VTFAElementBlock*> self._vtf

    def BindNodeBlock(self, NodeBlock blk):
        self.vtf().SetNodeBlockID(blk.GetBlockID())
        self.vtf().SetPartID(blk.GetBlockID())

    def SetPartName(self, name):
        self.vtf().SetPartName(name.encode())

    def AddElements(self, elements, dim):
        cdef np.ndarray[int] data = np.ascontiguousarray(elements, dtype=ctypes.c_int)
        if dim == 1:
            self.vtf().AddElements(VTFA_BEAMS, &data[0], len(elements) // 2)
        elif dim == 2:
            self.vtf().AddElements(VTFA_QUADS, &data[0], len(elements) // 4)
        else:
            self.vtf().AddElements(VTFA_HEXAHEDRONS, &data[0], len(elements) // 8)


cdef class GeometryBlock(Block):

    def __init__(self, parent):
        self.parent = parent
        self._vtf = new VTFAGeometryBlock()

    cdef VTFAGeometryBlock* vtf(self):
        return <VTFAGeometryBlock*> self._vtf

    def BindElementBlocks(self, *blocks, step=None):
        if step is None:
            for blk in blocks:
                self.vtf().AddGeometryElementBlock(blk.GetBlockID())
        else:
            for blk in blocks:
                self.vtf().AddGeometryElementBlock(blk.GetBlockID(), step)


cdef class ResultBlock(Block):

    def __init__(self, parent, blockid, vector=False, cells=False):
        self.parent = parent
        self._vtf = new VTFAResultBlock(
            blockid,
            VTFA_DIM_VECTOR if vector else VTFA_DIM_SCALAR,
            VTFA_RESMAP_ELEMENT if cells else VTFA_RESMAP_NODE,
            0
        )

    cdef VTFAResultBlock* vtf(self):
        return <VTFAResultBlock*> self._vtf

    def BindBlock(self, blk):
        self.vtf().SetMapToBlockID(blk.GetBlockID())

    def SetResults(self, results):
        cdef np.ndarray[float] data = np.ascontiguousarray(results, dtype=ctypes.c_float)
        if self.GetDimension() == VTFA_DIM_SCALAR:
            self.vtf().SetResults1D(&data[0], len(results))
        elif self.GetDimension() == VTFA_DIM_VECTOR:
            self.vtf().SetResults3D(&data[0], len(results) / 3)

    def GetDimension(self):
        return self.vtf().GetDimension()


cdef class ScalarBlock(Block):

    def __init__(self, parent, blockid):
        self.parent = parent
        self._vtf = new VTFAScalarBlock(blockid)

    cdef VTFAScalarBlock* vtf(self):
        return <VTFAScalarBlock*> self._vtf

    def SetName(self, name):
        self.vtf().SetName(name.encode())

    def BindResultBlocks(self, step, *blocks):
        for blk in blocks:
            self.vtf().AddResultBlock(blk.GetBlockID(), step)


cdef class VectorBlock(Block):

    def __init__(self, parent, blockid):
        self.parent = parent
        self._vtf = new VTFAVectorBlock(blockid)

    cdef VTFAVectorBlock* vtf(self):
        return <VTFAVectorBlock*> self._vtf

    def SetName(self, name):
        self.vtf().SetName(name.encode())

    def BindResultBlocks(self, step, *blocks):
        for blk in blocks:
            self.vtf().AddResultBlock(blk.GetBlockID(), step)


cdef class DisplacementBlock(Block):

    def __init__(self, parent, blockid, relative=True):
        self.parent = parent
        self._vtf = new VTFADisplacementBlock(blockid)
        self.vtf().SetRelativeDisplacementResults(1 if relative else 0)

    cdef VTFADisplacementBlock* vtf(self):
        return <VTFADisplacementBlock*> self._vtf

    def SetName(self, name):
        self.vtf().SetName(name.encode())

    def BindResultBlocks(self, step, *blocks):
        for blk in blocks:
            self.vtf().AddResultBlock(blk.GetBlockID(), step)


cdef class StateInfoBlock(Block):

    def __init__(self, parent):
        self.parent = parent
        self._vtf = new VTFAStateInfoBlock()

    cdef VTFAStateInfoBlock* vtf(self):
        return <VTFAStateInfoBlock*> self._vtf

    def SetStepData(self, step, name, time):
        self.vtf().SetStepData(step, name.encode(), time, 0)

    def SetModeData(self, mode, name, time):
        self.vtf().SetStepData(mode, name.encode(), time, 1)
