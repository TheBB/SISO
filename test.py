import numpy as np
import vtfwriter as vtf

with vtf.File('test.vtf', 'w') as f:
    nodes = vtf.NodeBlock(1)
    nodes.SetNodes([
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        2.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        1.0, 1.0, 0.0,
        2.0, 1.0, 0.0,
    ])
    f.WriteBlock(nodes)

    elements = vtf.ElementBlock(1)
    elements.AddElements([0, 1, 4, 3, 1, 2, 5, 4], 2)
    elements.SetPartID(1)
    elements.SetPartName('Patch 1')
    elements.SetNodeBlockID(1)
    f.WriteBlock(elements)

    geometry = vtf.GeometryBlock()
    geometry.SetGeometryElementBlocks([1])
    f.WriteBlock(geometry)

    result = vtf.ResultBlock(1)
    result.SetResults([0.0, 1.0, 2.0, 0.0, -1.0, -2.0])
    result.SetMapToBlockID(1)
    f.WriteBlock(result)

    scalar = vtf.ScalarBlock(2)
    scalar.SetName('Whatever')
    scalar.SetResultBlocks([1], 1)
    f.WriteBlock(scalar)

    states = vtf.StateInfoBlock()
    states.SetStepData(1, 'Time 0.0', 0.0)
    f.WriteBlock(states)
