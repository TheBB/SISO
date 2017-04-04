import numpy as np
import vtfwriter as vtf

with vtf.File('test.vtf', 'w') as f:
    with f.NodeBlock() as nodes:
        nodes.SetNodes([
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            2.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            2.0, 1.0, 0.0,
        ])

    with f.ElementBlock() as elements:
        elements.AddElements([0, 1, 4, 3, 1, 2, 5, 4], 2)
        elements.SetPartName('Patch 1')
        elements.BindNodeBlock(nodes)

    with f.GeometryBlock() as geometry:
        geometry.BindElementBlocks(elements)

    with f.ResultBlock() as result:
        result.SetResults([0.0, 1.0, 2.0, 0.0, -1.0, -2.0])
        result.BindBlock(nodes)

    with f.ScalarBlock() as scalar:
        scalar.SetName('Whatever')
        scalar.BindResultBlocks(1, result)

    with f.StateInfoBlock() as states:
        states.SetStepData(1, 'Time 0.0', 0.0)
