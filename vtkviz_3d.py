import vtk
import numpy as np
import math
import multneighbors_conv as sim


def main():
    # Global config
    scale = 5.
    steps = 250
    sim.nx = 512
    sim.ny = 512
    simulation = sim.simulate(steps)

    # Create the RenderWindow, Renderer and Interactor.
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # create plane to warp
    plane = vtk.vtkPlaneSource()
    plane.SetResolution(sim.nx-1, sim.ny-1)
    transform = vtk.vtkTransform()
    transform.Scale(scale, scale, 1.)
    transF = vtk.vtkTransformPolyDataFilter()
    transF.SetInputConnection(plane.GetOutputPort())
    transF.SetTransform(transform)
    transF.Update()

    # map simulation array in VTKPolyData
    inputPd = transF.GetOutput()
    numPts = inputPd.GetNumberOfPoints()

    newPts = vtk.vtkPoints()
    newPts.SetNumberOfPoints(numPts)

    derivs = vtk.vtkDoubleArray()
    derivs.SetNumberOfTuples(numPts)

    bessel = vtk.vtkPolyData()
    bessel.CopyStructure(inputPd)
    bessel.SetPoints(newPts)
    bessel.GetPointData().SetScalars(derivs)

    x = np.zeros(3)
    z = simulation[-1].flatten()
    zmin, zmax = np.min(z), np.max(z)
    for i in range(0, numPts):
        inputPd.GetPoint(i, x)
        x[2] = (z[i] - zmin) / (zmax - zmin)
        newPts.SetPoint(i, x)
        derivs.SetValue(i, x[2])

    # Warp the plane.
    warp = vtk.vtkWarpScalar()
    warp.SetInputData(bessel)
    warp.XYPlaneOn()
    warp.SetScaleFactor(0.25)

    # Mapper and actor.
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(warp.GetOutputPort())
    tmp = bessel.GetScalarRange()
    mapper.SetScalarRange(tmp[0], tmp[1])
    carpet = vtk.vtkActor()
    carpet.SetMapper(mapper)
    ren.AddActor(carpet)

    # Update out window
    renWin.Render()
    iren.Start()

if __name__ == '__main__':
    main()