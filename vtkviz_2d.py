import vtk
import numpy as np
import multneighbors_conv as sim

class vtkTimerCallback():
    def __init__(self, simulation, steps, scalars, actor, iren):
        self.timer_count = 0
        self.simulation = simulation
        self.steps = steps
        self.scalars = scalars
        self.npoints = scalars.GetNumberOfValues()
        self.actor = actor
        self.iren = iren
        self.timerId = None
    
    def MakeLut(self, inputArr):
        colorSeries = vtk.vtkColorSeries()
        colorSeries.SetColorScheme(vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_11)
        lut = vtk.vtkColorTransferFunction()
        lut.SetColorSpaceToHSV()
        nColors = colorSeries.GetNumberOfColors()
        zMin = np.min(inputArr)
        zMax = np.max(inputArr)
        for i in range(0, nColors):
            color = colorSeries.GetColor(i)
            color = [c/255.0 for c in color]
            t = zMin + float(zMax - zMin)/(nColors - 1) * i
            lut.AddRGBPoint(t, color[0], color[1], color[2])
        return lut

    def execute(self, obj, event):
        for step in range(self.steps):
            z = self.simulation[step].flatten()
            for i in range(self.npoints):
                scalars.SetValue(i, z[i])
            lut_intm = self.MakeLut(z)
            self.actor.GetMapper().SetLookupTable(lut_intm)
            scalars.Modified()
            iren = obj
            iren.GetRenderWindow().Render()
            self.timer_count += 1

# Simulation.
steps = 250
sim.nx = 512
sim.ny = 512
simulation = sim.simulate(steps)

# Render window and interactor.
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.SetWindowName('Multiple Neighborhoods CA')
renderWindow.AddRenderer(renderer)
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(renderWindow)

# Plane to be filled with Double Array
plane = vtk.vtkPlaneSource()
plane.SetResolution(sim.nx-1, sim.ny-1)
plane.Update()
nPoints = plane.GetOutput().GetNumberOfPoints()
scalars = vtk.vtkDoubleArray()
scalars.SetNumberOfValues(nPoints)
plane.GetOutput().GetPointData().SetScalars(scalars)

# Mapper & Actor.
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(plane.GetOutputPort())
actor = vtk.vtkActor()
actor.SetMapper(mapper)
renderer.AddActor(actor)

# Animation callback.
interactor.Initialize()
cb = vtkTimerCallback(simulation, steps, scalars, actor, interactor)
interactor.AddObserver("TimerEvent", cb.execute)
cb.timerId = interactor.CreateRepeatingTimer(steps)

# Update out-stream
renderWindow.Render()
interactor.Start()
