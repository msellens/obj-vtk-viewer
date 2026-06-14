import vtkmodules.vtkRenderingOpenGL2  # noqa
import vtk


class VTKScene:
    def __init__(self):
        self.renderer = vtk.vtkRenderer()
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.SetOffScreenRendering(1)
        self.renderWindow.AddRenderer(self.renderer)

        self.renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        self.renderWindowInteractor.SetRenderWindow(self.renderWindow)
        self.renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

        self.renderer.SetBackground(0.1, 0.2, 0.4)
        self.renderer.ResetCamera()

        self.vtk_actors = {}
        self.volume_data = None
        self.lut = None
        self.scalar_range = (0.0, 1.0)
        self.plane = None
        self.cutter = None
        self.slice_actor = None

    def load_volume(self, vtk_file):
        reader = vtk.vtkStructuredPointsReader()
        reader.SetFileName(vtk_file)
        reader.Update()

        self.volume_data = reader.GetOutput()
        center = self.volume_data.GetCenter()
        bounds = self.volume_data.GetBounds()
        self.scalar_range = self.volume_data.GetPointData().GetScalars().GetRange()

        self.lut = vtk.vtkLookupTable()
        self.lut.SetTableRange(self.scalar_range[0], self.scalar_range[1])
        self.lut.SetHueRange(0.667, 0.0)
        self.lut.Build()

        self.plane = vtk.vtkPlane()
        self.plane.SetOrigin(center[0], center[1], center[2])
        self.plane.SetNormal(0, 0, 1)

        self.cutter = vtk.vtkCutter()
        self.cutter.SetInputConnection(reader.GetOutputPort())
        self.cutter.SetCutFunction(self.plane)

        slice_mapper = vtk.vtkPolyDataMapper()
        slice_mapper.SetInputConnection(self.cutter.GetOutputPort())
        slice_mapper.SetLookupTable(self.lut)
        slice_mapper.SetScalarRange(self.scalar_range)

        slice_actor = vtk.vtkActor()
        slice_actor.SetMapper(slice_mapper)

        if self.slice_actor:
            self.renderer.RemoveActor(self.slice_actor)

        self.slice_actor = slice_actor
        self.vtk_actors["vtk_slice_actor"] = slice_actor
        self.renderer.AddActor(slice_actor)

        return {
            "slice_min": bounds[4],
            "slice_max": bounds[5],
            "slice_value": center[2],
            "slice_enabled": True,
        }

    def reset_camera(self, bounds=None):
        if bounds is not None:
            self.renderer.ResetCamera(bounds)
        else:
            self.renderer.ResetCamera()

    def set_slice_position(self, z_value):
        if not self.plane:
            return

        x, y, _ = self.plane.GetOrigin()
        self.plane.SetOrigin(x, y, float(z_value))

    def set_slice_visibility(self, visible):
        actor = self.vtk_actors.get("vtk_slice_actor")
        if actor:
            actor.SetVisibility(1 if visible else 0)

    def add_actor(self, actor, key=None):
        self.renderer.AddActor(actor)
        if key:
            self.vtk_actors[key] = actor

    def remove_actor(self, actor=None, key=None):
        if key is not None:
            actor = self.vtk_actors.pop(key, None)

        if actor:
            self.renderer.RemoveActor(actor)

    def get_actor(self, key):
        return self.vtk_actors.get(key)

    def get_volume_data(self):
        return self.volume_data

    def get_lookup_table(self):
        return self.lut

    def get_scalar_range(self):
        return self.scalar_range
