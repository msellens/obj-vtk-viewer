from pathlib import Path

import vtk
from vtk.util.numpy_support import vtk_to_numpy

class ObjModel:
    def __init__(self, file_path, base_color=(1.0, 1.0, 1.0), opacity=0.5, use_field_coloring=False):
        self.file_path = str(Path(file_path))
        self.file_name = Path(self.file_path).name
        self.visible = True
        self.selected = False
        self.base_color = base_color
        self.opacity = float(opacity)
        self.use_field_coloring = bool(use_field_coloring)
        self.actor = None
        self.mapper = None
        self.average = 0.0

    def build_actor(self, volume_data, lut, scalar_range):
        reader = vtk.vtkOBJReader()
        reader.SetFileName(self.file_path)
        reader.Update()

        decimate = vtk.vtkDecimatePro()
        decimate.SetInputConnection(reader.GetOutputPort())
        decimate.SetTargetReduction(0.70)
        decimate.SetBoundaryVertexDeletion(0)
        decimate.PreserveTopologyOn()
        decimate.Update()

        mesh = decimate.GetOutput()

        uv_generator = vtk.vtkTextureMapToPlane()
        uv_generator.SetInputData(mesh)
        uv_generator.AutomaticPlaneGenerationOn()
        uv_generator.SetSRange(0.0, 1.0)
        uv_generator.SetTRange(0.0, 1.0)
        uv_generator.Update()

        probe = vtk.vtkProbeFilter()
        probe.SetInputConnection(uv_generator.GetOutputPort())
        probe.SetSourceData(volume_data)
        probe.Update()

        probe_data = probe.GetOutput()
        field_data = probe_data.GetPointData().GetScalars()
        arr = vtk_to_numpy(field_data)
        self.average = arr.mean()

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(probe.GetOutputPort())
        self.mapper.SetScalarModeToUsePointData()
        self.mapper.SetColorModeToMapScalars()
        self.mapper.SetLookupTable(lut)
        self.mapper.SetScalarRange(scalar_range)

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetOpacity(self.opacity)
        self.actor.GetProperty().SetColor(*self.base_color)
        self.actor.SetVisibility(1 if self.visible else 0)

        self.apply_color_mode()
        return self.actor

    def apply_color_mode(self):
        if self.mapper is None or self.actor is None:
            return

        if self.selected:
            self.mapper.ScalarVisibilityOff()
            self.actor.GetProperty().SetColor(1.0, 0.9, 0.0)
            self.actor.GetProperty().SetOpacity(1.0)
            self.actor.GetProperty().SetAmbient(0.2)
            return

        if self.use_field_coloring:
            self.mapper.ScalarVisibilityOn()
        else:
            self.mapper.ScalarVisibilityOff()
            self.actor.GetProperty().SetColor(*self.base_color)

        self.actor.GetProperty().SetOpacity(self.opacity)
        self.actor.GetProperty().SetAmbient(0.0)

    def set_visibility(self, visible):
        self.visible = bool(visible)
        if self.actor:
            self.actor.SetVisibility(1 if self.visible else 0)

    def set_selected(self, selected):
        self.selected = bool(selected)
        self.apply_color_mode()

    def set_opacity(self, opacity):
        self.opacity = float(opacity)
        if self.actor and not self.selected:
            self.actor.GetProperty().SetOpacity(self.opacity)

    def set_field_coloring(self, enabled):
        self.use_field_coloring = bool(enabled)
        self.apply_color_mode()
