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

        # Add picker for ray-object intersection
        self.cell_picker = vtk.vtkCellPicker()
        self.cell_picker.SetTolerance(0.00005)  # Adjust for picking precision
        
        # Annotation actor for displaying probed values
        self.annotation_actor = None
        
        # Cached point locator for fast probing
        self.point_locator = None
 
        self.renderer.SetBackground(0.1, 0.2, 0.4)
        self.renderer.ResetCamera()

        self.vtk_actors = {}
        self.volume_data = None
        self.lut = None
        self.scalar_range = (0.0, 1.0)
        self.plane = None
        self.cutter = None
        self.slice_actor = None

    def probe_at_point(self, x, y, z, volume_data):
        """Probe scalar value at given 3D point in volume data using cached locator."""
        if not volume_data or not self.point_locator:
            return None
        
        # Use cached point locator for fast lookup
        point_id = self.point_locator.FindClosestPoint(x, y, z)
        scalars = volume_data.GetPointData().GetScalars()
        
        if scalars and point_id >= 0:
            value = scalars.GetValue(point_id)
            return {
                'point_id': point_id,
                'value': value,
                'position': (x, y, z)
            }
        return None

    def pick_actor_at_display_coords(self, x, y):
        """
        Perform ray picking at display coordinates.
        Returns intersection point and picked actor info.
        """
        self.cell_picker.Pick(x, y, 0, self.renderer)
        
        if self.cell_picker.GetCellId() != -1:
            # Ray intersected something
            pick_pos = self.cell_picker.GetPickPosition()  # 3D world position
            picked_actor = self.cell_picker.GetActor()
            
            return {
                'intersection_point': pick_pos,
                'actor': picked_actor,
                'cell_id': self.cell_picker.GetCellId(),
                'point_id': self.cell_picker.GetPointId(),
            }
        
        return None

    def create_callout(self, position, text, text_scale=0.002):
        """Create a text annotation at the given 3D position."""
        # Remove old annotation if exists
        if self.annotation_actor:
            self.renderer.RemoveActor(self.annotation_actor)
        
        # Use vtkVectorText + mapper + actor for more reliable 3D text rendering
        text_source = vtk.vtkVectorText()
        text_source.SetText(text)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(text_source.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetPosition(*position)
        print(f"Creating annotation at {position} with text: '{text}'")
        actor.GetProperty().SetColor(0.0, 0.0, 0.0)  # Yellow
        actor.SetScale(text_scale, text_scale, text_scale)
       
        self.annotation_actor = actor
        self.renderer.AddActor(actor)
        print("Adding Annotation")
        
        return actor

    def clear_callout(self):
        """Remove annotation."""
        if self.annotation_actor:
            print("Removing Annotation")
            self.renderer.RemoveActor(self.annotation_actor)
            self.annotation_actor = None

    def load_volume(self, vtk_file):
        reader = vtk.vtkStructuredPointsReader()
        reader.SetFileName(vtk_file)
        reader.Update()

        self.volume_data = reader.GetOutput()
        center = self.volume_data.GetCenter()
        bounds = self.volume_data.GetBounds()
        self.scalar_range = self.volume_data.GetPointData().GetScalars().GetRange()

        # Build point locator once for fast probing on clicks
        self.point_locator = vtk.vtkPointLocator()
        self.point_locator.SetDataSet(self.volume_data)
        self.point_locator.BuildLocator()

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

        self.remove_actor(self.slice_actor)  # Remove old slice actor if exists
        self.add_actor(slice_actor, key="vtk_slice_actor")
        self.slice_actor = slice_actor

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
