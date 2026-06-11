import os
from pathlib import Path
from trame.app import TrameApp
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as v3
from trame.widgets import vtk as vtk3

# Force VTK to load its OpenGL2 rendering backend
import vtkmodules.vtkRenderingOpenGL2  # noqa
import vtk

class ObjViewerApp(TrameApp):
    def __init__(self, server=None):
        super().__init__(server)
        # self._server = get_server()
        # self.state, self.ctrl = self.server.state, self.server.controller

        self.vtk_actors = {}  # Local cache for VTK actors keyed by file name

        self.setup_vtk_pipeline()

        # Initialize shared state variables
        self.state.setdefault("directory_path", "")
        self.state.setdefault("files_list", [])  # Holds items: {"name": "...", "visible": True}

        self.state.change("directory_path")(self.load_directory)

        self._build_ui()

    def load_directory(self, directory_path, **kwargs):
        """Triggered automatically when directory_path changes via the UI text input."""
        if not directory_path or not os.path.isdir(directory_path):
            return

        # Clear any previous actors from the scene
        for actor in self.vtk_actors.values():
            self.renderer.RemoveActor(actor)
        self.vtk_actors.clear()

        ui_files = []
        path = Path(directory_path)
        
        # Scan directory for OBJ files
        for obj_file in path.glob("*.obj"):
            file_name = obj_file.name
            
            try:
                # Initialize VTK Pipeline for this OBJ file
                reader = vtk.vtkOBJReader()
                reader.SetFileName(str(obj_file))
                
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(reader.GetOutputPort())
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                
                # Add to VTK scene
                self.renderer.AddActor(actor)
                
                # Cache the actor reference locally
                self.vtk_actors[file_name] = actor
                
                # Append entry for the Vuetify state
                ui_files.append({"name": file_name, "visible": True})
                
            except Exception as e:
                print(f"Error loading {file_name}: {e}")

        # Update state to refresh UI drawer and reset the camera view
        self.state.files_list = ui_files
        self.renderer.ResetCamera()
        self.ctrl.view_update()


    def toggle_visibility(self, item):
        """Triggered manually by a change event on individual switch toggles."""
        file_name = item.get("name")
        is_visible = item.get("visible", True)
        
        actor = self.vtk_actors.get(file_name)
        if actor:
            # Convert Python boolean to VTK int flag (1 = True, 0 = False)
            actor.SetVisibility(1 if is_visible else 0)
            self.ctrl.view_update()


    def setup_vtk_pipeline(self):
        # ---------------------------------------------------------------------
        # 1. Setup Renderer and Window
        # ---------------------------------------------------------------------
        self.renderer = vtk.vtkRenderer()
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.AddRenderer(self.renderer)

        self.renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        self.renderWindowInteractor.SetRenderWindow(self.renderWindow)
        self.renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

        # ---------------------------------------------------------------------
        # 2. Read Data & Get Range
        # ---------------------------------------------------------------------
        reader = vtk.vtkStructuredPointsReader()
        reader.SetFileName(Path("/Users/marcellens/dev/3jsdemo/soln_2048x2048x128.vtk"))
        reader.Update()

        volume_data = reader.GetOutput()
        center = volume_data.GetCenter()
        print(f"Center coordinates: X: {center[0]:.2f}, Y: {center[1]:.2f}, Z: {center[2]:.2f}")

        # Safely grab the actual min/max scalar range of your dataset
        scalar_range = volume_data.GetPointData().GetScalars().GetRange()
        print(f"Data Scalar Range: Min={scalar_range[0]:.2f}, Max={scalar_range[1]:.2f}")

        # ---------------------------------------------------------------------
        # 3. Create a Lookup Table (Color Map)
        # ---------------------------------------------------------------------
        # This guarantees your raw data values map onto visible colors
        lut = vtk.vtkLookupTable()
        lut.SetTableRange(scalar_range[0], scalar_range[1])
        lut.SetHueRange(0.667, 0.0)  # Blue-to-Red rainbow spectrum
        lut.Build()

        # ---------------------------------------------------------------------
        # 4. Isosurface Mesh (Your existing working code)
        # ---------------------------------------------------------------------
        iso = vtk.vtkContourFilter()
        iso.SetInputConnection(reader.GetOutputPort())
        iso.SetValue(0, 50)

        isoMapper = vtk.vtkPolyDataMapper()
        isoMapper.SetInputConnection(iso.GetOutputPort())
        isoMapper.ScalarVisibilityOff()

        isoActor = vtk.vtkActor()
        isoActor.SetMapper(isoMapper)
        isoActor.GetProperty().SetRepresentationToWireframe()
        isoActor.GetProperty().SetOpacity(0.15)  # Slightly more translucent to see slice
        self.renderer.AddActor(isoActor)

        # ---------------------------------------------------------------------
        # 5. Extract and Map the Slice 
        # ---------------------------------------------------------------------
        reslice = vtk.vtkImageReslice()
        reslice.SetInputConnection(reader.GetOutputPort())
        reslice.SetOutputDimensionality(2)
        reslice.SetInterpolationModeToLinear()

        reslice_axes = vtk.vtkMatrix4x4()
        reslice_axes.Identity()
        reslice_axes.SetElement(0, 3, center[0])
        reslice_axes.SetElement(1, 3, center[1])
        reslice_axes.SetElement(2, 3, center[2])
        reslice.SetResliceAxes(reslice_axes)
        reslice.Update()

        # CONVERT TO GEOMETRY: This ensures Trame's local WebGL view can render it perfectly
        surface_filter = vtk.vtkImageDataGeometryFilter()
        surface_filter.SetInputConnection(reslice.GetOutputPort())

        # Standard PolyData mapper maps the slice geometry and applies the color map
        sliceMapper = vtk.vtkPolyDataMapper()
        sliceMapper.SetInputConnection(surface_filter.GetOutputPort())
        sliceMapper.SetLookupTable(lut)
        sliceMapper.SetScalarRange(scalar_range)

        sliceActor = vtk.vtkActor()
        sliceActor.SetMapper(sliceMapper)
        self.renderer.AddActor(sliceActor)

        # ---------------------------------------------------------------------
        # 6. Finalize Render Window State
        # ---------------------------------------------------------------------
        self.renderer.SetBackground(0.1, 0.2, 0.4)
        self.renderer.ResetCamera()

        # Synchronize with Trame lifecycle when the server goes live
        @self.server.controller.on_server_ready.add
        def ctrl_ready(**kwargs):
            self.renderer.ResetCamera()
            if hasattr(self, 'html_view'):
                self.html_view.update()

    def _build_ui(self):
        """Defines the Vuetify structure using layout wrappers."""
        with SinglePageWithDrawerLayout(self.server) as layout:
            layout.title.set_text("Trame OBJ Directory Viewer (Class Based)")

            # Drawer UI
            with layout.drawer:
                with v3.VContainer(fluid=True):
                    v3.VTextField(
                        v_model=("directory_path",),
                        label="Absolute Directory Path",
                        prepend_inner_icon="mdi-folder-open",
                        variant="outlined",
                        density="compact",
                        clearable=True,
                    )

                v3.VDivider()

                with v3.VList(v_if="files_list.length > 0"):
                    with v3.VListItem(
                        v_for="(item, index) in files_list",
                        key="index",
                        title=("item.name",)
                    ):
                        with v3.Template(v_slot_append=True):
                            v3.VSwitch(
                                v_model=("item.visible",),
                                color="primary",
                                hide_details=True,
                                density="compact",
                                # Point directly to the bound instance method
                                change=(self.toggle_visibility, "item"), 
                            )
                            
                with v3.VContainer(v_else=True, classes="text-center text-grey mt-5"):
                    v3.VIcon("mdi-file-cad", size="x-large")
                    v3.VCardText(html="Provide a valid path containing .obj models.")

            # Main Context View
            with layout.content:
                with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
                    html_view = vtk3.VtkLocalView(self.renderWindow)
                    self.ctrl.view_update = html_view.update
                    self.ctrl.on_server_ready.add(html_view.update)

def main():
    app = ObjViewerApp()
    app.server.start()

if __name__ == "__main__":
    main()