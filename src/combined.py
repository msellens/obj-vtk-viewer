import os
from pathlib import Path
from trame.app import TrameApp
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as v3
from trame.widgets import vtk as vtk3

import vtkmodules.vtkRenderingOpenGL2  # noqa
import vtk

def toggle_visibility(file_name):
    """Controller method to toggle visibility of a specific file's actor."""
    app = ObjViewerApp.get_instance()
    app.toggle_visibility(file_name)
    

class ObjViewerApp(TrameApp):
    def __init__(self, server=None):
        super().__init__(server)

        self.vtk_actors = {}  # Local cache for VTK actors keyed by file name

        self.setup_vtk_pipeline()

        # Initialize shared state variables
        self.state.setdefault("directory_path", "")
        self.state.setdefault("files_list", [])     # Pure list of strings/names
        self.state.setdefault("visibilities", {})   # Flat dictionary: {"filename.obj": True}

        self.state.change("directory_path")(self.load_directory)

        self.state.change("visibilities")(self.on_visibilities_change)

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
        initial_visibilities = {}
        path = Path(directory_path)
        
        # Scan directory for OBJ files
        for obj_file in path.glob("*.obj"):
            file_name = obj_file.name
            
            try:
                reader = vtk.vtkOBJReader()
                reader.SetFileName(str(obj_file))
                
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(reader.GetOutputPort())
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                
                self.renderer.AddActor(actor)
                self.vtk_actors[file_name] = actor
                
                # Append string representation
                ui_files.append(file_name)
                initial_visibilities[file_name] = True
                
            except Exception as e:
                print(f"Error loading {file_name}: {e}")

        # Update state cleanly using explicit assignments
        self.state.visibilities = initial_visibilities
        self.state.files_list = ui_files
        
        self.renderer.ResetCamera()
        self.ctrl.view_update()

    def toggle_visibility(self, file_name):
        """Triggered manually by a change event on individual switch toggles."""
        # Grab visibility state out of our flat dictionary map
        is_visible = self.state.visibilities.get(file_name, True)
        # print(f"Toggling visibility for {file_name}")
        
        actor = self.vtk_actors.get(file_name)
        if actor:
            actor.SetVisibility(1 if is_visible else 0)
            # print(f"Toggled visibility for {file_name}: {'Visible' if is_visible else 'Hidden'}")
            self.ctrl.view_update()
    
    def on_visibilities_change(self, visibilities, **kwargs):
        """Automatically fires whenever ANY switch in the UI is flipped."""
        if not visibilities:
            return
            
        # print(f"Visibilities updated state: {visibilities}")

        for file_name, is_visible in visibilities.items():
            actor = self.vtk_actors.get(file_name)
            if actor:
                # Synchronize the VTK actor state with the updated dict state
                actor.SetVisibility(1 if is_visible else 0)
                # print(f"Toggled visibility for {file_name}: {'Visible' if is_visible else 'Hidden'}")
        
        # Force the render window to update the view
        self.ctrl.view_update()

    def setup_vtk_pipeline(self):
        self.renderer = vtk.vtkRenderer()
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.SetOffScreenRendering(1) # Keep headless
        self.renderWindow.AddRenderer(self.renderer)

        self.renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        self.renderWindowInteractor.SetRenderWindow(self.renderWindow)
        self.renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

        # Dummy pathing example - swap out for your real vtk structure file
        reader = vtk.vtkStructuredPointsReader()
        reader.SetFileName(Path("/Users/marcellens/data/soln_2048x2048x128.vtk"))
        reader.Update()

        volume_data = reader.GetOutput()
        center = volume_data.GetCenter()
        scalar_range = volume_data.GetPointData().GetScalars().GetRange()

        lut = vtk.vtkLookupTable()
        lut.SetTableRange(scalar_range[0], scalar_range[1])
        lut.SetHueRange(0.667, 0.0)
        lut.Build()

        # Isosurface Mesh
        iso = vtk.vtkContourFilter()
        iso.SetInputConnection(reader.GetOutputPort())
        iso.SetValue(0, 50)

        isoMapper = vtk.vtkPolyDataMapper()
        isoMapper.SetInputConnection(iso.GetOutputPort())
        isoMapper.ScalarVisibilityOff()

        isoActor = vtk.vtkActor()
        isoActor.SetMapper(isoMapper)
        isoActor.GetProperty().SetRepresentationToWireframe()
        isoActor.GetProperty().SetOpacity(0.15)
        # self.renderer.AddActor(isoActor)

        # Cutter Slice
        plane = vtk.vtkPlane()
        plane.SetOrigin(center[0], center[1], center[2])
        plane.SetNormal(0, 0, 1)

        cutter = vtk.vtkCutter()
        cutter.SetInputConnection(reader.GetOutputPort())
        cutter.SetCutFunction(plane)

        sliceMapper = vtk.vtkPolyDataMapper()
        sliceMapper.SetInputConnection(cutter.GetOutputPort())
        sliceMapper.SetLookupTable(lut)
        sliceMapper.SetScalarRange(scalar_range)

        sliceActor = vtk.vtkActor()
        sliceActor.SetMapper(sliceMapper)
        self.renderer.AddActor(sliceActor)

        self.renderer.SetBackground(0.1, 0.2, 0.4)
        self.renderer.ResetCamera()

        @self.server.controller.on_server_ready.add
        def ctrl_ready(**kwargs):
            self.renderer.ResetCamera()
            if hasattr(self, 'html_view'):
                self.html_view.update()

    def _build_ui(self):
        with SinglePageWithDrawerLayout(self.server) as layout:
            layout.title.set_text("Trame OBJ Directory Viewer (Fixed Rendering State)")

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

                # Loop through flat filenames instead of raw objects
                with v3.VList(v_if="files_list.length > 0"):
                    with v3.VListItem(
                        v_for="(fileName, index) in files_list",
                        key="index",
                        title=("fileName",)
                    ):
                        with v3.Template(v_slot_append=True):
                            v3.VSwitch(
                                # Bind target dynamically to visibilities['your_file_name.obj']
                                v_model=("visibilities[fileName]",),
                                color="primary",
                                hide_details=True,
                                density="compact",
                                # Safely pass just the fileName string argument
                                # change=(self.toggle_visibility, "fileName"), 
                                # change="trigger('toggle_visibility', [fileName])", 
                                update_modelValue="visibilities[fileName] = $event; flushState(['visibilities'])",
                            )
                            
                with v3.VContainer(v_else=True, classes="text-center text-grey mt-5"):
                    v3.VIcon("mdi-file-cad", size="x-large")
                    v3.VCardText(html="Provide a valid path containing .obj models.")

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