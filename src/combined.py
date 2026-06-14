import os
import asyncio
from pathlib import Path
from trame.app import TrameApp
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as v3
from trame.widgets import vtk as vtk3
from trame.widgets import html

import vtkmodules.vtkRenderingOpenGL2  # noqa
import vtk

class ObjViewerApp(TrameApp):
    def __init__(self, server=None):
        super().__init__(server)

        self.vtk_actors = {}  # Local cache for VTK actors keyed by file name

        self.setup_vtk_pipeline()

        # Initialize shared state variables
        self.state.setdefault("loading", True)
        self.state.setdefault("directory_path", "")
        self.state.setdefault("vtk_file","../data/soln_2048x2048x128.vtk")
        self.state.setdefault("files_list", [])     
        self.state.setdefault("visibilities", {})   
        self.state.setdefault("slice_visible", True)
        self.state.setdefault("slice_value", 0.0)
        self.state.setdefault("slice_min", 0.0)
        self.state.setdefault("slice_max", 100.0)
        self.state.setdefault("slice_enabled", False)
        self.state.setdefault("selected_files", {})   
        self.state.setdefault("key_pressed", "None")
        self.state.setdefault("obj_opacity", 0.5)
        self.state.setdefault("use_field_coloring", False)
        
        self._build_ui()

        self.state.change("directory_path")(self.load_directory)
        self.state.change("vtk_file")(self.load_vtk_file)
        self.state.change("visibilities")(self.on_visibilities_change)
        self.state.change("slice_value")(self.on_slice_value_change)
        self.state.change("slice_visible")(self.on_slice_visibility_toggle)
        self.state.change("selected_files")(self.on_selection_change)
        self.state.change("slice_value")(self.on_slice_value_change)
        self.state.change("key_pressed")(self.on_key_pressed)
        self.state.change("obj_opacity")(self.on_obj_opacity_change)
        self.state.change("use_field_coloring")(self.on_field_coloring_toggle)
        
    def _process_vtk_pipeline(self, vtk_file):
        """Heavy blocking operations run safely inside a background thread."""
        print(f"Loading VTK file in background: {vtk_file}")
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

        # Push the dynamic range configurations back to the UI state
        self.state.slice_min = bounds[4]  # Z-min
        self.state.slice_max = bounds[5]  # Z-max
        self.state.slice_value = center[2]
        self.state.slice_enabled = True

        self.plane = vtk.vtkPlane()
        # When editing, we will reset the origin's Z value based on the slider, but keep X and Y centered
        self.plane.SetOrigin(center[0], center[1], center[2])
        self.plane.SetNormal(0, 0, 1)

        self.cutter = vtk.vtkCutter()
        self.cutter.SetInputConnection(reader.GetOutputPort())
        self.cutter.SetCutFunction(self.plane)

        sliceMapper = vtk.vtkPolyDataMapper()
        sliceMapper.SetInputConnection(self.cutter.GetOutputPort())
        sliceMapper.SetLookupTable(self.lut)
        sliceMapper.SetScalarRange(self.scalar_range)

        sliceActor = vtk.vtkActor()
        sliceActor.SetMapper(sliceMapper)

        # Safely remove old actor and switch to new one
        actor = self.vtk_actors.get("vtk_slice_actor")
        if actor:
            self.renderer.RemoveActor(actor)
            
        self.vtk_actors["vtk_slice_actor"] = sliceActor
        self.renderer.AddActor(sliceActor)


    async def load_vtk_file(self, vtk_file, **kwargs):
        """Triggered automatically when vtk_file changes via the UI text input."""
        # Safety check: If empty/invalid, turn loading off right away
        if not vtk_file or not os.path.isfile(vtk_file):
            self.state.loading = False
            self.state.slice_enabled = False
            self.state.flush()
            return

        self.state.loading = True
        self.state.flush()

        try:
            # Run heavy work in background thread
            await asyncio.to_thread(self._process_vtk_pipeline, vtk_file)
            
            self.renderer.ResetCamera()
            self.ctrl.view_update()

        finally:
            # This turns off the initial launch loading mask cleanly!
            self.state.loading = False
            self.state.flush()

    def load_directory(self, directory_path, **kwargs):
        """Triggered automatically when directory_path changes via the UI text input."""
        if not directory_path or not os.path.isdir(directory_path):
            return

        # Clear any previous actors from the scene
        obj_keys = [k for k in self.vtk_actors.keys() if k != "vtk_slice_actor"]
        for key in obj_keys:
            actor = self.vtk_actors.pop(key)
            self.renderer.RemoveActor(actor)

        self.renderer.ResetCamera()
        self.ctrl.view_update()

        ui_files = []
        initial_visibilities = {}
        initial_selections = {}
        path = Path(directory_path)
        
        # Scan directory for OBJ files
        for obj_file in path.glob("*.obj"):
            file_name = obj_file.name

            try:
                reader = vtk.vtkOBJReader()
                reader.SetFileName(str(obj_file))
                
                decimate = vtk.vtkDecimatePro()
                decimate.SetInputConnection(reader.GetOutputPort())

                # Set reduction target (e.g., 0.70 means remove 70% of triangles)
                decimate.SetTargetReduction(0.70) 
                decimate.SetBoundaryVertexDeletion(0)  # Preserve boundaries
                decimate.PreserveTopologyOn()  # Helps prevent holes from forming
                
                decimate.Update()
                mesh = decimate.GetOutput()
    
                # 2. Generate UV Texture Coordinates via vtkTextureMapToPlane
                # This maps the 3D points to a 2D coordinate space [0,1]
                uv_generator = vtk.vtkTextureMapToPlane()
                uv_generator.SetInputData(mesh)
                
                # Enable automatic plane estimation using a least-squares fit of the points
                uv_generator.AutomaticPlaneGenerationOn()
                uv_generator.SetSRange(0.0, 1.0)
                uv_generator.SetTRange(0.0, 1.0)
                uv_generator.Update()
                
                probe = vtk.vtkProbeFilter()
                probe.SetInputConnection(uv_generator.GetOutputPort())          # Target geometry to color
                probe.SetSourceData(self.volume_data)   # The 3D Volume source
                probe.Update()
                
                # textured_mesh = probe.GetOutput()
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(probe.GetOutputPort())
                mapper.SetScalarModeToUsePointData()
                mapper.SetColorModeToMapScalars()
                mapper.SetLookupTable(self.lut)
                mapper.SetScalarRange(self.scalar_range)
                
                actor = vtk.vtkActor()
                actor.GetProperty().SetOpacity(self.state.obj_opacity)
                actor.GetProperty().SetColor(1.0, 1.0, 1.0)
                actor.SetMapper(mapper)
                actor.SetVisibility(1)

                self.renderer.AddActor(actor)
                self.vtk_actors[file_name] = actor
                
                # Append string representation
                ui_files.append(file_name)
                initial_visibilities[file_name] = True
                initial_selections[file_name] = False

            except Exception as e:
                print(f"Error loading {file_name}: {e}")

        # Update state cleanly using explicit assignments
        self.state.visibilities = initial_visibilities
        self.state.selected_files = initial_selections
        self.state.files_list = ui_files

        # Ensure clean state sync
        self.state.flush()
        self.update_visibilities(initial_visibilities)  # Force explicit visibility sync

        self.renderer.ResetCamera()
        self.ctrl.view_update()

    def set_all_obj_visibilities_true(self, **kwargs):
        self.set_all_obj_visibilities(True)

    def set_all_obj_visibilities_false(self, **kwargs):
        self.set_all_obj_visibilities(False)
        
    def set_all_obj_visibilities(self, visible):
        self.state.visibilities = {name: visible for name in self.state.visibilities}
        self.update_visibilities(self.state.visibilities)

    def on_field_coloring_toggle(self, use_field_coloring, **kwargs):
        """Toggle between field coloring and flat color."""
        obj_keys = [k for k in self.vtk_actors.keys() if k != "vtk_slice_actor"]
        for key in obj_keys:
            actor = self.vtk_actors[key]
            mapper = actor.GetMapper()
            if use_field_coloring:
                mapper.ScalarVisibilityOn()
            else:
                mapper.ScalarVisibilityOff()
                actor.GetProperty().SetColor(1.0, 1.0, 1.0)
    
        self.ctrl.view_update()

    def on_selection_change(self, selected_files, **kwargs):
        """Fires whenever an item is selected or deselected in the list."""
        if not selected_files:
            return

        for file_name, is_selected in selected_files.items():
            actor = self.vtk_actors.get(file_name)
            if actor:
                if is_selected:
                    # Highlight color (Yellow)
                    actor.GetProperty().SetOpacity(1.0)
                    actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # Yellow
                    actor.GetProperty().SetAmbient(0.2)
                else:
                    # Default material color (White/Grey)
                    actor.GetProperty().SetOpacity(self.state.obj_opacity)
                    actor.GetProperty().SetColor(1.0, 1.0, 1.0)
                    actor.GetProperty().SetAmbient(0.0)

        self.ctrl.view_update()

    def update_visibilities(self, visibilities):
        for file_name, is_visible in visibilities.items():
            actor = self.vtk_actors.get(file_name)
            if actor:
                # Synchronize the VTK actor state with the updated dict state
                actor.SetVisibility(1 if is_visible else 0)
        self.ctrl.view_update()

    def on_visibilities_change(self, visibilities, **kwargs):
        """Automatically fires whenever ANY switch in the UI is flipped."""
        if not visibilities:
            return
           
        self.update_visibilities(visibilities)

    def on_slice_value_change(self, slice_value, **kwargs):
        """Callback fired when the user shifts the VTK scalar range slider."""
        actor = self.vtk_actors.get("vtk_slice_actor")
        print(f"New slice value: {slice_value}")
        if not actor or not hasattr(self, "plane"):
            return
        
        # Keep X and Y centered, update Z position dynamically
        x, y, _ = self.plane.GetOrigin()
        self.plane.SetOrigin(x, y, float(slice_value))
        print(f"Slice value changed")
        
        self.ctrl.view_update()        

    def on_slice_visibility_toggle(self, slice_visible, **kwargs):
        """Callback fired when the slice checkbox/switch is toggled."""
        actor = self.vtk_actors.get("vtk_slice_actor")
        if actor:
            actor.SetVisibility(1 if slice_visible else 0)
            self.ctrl.view_update()

    def on_obj_opacity_change(self, obj_opacity, **kwargs):
        """Callback fired when the user shifts the OBJ opacity range slider."""
        obj_keys = [k for k in self.vtk_actors.keys() if k != "vtk_slice_actor"]
        for key in obj_keys:
            actor = self.vtk_actors[key]
            if not self.state.selected_files[key]:
                actor.GetProperty().SetOpacity(obj_opacity)

        self.ctrl.view_update()        

    def setup_vtk_pipeline(self):
        self.renderer = vtk.vtkRenderer()
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.SetOffScreenRendering(1) # Keep headless
        self.renderWindow.AddRenderer(self.renderer)

        self.renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        self.renderWindowInteractor.SetRenderWindow(self.renderWindow)
        self.renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

        self.renderer.SetBackground(0.1, 0.2, 0.4)
        self.renderer.ResetCamera()

        @self.server.controller.on_server_ready.add
        def ctrl_ready(**kwargs):
            self.renderer.ResetCamera()
            if hasattr(self, 'html_view'):
                self.html_view.update()

    def on_focus_selected_objects(self, event_list=None, **kwargs):
        # Print the incoming event data to your terminal to inspect it
        if event_list:
            print(f"Keystroke Event Data: {event_list}")
            # Example: check if the key object is present
            # key_pressed = event_list[0].get("key")

        """Calculates the bounding box of all selected actors and frames the camera on them."""
        print("Focus on selected objects triggered")
        # Get dictionary of selection states from Trame
        selected_dict = self.state.selected_files or {}
        
        # Filter down to the names of the files that are actively selected (True)
        active_selections = [name for name, is_sel in selected_dict.items() if is_sel]
        
        # Fallback: If nothing is selected, reset camera to the entire scene
        if not active_selections:
            print("No objects selected to focus. Resetting camera to all visible elements.")
            self.renderer.ResetCamera()
            self.html_view.push_camera()
            self.ctrl.view_update()
            return

        # Initialize global bound tracking array
        # Format: [xmin, xmax, ymin, ymax, zmin, zmax]
        global_bounds = [float('inf'), float('-inf'), float('inf'), float('-inf'), float('inf'), float('-inf')]
        valid_actor_found = False

        for file_name in active_selections:
            actor = self.vtk_actors.get(file_name)
            if actor and actor.GetVisibility():
                valid_actor_found = True
                bounds = actor.GetBounds() # Returns tuple: (xmin, xmax, ymin, ymax, zmin, zmax)
                
                # Expand global bounding limits to include this actor
                global_bounds[0] = min(global_bounds[0], bounds[0]) * 0.95 # xmin
                global_bounds[1] = max(global_bounds[1], bounds[1]) * 1.05  # xmax
                global_bounds[2] = min(global_bounds[2], bounds[2]) * 0.95  # ymin
                global_bounds[3] = max(global_bounds[3], bounds[3]) * 1.05 # ymax
                global_bounds[4] = min(global_bounds[4], bounds[4]) * 0.95  # zmin
                global_bounds[5] = max(global_bounds[5], bounds[5]) * 1.05 # zmax

        if valid_actor_found:
            print(f"Centering view on selected bounds: {global_bounds}")
            # Center and frame the camera safely using VTK's native bounds utility
            self.renderer.ResetCamera(global_bounds)
            self.html_view.push_camera()
            self.ctrl.view_update()

    def on_key_pressed(self, **kwargs):
        print(f"Key pressed state: {self.state.key_pressed}")
        if self.state.key_pressed == "F":
            self.on_focus_selected_objects()

        self.state.key_pressed = "None"  # Reset after handling

    def _build_ui(self):
        with SinglePageWithDrawerLayout(self.server, **{"@window:keydown.esc": "self.ctrl.on_escape()"}) as layout:
            layout.title.set_text("Trame OBJ Directory Viewer (Fixed Rendering State)")
            
            layout.drawer.width = 400

            with layout.drawer:
                with v3.VContainer(fluid=True):
                    v3.VTextField(
                        v_model=("vtk_file",),
                        label="Path to VTK File",
                        prepend_inner_icon="mdi-folder-open",
                        variant="outlined",
                        density="compact",
                        clearable=True,
                    )
                    # Wrap the slider and label together inside a column, keeping the toggle inline next to it
                    with html.Div(
                        v_if="slice_enabled", 
                        classes="d-flex align-center px-3 mt-n1 mb-2", 
                        style="gap: 16px; width: 100%;"
                    ):
                        # Left side: Stacked Label + Slider
                        with html.Div(style="flex-grow: 1; display: flex; flex-direction: column;"):
                            html.Div(
                                "Slice Position", 
                                classes="text-caption text-grey-darken-1 mb-n1" # Small, muted text pulled close to the slider
                            )
                            # Horizontal container to place the slider and its value side-by-side
                            with html.Div(classes="d-flex align-center", style="gap: 12px;"):
                                v3.VSlider(
                                    v_model=("slice_value",),
                                    min=("slice_min",),
                                    max=("slice_max",),
                                    step="any",
                                    density="compact",
                                    hide_details=True,
                                    color="primary",
                                    style="flex-grow: 1;"
                                )
                                # Live value display box
                                html.Div(
                                    "{{ Number(slice_value).toFixed(5) }}", # Dynamic text formatting to 2 decimal places
                                    classes="text-body-2 font-weight-medium text-grey-darken-2",
                                    style="min-width: 50px; text-align: right;"
                                )                            
                        # Right side: Inline Visibility Toggle
                        v3.VSwitch(
                            v_model=("slice_visible",),
                            color="primary",
                            density="compact",
                            hide_details=True,
                        )
                    v3.VTextField(
                        v_model=("directory_path",),
                        label="Path to OBJ Files",
                        prepend_inner_icon="mdi-folder-open",
                        variant="outlined",
                        density="compact",
                        clearable=True,
                    )

                v3.VDivider()

                with v3.VRow(classes="px-3 pb-2", style="gap: 8px;"):
                    v3.VBtn(
                        "Show all",
                        color="primary",
                        variant="tonal",
                        click=self.set_all_obj_visibilities_true
                    )
                    v3.VBtn(
                        "Hide all",
                        color="secondary",
                        variant="tonal",
                        click=self.set_all_obj_visibilities_false
                    )
                    v3.VSwitch(
                        v_model=("use_field_coloring",),
                        label="Field",
                        color="primary",
                        density="compact",
                    )

                v3.VSlider(
                    v_model=("obj_opacity",),
                    min=0.0,
                    max=1.0,
                    label="Opacity",
                    step="any",
                    density="compact",
                    hide_details=True,
                    color="primary",
                    style="flex-grow: 1;"
                )

                # Loop through flat filenames instead of raw objects
                with v3.VList(v_if="files_list.length > 0"):
                    with v3.VListItem(
                        v_for="(fileName, index) in files_list",
                        key="index",
                        # title=("fileName",)
                        classes="{'bg-yellow-lighten-5': selected_files[fileName]}"
                    ):
                        # Prepend: Selection Checkbox
                        with v3.Template(v_slot_prepend=True):
                            v3.VCheckboxBtn(
                                v_model=("selected_files[fileName]",),
                                color="amber-darken-2",
                                density="compact",
                                hide_details=True,
                                update_modelValue="selected_files[fileName] = $event; flushState(['selected_files'])"
                            )
                        # Center: Title text (styled to match highlight state)
                        v3.VListItemTitle(
                            "{{ fileName }}",
                            classes=(
                                "{'text-amber-darken-3 font-weight-bold': selected_files[fileName], "
                                "'text-right flex-grow-1': true}"
                            )
                        )

                        with v3.Template(v_slot_append=True):
                            v3.VSwitch(
                                # Bind target dynamically to visibilities['your_file_name.obj']
                                v_model=("visibilities[fileName]",),
                                color="primary",
                                hide_details=True,
                                density="compact",
                                update_modelValue="visibilities[fileName] = $event; flushState(['visibilities'])",
                            )
                            
                with v3.VContainer(v_else=True, classes="text-center text-grey mt-5"):
                    v3.VIcon("mdi-file-cad", size="x-large")
                    v3.VCardText(html="Provide a valid path containing .obj models.")

            with layout.content:
                with v3.VApp(
                    v_on_keydown_space="key_pressed = 'Space'",
                    v_on_keydown_enter="key_pressed = 'Enter'",
                    v_on_keydown_s="key_pressed = 'S'",
                    v_on_keydown_f="key_pressed = 'F'"
                ):
                    with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
                        with v3.VContainer(
                            fluid=True, 
                            classes="pa-0 fill-height", 
                            style="position: relative; overflow: hidden;"
                        ):
                            self.html_view = vtk3.VtkLocalView(self.renderWindow)
                            self.ctrl.view_update = self.html_view.update
                            self.ctrl.on_server_ready.add(self.html_view.update)

                            # 2. Loading Overlay container centered on top
                            with html.Div(
                                v_if="loading",
                                classes="d-flex flex-column justify-center align-center position-absolute fill-height",
                                style=(
                                    "position: absolute; "
                                    "top: 0; left: 0; right: 0; bottom: 0; "
                                    "width: 100%; height: 100%; "
                                    "background: rgba(255, 255, 255, 0.7); "
                                    "z-index: 5;"
                                )
                            ):
                                v3.VProgressCircular(
                                    indeterminate=True, 
                                    color="primary", 
                                    size=64,
                                    classes="mx-auto"
                                )
                                html.Div(
                                    "Loading VTK File...",
                                    classes="text-h6 text-center text-primary mt-4"
                                )


def main():
    app = ObjViewerApp()
    app.server.start()

if __name__ == "__main__":
    main()