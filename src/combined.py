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

from obj_model import ObjModel
from vtk_scene import VTKScene

class ObjViewerApp(TrameApp):
    def __init__(self, server=None):
        super().__init__(server)

        self.vtk_scene = VTKScene()
        self.obj_models = {}  # Per-OBJ render model storage

        # Initialize shared state variables
        self.state.setdefault("loading", True)
        self.state.setdefault("directory_path", "")
        self.state.setdefault("vtk_file","../data/soln_2048x2048x128.vtk")
        self.state.setdefault("files_list", [])     
        self.state.setdefault("file_display_names", {})
        self.state.setdefault("visibilities", {})   
        self.state.setdefault("slice_visible", True)
        self.state.setdefault("slice_value", 0.0)
        self.state.setdefault("slice_min", 0.0)
        self.state.setdefault("slice_max", 100.0)
        self.state.setdefault("slice_enabled", False)
        self.state.setdefault("selected_files", {})   
        self.state.setdefault("key_pressed", "None")
        self.state.setdefault("obj_opacity", 0.5)
        self.state.setdefault("rank_visibility", 50.0)
        self.state.setdefault("min_field", 0.0)
        self.state.setdefault("max_field", 100.0)
        self.state.setdefault("use_field_coloring", False)
        self.state.setdefault("probe_data", "")
        self.state.setdefault("last_click_pos", [0, 0])

        self._build_ui()

        @self.server.controller.on_server_ready.add
        def ctrl_ready(**kwargs):
            self.vtk_scene.reset_camera()
            if hasattr(self, "html_view"):
                self.html_view.update()

        self.state.change("directory_path")(self.load_directory)
        self.state.change("vtk_file")(self.load_vtk_file)
        self.state.change("visibilities")(self.on_visibilities_change)
        self.state.change("slice_value")(self.on_slice_value_change)
        self.state.change("slice_visible")(self.on_slice_visibility_toggle)
        self.state.change("selected_files")(self.on_selection_change)
        self.state.change("slice_value")(self.on_slice_value_change)
        self.state.change("key_pressed")(self.on_key_pressed)
        self.state.change("obj_opacity")(self.on_obj_opacity_change)
        self.state.change("rank_visibility")(self.on_rank_visibility_change)
        self.state.change("use_field_coloring")(self.on_field_coloring_toggle)
        self.state.change("last_click_pos")(self.on_click)

    def _process_vtk_pipeline(self, vtk_file):
        """Heavy blocking operations run safely inside a background thread."""
        return self.vtk_scene.load_volume(vtk_file)


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
            metadata = await asyncio.to_thread(self._process_vtk_pipeline, vtk_file)

            self.state.slice_min = metadata["slice_min"]
            self.state.slice_max = metadata["slice_max"]
            self.state.slice_value = metadata["slice_value"]
            self.state.slice_enabled = metadata["slice_enabled"]

            self.vtk_scene.reset_camera()
            self.ctrl.view_update()

        finally:
            # This turns off the initial launch loading mask cleanly!
            self.state.loading = False
            self.state.flush()

    def load_directory(self, directory_path, **kwargs):
        """Triggered automatically when directory_path changes via the UI text input."""
        if not directory_path or not os.path.isdir(directory_path):
            return

        # Clear any previous OBJ models from the scene
        for key, model in list(self.obj_models.items()):
            if model.actor:
                self.vtk_scene.remove_actor(actor=model.actor)
            self.obj_models.pop(key, None)

        self.vtk_scene.reset_camera()
        self.ctrl.view_update()

        ui_files = []
        file_display_names = {}
        initial_visibilities = {}
        initial_selections = {}
        path = Path(directory_path)

        # Scan directory for OBJ files
        for obj_file in path.glob("*.obj"):
            try:
                model = ObjModel(
                    str(obj_file),
                    base_color=(1.0, 1.0, 1.0),
                    opacity=self.state.obj_opacity,
                    use_field_coloring=self.state.use_field_coloring,
                )
                actor = model.build_actor(
                    self.vtk_scene.get_volume_data(),
                    self.vtk_scene.get_lookup_table(),
                    self.vtk_scene.get_scalar_range(),
                )
                self.vtk_scene.add_actor(actor)
                self.obj_models[model.file_path] = model

                ui_files.append(model.file_path)
                file_display_names[model.file_path] = model.file_name
                initial_visibilities[model.file_path] = model.visible
                initial_selections[model.file_path] = model.selected

            except Exception as e:
                print(f"Error loading {obj_file.name}: {e}")

        sorted_files = sorted(
            self.obj_models,
            key=lambda key: self.obj_models[key].average,
            reverse=True,
        )
        self.state.files_list = sorted_files

        # Update state cleanly using explicit assignments
        self.state.file_display_names = file_display_names
        self.state.visibilities = initial_visibilities
        self.state.selected_files = initial_selections
        # self.state.files_list = ui_files

        # Ensure clean state sync
        self.state.flush()
        self.update_visibilities(initial_visibilities)

        self.vtk_scene.reset_camera()
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
        for model in self.obj_models.values():
            model.set_field_coloring(use_field_coloring)

        self.ctrl.view_update()

    def on_selection_change(self, selected_files, **kwargs):
        """Fires whenever an item is selected or deselected in the list."""
        if not selected_files:
            return

        for file_key, is_selected in selected_files.items():
            model = self.obj_models.get(file_key)
            if model:
                model.set_selected(is_selected)

        self.ctrl.view_update()

    def update_visibilities(self, visibilities):
        for file_key, is_visible in visibilities.items():
            model = self.obj_models.get(file_key)
            if model:
                model.set_visibility(is_visible)
        self.ctrl.view_update()

    def on_visibilities_change(self, visibilities, **kwargs):
        """Automatically fires whenever ANY switch in the UI is flipped."""
        if not visibilities:
            return
           
        self.update_visibilities(visibilities)

    def on_slice_value_change(self, slice_value, **kwargs):
        """Callback fired when the user shifts the VTK slice slider."""
        print(f"New slice value: {slice_value}")
        self.vtk_scene.set_slice_position(slice_value)
        self.ctrl.view_update()

    def on_slice_visibility_toggle(self, slice_visible, **kwargs):
        """Callback fired when the slice checkbox/switch is toggled."""
        self.vtk_scene.set_slice_visibility(slice_visible)
        self.ctrl.view_update()

    def on_obj_opacity_change(self, obj_opacity, **kwargs):
        """Callback fired when the user shifts the OBJ opacity range slider."""
        for model in self.obj_models.values():
            model.set_opacity(obj_opacity)

        self.ctrl.view_update()        

    def on_rank_visibility_change(self, rank_visibility, **kwargs):
        """Callback fired when the user shifts the rank visibility range slider."""
        for model in self.obj_models.values():
            if model.average <= rank_visibility:
                model.set_opacity(rank_visibility)
            print("Do something")

        self.ctrl.view_update()        



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
            self.vtk_scene.reset_camera()
            self.html_view.push_camera()
            self.ctrl.view_update()
            return

        # Initialize global bound tracking array
        # Format: [xmin, xmax, ymin, ymax, zmin, zmax]
        global_bounds = [float('inf'), float('-inf'), float('inf'), float('-inf'), float('inf'), float('-inf')]
        valid_actor_found = False

        for file_key in active_selections:
            model = self.obj_models.get(file_key)
            actor = model.actor if model else None
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
            self.vtk_scene.reset_camera(global_bounds)
            self.html_view.push_camera()
            self.ctrl.view_update()

    def on_key_pressed(self, **kwargs):
        print(f"Key pressed state: {self.state.key_pressed}")
        if self.state.key_pressed == "F":
            self.on_focus_selected_objects()

        self.state.key_pressed = "None"  # Reset after handling

    def on_click(self, last_click_pos, **kwargs):
        rw, rh = self.vtk_scene.renderWindow.GetSize()
        print(f"Click coords: {last_click_pos} -> render window size: {rw}x{rh}")
        
        # self.on_vtk_click(last_click_pos)
        # last_click_pos = [0, 0]  # Reset click position after handling

    def on_vtk_click(self, click_coords):
        """Handle VTK view click for ray picking and probing."""
        if not click_coords or len(click_coords) < 4:
            return
        
        self.vtk_scene.clear_callout()

        off_x, off_y, dom_w, dom_h, clientX, clientY, rectX, rectY = click_coords
        rw, rh = self.vtk_scene.renderWindow.GetSize()
        print(f"Click coords: {click_coords} -> render window size: {rw}x{rh}")
        self.vtk_scene.renderWindow.Render()

        # map DOM coords (origin top-left) -> VTK display coords (origin bottom-left)
        display_x = int(off_x * rw / dom_w)
        display_y = int((dom_h - off_y) * rh / dom_h)
        # if canvas resolution differs from CSS size, use actual canvas w/h instead

        # print(f"click: {off_x},{off_y} dom: {dom_w}x{dom_h} -> display: {display_x},{display_y}")
        print(f"off:{off_x},{off_y} dom_size:{dom_w}x{dom_h} -> display:{display_x},{display_y} rw,rh:{rw},{rh}")

        # for i in range(0,dom_w,10):
        #     for j in range(0,dom_h,10):
        #         pick_result = self.vtk_scene.pick_actor_at_display_coords(i, j)
        #         if pick_result:
        #             print(f"Successful pick at {i},{j}: {pick_result}")

        pick_result = self.vtk_scene.pick_actor_at_display_coords(display_x, display_y)

        # x, y = click_coords
        # print(f"Click coords: {click_coords}")
        # # Perform ray picking
        # pick_result = self.vtk_scene.pick_actor_at_display_coords(x, y)
        print(f"Pick result: {pick_result}")
        
        if pick_result:
            intersection_pt = pick_result['intersection_point']
            print(f"Intersection point in world coordinates: {intersection_pt}")
            # Probe the volume data at the intersection point
            probe_result = self.vtk_scene.probe_at_point(
                intersection_pt[0], 
                intersection_pt[1], 
                intersection_pt[2],
                self.vtk_scene.get_volume_data()
            )
            print(f"Probe result: {probe_result}")
            
            if probe_result:
                value = probe_result['value']
                # Create callout with probed value
                callout_text = f"Value: {value:.4f}"
                self.vtk_scene.create_callout(intersection_pt, callout_text)
                
                # Update state for UI display
                self.state.probe_data = f"Position: ({intersection_pt[0]:.2f}, {intersection_pt[1]:.2f}, {intersection_pt[2]:.2f})\nValue: {value:.4f}"
                
                self.ctrl.view_update()

    def _build_ui(self):
        with SinglePageWithDrawerLayout(self.server) as layout:
            layout.title.set_text("OBJ/VTK Viewer")
            
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
                v3.VDivider(classes="my-2")

                with v3.VContainer(fluid=True):
                    v3.VTextField(
                        v_model=("directory_path",),
                        label="Path to OBJ Files",
                        prepend_inner_icon="mdi-folder-open",
                        variant="outlined",
                        density="compact",
                        clearable=True,
                    )

                    with v3.VRow(classes="mt-2 px-3 pb-2", style="gap: 8px;"):
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
                            label="Field Color",
                            color="primary",
                            density="compact",
                            classes="ml-auto",
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
                    v3.VSlider(
                        v_model=("rank_visibility",),
                        min=("min_field",),
                        max=("max_field",),
                        label="Rank Vis",
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
                            "{{ file_display_names[fileName] }}",
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
                with v3.VContainer(
                    fluid=True, 
                    classes="pa-0 fill-height d-flex flex-column align-center justify-center",
                    style="min-height: 0;",
                    v_on_keydown_space="key_pressed = 'Space'",
                    v_on_keydown_enter="key_pressed = 'Enter'",
                    v_on_keydown_s="key_pressed = 'S'",
                    v_on_keydown_f="key_pressed = 'F'"
                ):
                    with v3.VContainer(
                        fluid=True, 
                        classes="pa-0 fill-height", 
                        style="position: relative; overflow: hidden;",
                        # v_on_click_shift="last_click_pos = [$event.offsetX, $event.offsetY, $event.target.clientWidth, $event.target.clientHeight, window.devicePixelRatio]"
                        # v_on_click_shift="last_click_pos = [$event.offsetX, $event.offsetY, $event.target.clientWidth, $event.target.clientHeight]"
                        # v_on_click_shift="last_click_pos = [$event.offsetX, $event.offsetY, $event.target.clientWidth, $event.target.clientHeight, $event.target.width, $event.target.height, window.devicePixelRatio]"
                        # v_on_click_shift="(function() { const canvas = $event.target.querySelector('canvas'); const cw = canvas ? canvas.width : $event.target.clientWidth; const ch = canvas ? canvas.height : $event.target.clientHeight; last_click_pos = [$event.offsetX, $event.offsetY, $event.target.clientWidth, $event.target.clientHeight, cw, ch, window.devicePixelRatio]; })()"
                        v_on_click_shift="(function() {"
                           "const canvas = $event.currentTarget.querySelector('canvas'); "
                            "if (!canvas) { console.log('No canvas found'); return; } "
                            "const rect = canvas.getBoundingClientRect(); "
                            "const x = $event.clientX - rect.left; "
                            "const y = $event.clientY - rect.top; "
                            "last_click_pos = [x, y, rect.width, rect.height, canvas.width, canvas.height]; "
                            "})()"
                        ):
                        self.html_view = vtk3.VtkLocalView(
                            self.vtk_scene.renderWindow,
                            ref="html_view",
                            interactive_ratio=1,
                        )
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