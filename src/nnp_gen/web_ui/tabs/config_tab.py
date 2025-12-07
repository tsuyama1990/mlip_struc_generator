import panel as pn
import param
import yaml
from typing import List, Dict, Type, Any
from pydantic import ValidationError

from nnp_gen.core.config import (
    AppConfig, SystemConfig, AlloySystemConfig, MoleculeSystemConfig,
    IonicSystemConfig, CovalentSystemConfig, PhysicsConstraints,
    ExplorationConfig, SamplingConfig
)
from nnp_gen.web_ui.job_manager import JobManager
from nnp_gen.generators.ionic import validate_element
from ase.data import chemical_symbols

class ConfigViewModel(param.Parameterized):
    # --- System Type Selector ---
    system_type = param.Selector(
        objects=["alloy", "ionic", "covalent", "molecule"],
        default="alloy",
        doc="Type of physical system"
    )

    # --- Common Fields ---
    elements_input = param.String(default="Fe", doc="Comma-separated list of elements")

    # --- Alloy Fields ---
    alloy_lattice_constant = param.Number(default=2.87, doc="Approx. Lattice Constant (A)")

    # --- Molecule Fields ---
    molecule_smiles = param.String(default="CCO", doc="SMILES string")

    # --- Constraints ---
    max_atoms = param.Integer(default=200, bounds=(1, 10000))
    min_distance = param.Number(default=1.5, bounds=(0.1, 10.0))

    # --- Exploration ---
    temperature_mode = param.Selector(objects=["constant", "gradient"], default="constant")
    temperature = param.Number(default=300.0, bounds=(0.1, 10000.0), doc="Temperature (Constant)")
    temp_start = param.Number(default=300.0, bounds=(0.1, 10000.0), doc="Start Temp (Gradient)")
    temp_end = param.Number(default=1000.0, bounds=(0.1, 10000.0), doc="End Temp (Gradient)")
    timestep = param.Number(default=1.0, bounds=(0.1, 10.0), doc="Time step (fs)")
    steps = param.Integer(default=1000, bounds=(10, 1000000))

    # --- Actions ---
    status_message = param.String(default="Ready")
    logs = param.String(default="", label="Job Logs")

    # --- Progress ---
    progress_value = param.Integer(default=0, bounds=(0, 100))
    progress_active = param.Boolean(default=False)

    def __init__(self, **params):
        super().__init__(**params)
        self.job_manager = JobManager()
        self._last_job_id = None
        self._raw_config_data = {} # Preserves loaded config to avoid data loss

    @param.depends("system_type")
    def system_settings_panel(self):
        """Dynamic panel based on system type."""
        if self.system_type == "alloy":
            return pn.Column(
                pn.Param(
                    self.param.alloy_lattice_constant, 
                    name="Lattice Constant",
                    widgets={'alloy_lattice_constant': pn.widgets.EditableFloatSlider}
                ),
            )
        elif self.system_type == "molecule":
            return pn.Column(
                pn.Param(self.param.molecule_smiles, name="SMILES"),
            )
        else:
            return pn.pane.Markdown(f"**{self.system_type} configuration not fully implemented in UI demo.**")

    @param.depends("temperature_mode")
    def exploration_settings_panel(self):
        """Dynamic temperature controls."""
        if self.temperature_mode == "constant":
            return pn.Column(
                pn.Param(self.param.temperature, widgets={'temperature': pn.widgets.EditableFloatSlider}, name="Temperature (K)"),
            )
        else:
            return pn.Column(
                pn.Param(self.param.temp_start, widgets={'temp_start': pn.widgets.EditableFloatSlider}, name="Start Temp (K)"),
                pn.Param(self.param.temp_end, widgets={'temp_end': pn.widgets.EditableFloatSlider}, name="End Temp (K)"),
            )

    @param.depends("elements_input", watch=True)
    def validate_elements_realtime(self):
        """
        Validate elements input in real-time.
        """
        if not self.elements_input:
            self.status_message = "Ready"
            return

        elements = [e.strip() for e in self.elements_input.split(",") if e.strip()]
        invalid = []
        for el in elements:
            try:
                # Basic check, or use validate_element?
                # Using chemical_symbols check directly to allow UI responsiveness without exceptions
                if el.capitalize() not in chemical_symbols:
                    invalid.append(el)
            except Exception:
                invalid.append(el)

        if invalid:
             self.status_message = f"⚠️ Invalid elements: {', '.join(invalid)}"
        else:
             self.status_message = "Ready"

    def get_pydantic_config(self) -> AppConfig:
        """
        Convert current UI state to AppConfig.
        If a config was loaded, merge UI changes into it to preserve unmapped fields.
        """
        # Start with raw data or empty dict
        config_data = self._raw_config_data.copy() if self._raw_config_data else {}

        # Helper to ensure dict structure exists
        def ensure_dict(d, key):
            if key not in d or not isinstance(d[key], dict):
                d[key] = {}
            return d[key]

        # 1. Update System Config
        sys_data = ensure_dict(config_data, "system")
        sys_data["type"] = self.system_type

        elements = [e.strip() for e in self.elements_input.split(",") if e.strip()]
        sys_data["elements"] = elements

        constraints = ensure_dict(sys_data, "constraints")
        constraints["max_atoms"] = self.max_atoms
        constraints["min_distance"] = self.min_distance

        # Type specific updates
        if self.system_type == "alloy":
            sys_data["lattice_constant"] = self.alloy_lattice_constant
        elif self.system_type == "molecule":
            sys_data["smiles"] = self.molecule_smiles
            # Ensure elements is optional or inferred for molecule if not provided?
            # Pydantic model will handle validation.

        expl_data = ensure_dict(config_data, "exploration")
        expl_data["steps"] = self.steps
        expl_data["timestep"] = self.timestep
        expl_data["temperature_mode"] = self.temperature_mode
        
        if self.temperature_mode == "constant":
            expl_data["temperature"] = self.temperature
        else:
            expl_data["temp_start"] = self.temp_start
            expl_data["temp_end"] = self.temp_end

        # 3. Ensure other required sections exist if creating from scratch
        if "sampling" not in config_data:
            config_data["sampling"] = {}
        if "output_dir" not in config_data:
            config_data["output_dir"] = "output"

        # 4. Construct AppConfig
        # This validates everything, including the merged data
        return AppConfig(**config_data)

    def load_config_from_yaml(self, content: str):
        try:
            data = yaml.safe_load(content)
            self._raw_config_data = data # Store raw data

            # Map known fields to UI
            if "system" in data:
                sys = data["system"]
                self.system_type = sys.get("type", "alloy")
                if "elements" in sys:
                    self.elements_input = ",".join(sys["elements"])

                # Update param values based on loaded data
                if "lattice_constant" in sys:
                    self.alloy_lattice_constant = sys["lattice_constant"]
                if "smiles" in sys:
                    self.molecule_smiles = sys["smiles"]

                if "constraints" in sys:
                    cons = sys["constraints"]
                    if "max_atoms" in cons:
                        self.max_atoms = cons["max_atoms"]
                    if "min_distance" in cons:
                        self.min_distance = cons["min_distance"]

            if "exploration" in data:
                expl = data["exploration"]
                self.temperature_mode = expl.get("temperature_mode", "constant")
                
                if "temperature" in expl:
                    self.temperature = expl["temperature"]
                if "temp_start" in expl:
                    self.temp_start = expl["temp_start"]
                if "temp_end" in expl:
                    self.temp_end = expl["temp_end"]
                if "timestep" in expl:
                    self.timestep = expl["timestep"]
                if "steps" in expl:
                    self.steps = expl["steps"]

            self.status_message = "Config loaded successfully."
        except Exception as e:
            self.status_message = f"Error loading config: {e}"

    def run_pipeline(self, event=None):
        # Validate first
        if "Invalid" in self.status_message:
            return

        try:
            self.progress_active = True
            self.progress_value = 0

            config = self.get_pydantic_config()
            job_id = self.job_manager.submit_job(config)
            self._last_job_id = job_id
            self.status_message = f"Job {job_id} submitted."

        except ValidationError as e:
            self.status_message = f"Validation Error: {e}"
            self.progress_active = False
        except Exception as e:
            self.status_message = f"Error: {e}"
            self.progress_active = False

    def update_logs(self):
        if self._last_job_id:
            content = self.job_manager.get_log_content(self._last_job_id)
            if content != self.logs:
                self.logs = content

            # Simple progress heuristic based on logs?
            # Ideally JobManager exposes status.
            status = self.job_manager.get_status(self._last_job_id)
            if status == "running":
                 # Check logs for keywords and update status message
                 if "Step 1: Structure Generation" in content:
                     self.status_message = "Step 1: Structure Generation..."
                     self.progress_value = max(self.progress_value, 25)
                 if "Step 2: Exploration" in content:
                     self.status_message = "Step 2: Exploration (MD/KMC)..."
                     self.progress_value = max(self.progress_value, 50)
                 if "Step 3: Sampling" in content:
                     self.status_message = "Step 3: Sampling Structures..."
                     self.progress_value = max(self.progress_value, 75)
                 if "Step 4: Saving" in content:
                     self.status_message = "Step 4: Saving Results..."
                     self.progress_value = max(self.progress_value, 90)
                 
                 # Detailed MD Progress
                 # Log format: "MD Progress: 500/1000"
                 # We want to map this to range [25, 50] (Exploration phase)
                 import re
                 md_prog_match = re.findall(r"MD Progress: (\d+)/(\d+)", content)
                 if md_prog_match:
                     # Take the last match
                     current, total = md_prog_match[-1]
                     try:
                         perc = float(current) / float(total)
                         # Map 0-100% MD to 50-75% overall (Step 2 to Step 3)
                         overall_perc = 50 + (perc * 25)
                         self.progress_value = max(self.progress_value, int(overall_perc))
                         self.status_message = f"MD Progress: {current}/{total} steps"
                     except ZeroDivisionError:
                         pass
            elif status == "completed":
                self.progress_value = 100
                self.progress_active = False
                self.status_message = f"Job {self._last_job_id} Completed."
            elif status == "failed":
                self.progress_active = False
                self.status_message = f"Job {self._last_job_id} Failed."


class ConfigTab:
    def __init__(self):
        self.vm = ConfigViewModel()

    def view(self):
        # File Input for Config Loading
        file_input = pn.widgets.FileInput(accept=".yaml,.yml")

        def on_file_upload(event):
            if file_input.value:
                content = file_input.value.decode("utf-8")
                self.vm.load_config_from_yaml(content)

        file_input.param.watch(on_file_upload, "value")

        # Run Button
        run_btn = pn.widgets.Button(name="Run Pipeline", button_type="primary")
        run_btn.on_click(self.vm.run_pipeline)

        # Progress Bar
        progress_bar = pn.widgets.Progress(
            value=self.vm.param.progress_value,
            active=self.vm.param.progress_active,
            bar_color="primary"
        )

        # Periodic Callback for logs
        log_area = pn.widgets.TextAreaInput.from_param(
            self.vm.param.logs, 
            height=300, 
            disabled=True,
            styles={'font-family': 'monospace'},
            stylesheets=["""
                textarea, textarea:disabled {
                    color: black !important;
                    -webkit-text-fill-color: black !important;
                    opacity: 1 !important;
                    background-color: #f8f9fa !important;
                    border: 1px solid #ccc;
                }
            """]
        )

        if pn.state.curdoc:
            pn.state.onload(lambda: pn.state.add_periodic_callback(self.vm.update_logs, period=1000))

        return pn.Row(
            pn.Column(
                "## Configuration",
                file_input,
                pn.Param(self.vm.param.system_type),
                pn.Param(self.vm.param.elements_input),
                self.vm.system_settings_panel,
                "### Physics",
                pn.Param(self.vm.param.max_atoms, widgets={'max_atoms': pn.widgets.EditableIntSlider}),
                pn.Param(self.vm.param.min_distance, widgets={'min_distance': pn.widgets.EditableFloatSlider}),
                "### MD Exploration",
                pn.Param(self.vm.param.temperature_mode, widgets={'temperature_mode': pn.widgets.RadioButtonGroup}),
                self.vm.exploration_settings_panel,
                pn.Param(self.vm.param.timestep, widgets={'timestep': pn.widgets.EditableFloatSlider}),
                pn.Param(self.vm.param.steps, widgets={'steps': pn.widgets.EditableIntSlider}),
                run_btn,
                progress_bar,
                pn.Param(self.vm.param.status_message, widgets={'status_message': {'type': pn.widgets.StaticText, 'styles': {'color': 'black', 'font-weight': 'bold'}}}),
            ),
            pn.Column(
                "## Logs",
                log_area
            )
        )
