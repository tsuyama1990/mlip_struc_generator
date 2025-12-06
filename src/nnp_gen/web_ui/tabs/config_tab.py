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
    temperature = param.Number(default=300.0, bounds=(0.1, 10000.0))
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
                pn.Param(self.param.alloy_lattice_constant, name="Lattice Constant"),
            )
        elif self.system_type == "molecule":
            return pn.Column(
                pn.Param(self.param.molecule_smiles, name="SMILES"),
            )
        else:
            return pn.pane.Markdown(f"**{self.system_type} configuration not fully implemented in UI demo.**")

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

        # 2. Update Exploration Config
        expl_data = ensure_dict(config_data, "exploration")
        expl_data["temperature"] = self.temperature
        expl_data["steps"] = self.steps

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
                if "temperature" in expl:
                    self.temperature = expl["temperature"]
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
                 # Check logs for keywords
                 if "Step 1: Structure Generation" in content:
                     self.progress_value = max(self.progress_value, 25)
                 if "Step 2: Exploration" in content:
                     self.progress_value = max(self.progress_value, 50)
                 if "Step 3: Sampling" in content:
                     self.progress_value = max(self.progress_value, 75)
                 if "Step 4: Saving" in content:
                     self.progress_value = max(self.progress_value, 90)
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
        log_area = pn.widgets.TextAreaInput.from_param(self.vm.param.logs, height=300, disabled=True)

        try:
            pn.state.add_periodic_callback(self.vm.update_logs, period=1000)
        except RuntimeError:
            pass

        return pn.Row(
            pn.Column(
                "## Configuration",
                file_input,
                pn.Param(self.vm.param.system_type),
                pn.Param(self.vm.param.elements_input),
                self.vm.system_settings_panel,
                "### Physics",
                pn.Param(self.vm.param.max_atoms),
                pn.Param(self.vm.param.min_distance),
                "### MD Exploration",
                pn.Param(self.vm.param.temperature),
                pn.Param(self.vm.param.steps),
                run_btn,
                progress_bar,
                pn.Param(self.vm.param.status_message, widgets={'status_message': pn.widgets.StaticText}),
            ),
            pn.Column(
                "## Logs",
                log_area
            )
        )
