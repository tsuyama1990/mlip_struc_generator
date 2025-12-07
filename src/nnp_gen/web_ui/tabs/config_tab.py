import panel as pn
import param
import yaml
from typing import List, Dict, Type, Any
from pydantic import ValidationError

from nnp_gen.core.config import (
    AppConfig, SystemConfig, AlloySystemConfig, MoleculeSystemConfig,
    IonicSystemConfig, CovalentSystemConfig, PhysicsConstraints,
    ExplorationConfig, SamplingConfig, UserFileSystemConfig
)
from nnp_gen.web_ui.job_manager import JobManager
from nnp_gen.generators.ionic import validate_element
from ase.data import chemical_symbols
from nnp_gen.core.physics import estimate_lattice_constant

class ConfigViewModel(param.Parameterized):
    # --- System Type Selector ---
    system_type = param.Selector(
        objects=["alloy", "ionic", "covalent", "molecule", "user_file"],
        default="alloy",
        doc="Type of physical system"
    )

    # --- Common Fields ---
    elements_input = param.String(default="Fe", doc="Comma-separated list of elements")

    # --- Alloy Fields ---
    alloy_lattice_constant = param.Number(default=2.87, doc="Approx. Lattice Constant (A)")

    # --- Molecule Fields ---
    molecule_smiles = param.String(default="CCO", doc="SMILES string")

    # --- File Fields ---
    file_path = param.String(default="", doc="Path to structure file (cif, xyz, etc.)")
    file_repeat = param.Integer(default=1, bounds=(1, 100), doc="Number of times to duplicate structures")

    # --- Advanced Physics ---
    vacancy_concentration = param.Number(default=0.0, bounds=(0.0, 0.25), doc="Fraction of vacancies (0.0 - 0.25)")

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

    # --- Monte Carlo ---
    mc_enabled = param.Boolean(default=False, doc="Enable Hybrid MD/MC")
    mc_swap_interval = param.Integer(default=100, bounds=(1, 10000), doc="Steps between MC moves")

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
        help_text = ""

        if self.system_type == "alloy":
            help_text = "Generate random solid solution alloys. Estimates lattice constant if unknown."
            content = pn.Column(
                pn.Param(
                    self.param.alloy_lattice_constant, 
                    name="Lattice Constant",
                    widgets={'alloy_lattice_constant': pn.widgets.EditableFloatSlider}
                ),
            )
        elif self.system_type == "molecule":
            help_text = "Generate molecular conformers from SMILES string."
            content = pn.Column(
                pn.Param(self.param.molecule_smiles, name="SMILES"),
            )
        elif self.system_type == "user_file":
            help_text = "Load structures from disk (CIF, XYZ, POSCAR). Use 'Repeat' to create multiple seeds from one file."
            content = pn.Column(
                pn.Param(self.param.file_path, name="File Path"),
                pn.Param(self.param.file_repeat, name="Repeat Count"),
            )
        elif self.system_type == "ionic":
            help_text = "Generate ionic crystals based on oxidation states (Requires pymatgen for prototypes)."
            content = pn.pane.Markdown("**Ionic generation relies on backend config for oxidation states.**")
        else:
            content = pn.pane.Markdown(f"**{self.system_type} configuration not fully implemented in UI demo.**")

        return pn.Column(
            pn.pane.Markdown(f"ℹ️ **Info:** {help_text}", styles={'background': '#eef', 'padding': '10px', 'border-radius': '5px'}),
            content
        )

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

    @param.depends("mc_enabled")
    def mc_settings_panel(self):
        if not self.mc_enabled:
            return pn.Column()

        return pn.Column(
            pn.pane.Markdown("Configuration for Hybrid MC/MD. Interleaves MC moves with MD steps.", styles={'font-size': '0.9em', 'color': 'gray'}),
            pn.Param(self.param.mc_swap_interval, name="Swap Interval (steps)"),
            # Strategy and Pairs are complex list types, hard to map to simple Param widgets in this demo.
            # We assume defaults (SWAP) or loaded from YAML for now.
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
        valid_elements = []
        for el in elements:
            try:
                # Basic check, or use validate_element?
                # Using chemical_symbols check directly to allow UI responsiveness without exceptions
                if el.capitalize() not in chemical_symbols:
                    invalid.append(el)
                else:
                    valid_elements.append(el)
            except Exception:
                invalid.append(el)

        if invalid:
             self.status_message = f"⚠️ Invalid elements: {', '.join(invalid)}"
        else:
             self.status_message = "Ready"
             # Auto-estimate lattice constant for Alloy
             if self.system_type == "alloy" and valid_elements:
                 try:
                     est_a = estimate_lattice_constant(valid_elements, structure='fcc') # Default/Fallback structure
                     if est_a > 0:
                         self.alloy_lattice_constant = round(est_a, 3)
                 except Exception:
                     pass

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

        # Vacancy Injection (Supported by Ionic, Alloy, Covalent, UserFile)
        if self.system_type in ["ionic", "alloy", "covalent", "user_file"]:
            sys_data["vacancy_concentration"] = self.vacancy_concentration

        # Type specific updates
        if self.system_type == "alloy":
            sys_data["lattice_constant"] = self.alloy_lattice_constant
        elif self.system_type == "molecule":
            sys_data["smiles"] = self.molecule_smiles
        elif self.system_type == "user_file":
            sys_data["path"] = self.file_path
            sys_data["repeat"] = self.file_repeat

        # 2. Exploration Config
        expl_data = ensure_dict(config_data, "exploration")
        expl_data["steps"] = self.steps
        expl_data["timestep"] = self.timestep
        expl_data["temperature_mode"] = self.temperature_mode
        
        if self.temperature_mode == "constant":
            expl_data["temperature"] = self.temperature
        else:
            expl_data["temp_start"] = self.temp_start
            expl_data["temp_end"] = self.temp_end

        # MC Config
        if self.mc_enabled:
            expl_data["method"] = "hybrid_mc_md"
            mc_data = ensure_dict(expl_data, "mc_config")
            mc_data["enabled"] = True
            mc_data["swap_interval"] = self.mc_swap_interval
            # Strategy defaults to SWAP if not present
        else:
            # If manually disabled in UI, ensure it's disabled in config
            if "mc_config" in expl_data:
                 expl_data["mc_config"]["enabled"] = False

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

                if "lattice_constant" in sys:
                    self.alloy_lattice_constant = sys["lattice_constant"]
                if "smiles" in sys:
                    self.molecule_smiles = sys["smiles"]
                if "path" in sys:
                    self.file_path = sys["path"]
                if "repeat" in sys:
                    self.file_repeat = sys["repeat"]
                if "vacancy_concentration" in sys:
                    self.vacancy_concentration = sys["vacancy_concentration"]

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

                if "mc_config" in expl and expl["mc_config"].get("enabled", False):
                    self.mc_enabled = True
                    self.mc_swap_interval = expl["mc_config"].get("swap_interval", 100)
                else:
                    self.mc_enabled = False

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

            status = self.job_manager.get_status(self._last_job_id)
            if status == "running":
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
                 
                 import re
                 md_prog_match = re.findall(r"MD Progress: (\d+)/(\d+)", content)
                 if md_prog_match:
                     current, total = md_prog_match[-1]
                     try:
                         perc = float(current) / float(total)
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

        # Tooltips / Guidance
        guidance_box = pn.pane.Markdown("""
        ### Quick Start
        1. **Select System Type** (e.g. Alloy, User File).
        2. **Enter Elements** (e.g. `Cu, Au`).
        3. **Set Vacancy %** to inject defects.
        4. **Configure Exploration** (Temp, Steps).
        5. **Enable MC** for hybrid sampling.
        6. Click **Run Pipeline**.
        """, styles={'background': '#f0f0f0', 'padding': '15px', 'border-left': '5px solid #007bff'})

        return pn.Row(
            pn.Column(
                guidance_box,
                "## Configuration",
                file_input,
                pn.Param(self.vm.param.system_type),
                pn.Param(self.vm.param.elements_input),
                self.vm.system_settings_panel,
                "### Advanced Physics",
                pn.Param(self.vm.param.vacancy_concentration, widgets={'vacancy_concentration': pn.widgets.FloatSlider}),
                pn.Param(self.vm.param.max_atoms, widgets={'max_atoms': pn.widgets.EditableIntSlider}),
                pn.Param(self.vm.param.min_distance, widgets={'min_distance': pn.widgets.EditableFloatSlider}),
                "### MD Exploration",
                pn.Param(self.vm.param.temperature_mode, widgets={'temperature_mode': pn.widgets.RadioButtonGroup}),
                self.vm.exploration_settings_panel,
                pn.Param(self.vm.param.timestep, widgets={'timestep': pn.widgets.EditableFloatSlider}),
                pn.Param(self.vm.param.steps, widgets={'steps': pn.widgets.EditableIntSlider}),
                pn.Param(self.vm.param.mc_enabled, widgets={'mc_enabled': pn.widgets.Toggle}),
                self.vm.mc_settings_panel,
                run_btn,
                progress_bar,
                pn.Param(self.vm.param.status_message, widgets={'status_message': {'type': pn.widgets.StaticText, 'styles': {'color': 'black', 'font-weight': 'bold'}}}),
            ),
            pn.Column(
                "## Logs",
                log_area
            )
        )
