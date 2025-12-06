import panel as pn
import param
import yaml
from typing import List, Dict, Type
from pydantic import ValidationError

from nnp_gen.core.config import (
    AppConfig, SystemConfig, AlloySystemConfig, MoleculeSystemConfig,
    IonicSystemConfig, CovalentSystemConfig, PhysicsConstraints,
    ExplorationConfig, SamplingConfig
)
from nnp_gen.web_ui.job_manager import JobManager

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

    def __init__(self, **params):
        super().__init__(**params)
        self.job_manager = JobManager()
        self._last_job_id = None

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

    def get_pydantic_config(self) -> AppConfig:
        """Convert current UI state to AppConfig."""
        elements = [e.strip() for e in self.elements_input.split(",") if e.strip()]

        constraints = PhysicsConstraints(
            max_atoms=self.max_atoms,
            min_distance=self.min_distance
        )

        if self.system_type == "alloy":
            system_config = AlloySystemConfig(
                elements=elements,
                constraints=constraints,
                lattice_constant=self.alloy_lattice_constant
            )
        elif self.system_type == "molecule":
            system_config = MoleculeSystemConfig(
                elements=elements, # Molecules might ignore this or verify against smiles
                constraints=constraints,
                smiles=self.molecule_smiles
            )
        elif self.system_type == "ionic":
             # Demo placeholder
             system_config = IonicSystemConfig(
                 elements=elements,
                 constraints=constraints,
                 oxidation_states={"O": -2} # dummy
             )
        else:
             # Covalent placeholder
             system_config = CovalentSystemConfig(
                 elements=elements,
                 constraints=constraints
             )

        expl_config = ExplorationConfig(
            temperature=self.temperature,
            steps=self.steps
        )

        samp_config = SamplingConfig()

        return AppConfig(
            system=system_config,
            exploration=expl_config,
            sampling=samp_config,
            output_dir="output"
        )

    def load_config_from_yaml(self, content: str):
        try:
            data = yaml.safe_load(content)
            # Validation via Pydantic
            # Note: A robust implementation would map every field back.
            # Here we do a partial mapping for demonstration.
            if "system" in data:
                sys = data["system"]
                self.system_type = sys.get("type", "alloy")
                if "elements" in sys:
                    self.elements_input = ",".join(sys["elements"])
                if "lattice_constant" in sys:
                    self.alloy_lattice_constant = sys["lattice_constant"]
                if "smiles" in sys:
                    self.molecule_smiles = sys["smiles"]

            self.status_message = "Config loaded successfully."
        except Exception as e:
            self.status_message = f"Error loading config: {e}"

    def run_pipeline(self, event=None):
        try:
            config = self.get_pydantic_config()
            job_id = self.job_manager.submit_job(config)
            self._last_job_id = job_id
            self.status_message = f"Job {job_id} submitted."
        except ValidationError as e:
            self.status_message = f"Validation Error: {e}"
        except Exception as e:
            self.status_message = f"Error: {e}"

    def update_logs(self):
        if self._last_job_id:
            content = self.job_manager.get_log_content(self._last_job_id)
            if content != self.logs:
                self.logs = content

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

        # Periodic Callback for logs
        # Note: In a real app, this should be added to the document or main loop.
        # Here we just define the UI.
        log_area = pn.widgets.TextAreaInput.from_param(self.vm.param.logs, height=300, disabled=True)

        # We need to register the callback when served
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
                pn.Param(self.vm.param.status_message, widgets={'status_message': pn.widgets.StaticText}),
            ),
            pn.Column(
                "## Logs",
                log_area
            )
        )
