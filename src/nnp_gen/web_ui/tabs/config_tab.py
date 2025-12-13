import panel as pn
import param
import yaml
from typing import List, Dict, Type, Any
from pydantic import ValidationError

from nnp_gen.core.config import (
    AppConfig, SystemConfig, AlloySystemConfig, MoleculeSystemConfig,
    IonicSystemConfig, CovalentSystemConfig, PhysicsConstraints,
    ExplorationConfig, SamplingConfig, FileSystemConfig
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
    output_dir = param.String(default="output", doc="Directory to save results")
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

    # --- Ensemble ---
    ensemble = param.Selector(objects=["AUTO", "NVT", "NPT"], default="AUTO")
    # Using lowercase to match Enum values exactly
    thermostat = param.Selector(objects=["langevin", "berendsen", "nose_hoover"], default="nose_hoover", doc="Thermostat Type")
    pressure = param.Number(default=None, bounds=(0.0, 1000.0), doc="Pressure (GPa) for NPT")
    ttime = param.Number(default=100.0, doc="Thermostat Time (fs)")


    # --- Monte Carlo ---
    mc_enabled = param.Boolean(default=False, doc="Enable Hybrid MD/MC")
    mc_swap_interval = param.Integer(default=100, bounds=(1, 10000), doc="Steps between MC moves")
    mc_swap_pairs_input = param.String(default="", doc="Swap pairs (e.g. 'Fe-Pt, A-B')")
    
    # --- Detail Configs (Missing from previous UI) ---
    seed = param.Integer(default=42, doc="Random Seed")
    
    # System Extras
    n_initial_structures = param.Integer(default=10, bounds=(1, 1000), doc="Initial Bulk Structures")
    n_surface_samples = param.Integer(default=0, bounds=(0, 100), doc="Surface Samples per Bulk")
    
    supercell_input = param.String(default="3,3,3", doc="Supercell (x,y,z)")
    rattle_std = param.Number(default=0.01, bounds=(0.0, 1.0), doc="Rattle Std Dev (A)")
    vol_scale_min = param.Number(default=0.95, bounds=(0.5, 1.5))
    vol_scale_max = param.Number(default=1.05, bounds=(0.5, 1.5))
    
    pbc_input = param.String(default="true,true,true", doc="PBC (x,y,z)")
    
    # Constraints Extras
    min_density = param.Number(default=0.0, bounds=(0.0, 20.0), doc="Min Density")
    r_cut = param.Number(default=5.0, bounds=(1.0, 10.0), doc="Cutoff Radius")
    min_cell_length_factor = param.Number(default=1.0, bounds=(0.1, 5.0))

    # Alloy Specifics
    composition_mode = param.Selector(objects=["random", "balanced", "range"], default="range")
    # Simplification: For range mode, we might need a text input or dynamic UI. 
    # For now, let's rely on config.yaml for complex range dicts, 
    # OR provide a simple text area for 'Fe: 0.1-0.9, Pt: 0.1-0.9'
    composition_ranges_input = param.String(default="", doc="Comp Ranges (e.g. Fe:0.1-0.9; Pt:0.1-0.9)")
    spacegroup = param.Integer(default=225, bounds=(1, 230), doc="Spacegroup (1-230)")

    # Exploration Extras
    model_name = param.Selector(objects=["mace", "sevenn", "emt"], default="mace")
    snapshot_interval = param.Integer(default=100, bounds=(1, 10000))
    zbl_enabled = param.Boolean(default=False, doc="Enable ZBL Potential")
    zbl_cutoff = param.Number(default=1.5, bounds=(0.1, 10.0), doc="ZBL Cutoff (A)")
    zbl_skin = param.Number(default=0.5, bounds=(0.0, 5.0), doc="ZBL Skin (A)")
    
    # --- Compute ---
    device = param.Selector(objects=["cpu", "cuda"], default="cpu", doc="Compute Device")

    # --- Sampling ---
    sampling_strategy = param.Selector(objects=["fps", "random"], default="fps")
    sampling_descriptor = param.Selector(objects=["soap", "ace"], default="soap")
    n_samples = param.Integer(default=100, bounds=(1, 100000))

    # --- Actions ---
    status_message = param.String(default="Ready")
    logs = param.String(default="", label="Job Logs")

    # --- Progress ---
    progress_value = param.Integer(default=0, bounds=(0, 100))
    progress_active = param.Boolean(default=False)
    # Helper for UI enabling/disabling (Inverse of active)
    progress_idle = param.Boolean(default=True)

    def __init__(self, **params):
        super().__init__(**params)
        self.job_manager = JobManager()
        self._last_job_id = None
        self._raw_config_data = {} # Preserves loaded config to avoid data loss

    @param.depends("progress_active", watch=True)
    def _sync_idle_state(self):
        self.progress_idle = not self.progress_active

    @param.depends("system_type")
    def system_settings_panel(self):
        """Dynamic panel based on system type."""
        help_text = ""

        if self.system_type == "alloy":
            help_text = "Generate random solid solution alloys. Estimates lattice constant if unknown."
            content = pn.Column(
                pn.Param(
                    self.param.composition_mode,
                    widgets={'composition_mode': pn.widgets.Select}
                ),
                pn.Param(self.param.composition_ranges_input, name="Comp Ranges (auto-filled, e.g. Fe:0.1-0.9, Pt:0.1-0.9)"),
                pn.Param(
                    self.param.alloy_lattice_constant, 
                    name="Lattice Constant",
                    widgets={'alloy_lattice_constant': {'type': pn.widgets.EditableFloatSlider, 'format': '0.000'}}
                ),
                pn.Param(self.param.spacegroup, widgets={'spacegroup': {'type': pn.widgets.EditableIntSlider}}),
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
                pn.Param(self.param.file_repeat, name="Repeat Count", widgets={'file_repeat': {'type': pn.widgets.EditableIntSlider}}),
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
                pn.Param(self.param.temperature, widgets={'temperature': {'type': pn.widgets.EditableFloatSlider, 'format': '0.0f'}}, name="Temperature (K)"),
            )
        else:
            return pn.Column(
                pn.Param(self.param.temp_start, widgets={'temp_start': {'type': pn.widgets.EditableFloatSlider, 'format': '0.0f'}}, name="Start Temp (K)"),
                pn.Param(self.param.temp_end, widgets={'temp_end': {'type': pn.widgets.EditableFloatSlider, 'format': '0.0f'}}, name="End Temp (K)"),
            )
    
    @param.depends("ensemble")
    def ensemble_settings_panel(self):
        # Always show thermostat for NVT/NPT/AUTO (if supported)
        # AUTO usually implies NVT/NPT logic anyway.
        
        common_controls = [
            pn.Param(self.param.thermostat, widgets={'thermostat': pn.widgets.Select}),
        ]
        
        if self.ensemble == "NPT":
             return pn.Column(
                 *common_controls,
                 pn.Param(self.param.pressure, name="Pressure (GPa)", widgets={'pressure': {'type': pn.widgets.EditableFloatSlider, 'format': '0.0f'}}),
                 pn.Param(self.param.ttime, name="Thermostat Time (fs)", widgets={'ttime': {'type': pn.widgets.EditableFloatSlider, 'format': '0.0f'}}),
             )
        return pn.Column(*common_controls)

    @param.depends("mc_enabled")
    def mc_settings_panel(self):
        if not self.mc_enabled:
            return pn.Column()

        return pn.Column(
            pn.pane.Markdown("Configuration for Hybrid MC/MD. Interleaves MC moves with MD steps.", styles={'font-size': '0.9em', 'color': 'gray'}),
            pn.Param(self.param.mc_swap_interval, name="Swap Interval (steps)", widgets={'mc_swap_interval': {'type': pn.widgets.EditableIntSlider}}),
            pn.Param(self.param.mc_swap_pairs_input, name="Swap Pairs (auto-filled, e.g. Fe-Pt, Cu-Au)"),
        )

    @param.depends("zbl_enabled")
    def zbl_settings_panel(self):
        if not self.zbl_enabled:
            return pn.Column()
        
        return pn.Column(
            pn.Param(self.param.zbl_cutoff, name="Cutoff (A)", widgets={'zbl_cutoff': {'type': pn.widgets.EditableFloatSlider, 'format': '0.00'}}),
            pn.Param(self.param.zbl_skin, name="Skin (A)", widgets={'zbl_skin': {'type': pn.widgets.EditableFloatSlider, 'format': '0.00'}}),
        )

    @param.depends("elements_input", "composition_mode", watch=True)
    def validate_elements_realtime(self):
        """
        Validate elements input in real-time and auto-populate related fields.
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
                 
                 # Auto-populate composition ranges for "range" mode
                 if self.composition_mode == "range" and len(valid_elements) >= 2:
                     ranges = [f"{el}:0.1-0.9" for el in valid_elements]
                     self.composition_ranges_input = ", ".join(ranges)
             
             # Auto-populate MC swap pairs (all binary combinations)
             if len(valid_elements) >= 2:
                 from itertools import combinations
                 pairs = [f"{el1}-{el2}" for el1, el2 in combinations(valid_elements, 2)]
                 self.mc_swap_pairs_input = ", ".join(pairs)

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

        # General System Settings
        try:
            sc = [int(x.strip()) for x in self.supercell_input.split(",")]
            if len(sc) == 3: sys_data["supercell_size"] = sc
        except: pass

        try:
            pbc = [x.strip().lower()=='true' for x in self.pbc_input.split(",")]
            if len(sc) == 3: sys_data["pbc"] = pbc
        except: pass

        sys_data["rattle_std"] = self.rattle_std
        sys_data["vol_scale_range"] = [self.vol_scale_min, self.vol_scale_max]
        
        # Constraints
        constraints["min_density"] = self.min_density
        constraints["r_cut"] = self.r_cut
        constraints["min_cell_length_factor"] = self.min_cell_length_factor

        # Type specific updates
        if self.system_type == "alloy":
            sys_data["lattice_constant"] = self.alloy_lattice_constant
            sys_data["spacegroup"] = self.spacegroup
            sys_data["n_initial_structures"] = self.n_initial_structures
            sys_data["n_surface_samples"] = self.n_surface_samples
            sys_data["composition_mode"] = self.composition_mode
            
            if self.composition_mode == "range" and self.composition_ranges_input:
                ranges = {}
                # Parse "Fe:0.1-0.9, Pt:0.1-0.9" (comma-separated)
                parts = self.composition_ranges_input.split(",")
                for p in parts:
                    if ":" in p:
                        el, r = p.split(":")
                        rmin, rmax = r.split("-")
                        ranges[el.strip()] = (float(rmin), float(rmax))
                if ranges:
                    sys_data["composition_ranges"] = ranges
        elif self.system_type == "molecule":
            sys_data["smiles"] = self.molecule_smiles
        elif self.system_type == "user_file":
            sys_data["path"] = self.file_path
            sys_data["repeat"] = self.file_repeat

        # 2. Exploration Config
        expl_data = ensure_dict(config_data, "exploration")
        expl_data["steps"] = self.steps
        expl_data["timestep"] = self.timestep
        expl_data["snapshot_interval"] = self.snapshot_interval
        expl_data["model_name"] = self.model_name
        
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
            
            # Parse Pairs (comma-separated, no quotes needed)
            if self.mc_swap_pairs_input:
                pairs = []
                # Simple parser: "Fe-Pt, A-B" -> [('Fe', 'Pt'), ('A', 'B')]
                raw = self.mc_swap_pairs_input.split(",")
                for r in raw:
                    parts = r.strip().split("-")
                    if len(parts) == 2:
                        el1 = parts[0].strip()
                        el2 = parts[1].strip()
                        pairs.append([el1, el2])
                if pairs:
                    mc_data["swap_pairs"] = pairs

        else:
            # If manually disabled in UI, ensure it's disabled in config
            if "mc_config" in expl_data:
                 expl_data["mc_config"]["enabled"] = False
        
        expl_data["ensemble"] = self.ensemble
        if self.ensemble == "NPT" and self.pressure is not None:
             expl_data["pressure"] = self.pressure
        if self.ttime:
             expl_data["ttime"] = self.ttime

        if self.ttime:
             expl_data["ttime"] = self.ttime

        # Fix Enum Case Sensitivity
        expl_data["thermostat"] = self.thermostat.lower()

        expl_data["device"] = self.device

        # ZBL Config
        zbl_data = ensure_dict(expl_data, "zbl_config")
        zbl_data["enabled"] = self.zbl_enabled
        if self.zbl_enabled:
            zbl_data["cutoff"] = self.zbl_cutoff
            zbl_data["skin"] = self.zbl_skin

        # 3. Ensure other required sections exist if creating from scratch
        if "sampling" not in config_data:
            config_data["sampling"] = {}
        
        samp_data = ensure_dict(config_data, "sampling")
        samp_data["strategy"] = self.sampling_strategy
        samp_data["descriptor_type"] = self.sampling_descriptor
        samp_data["n_samples"] = self.n_samples
        samp_data["n_samples"] = self.n_samples
        
        # User defined output dir (relative or absolute)
        # Note: JobManager will create a subdir inside this, or if we want exact path?
        # Current JobManager logic: base_output / job_id
        # Let's trust config_data["output_dir"] = self.output_dir
        config_data["output_dir"] = self.output_dir
        config_data["seed"] = self.seed

        # 4. Construct AppConfig
        # This validates everything, including the merged data
        return AppConfig(**config_data)

    def load_config_from_yaml(self, content: str):
        try:
            data = yaml.safe_load(content)
            self._raw_config_data = data # Store raw data

            # Map known fields to UI
            if "seed" in data:
                self.seed = data["seed"]
            
            if "system" in data:
                sys = data["system"]
                self.system_type = sys.get("type", "alloy")
                if "elements" in sys:
                    self.elements_input = ",".join(sys["elements"])
                
                # Extras
                if "supercell_size" in sys:
                    self.supercell_input = ",".join(map(str, sys["supercell_size"]))
                if "pbc" in sys:
                    self.pbc_input = ",".join(map(str, sys["pbc"]))
                if "rattle_std" in sys:
                    self.rattle_std = sys["rattle_std"]
                if "vol_scale_range" in sys:
                    self.vol_scale_min = sys["vol_scale_range"][0]
                    self.vol_scale_max = sys["vol_scale_range"][1]
                if "n_initial_structures" in sys:
                    self.n_initial_structures = sys["n_initial_structures"]
                if "n_surface_samples" in sys:
                    self.n_surface_samples = sys["n_surface_samples"]


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

                if "thermostat" in expl:
                    # Robust loading: handle upper/mixed case
                    val = str(expl["thermostat"]).lower()
                    if val in ["langevin", "berendsen", "nose_hoover"]:
                        self.thermostat = val
                
                if "mc_config" in expl and expl["mc_config"].get("enabled", False):
                    self.mc_enabled = True
                    self.mc_swap_interval = expl["mc_config"].get("swap_interval", 100)
                    
                    pairs = expl["mc_config"].get("swap_pairs", [])
                    if pairs:
                        # Convert list of lists/tuples to string "A-B, C-D"
                        pair_strs = [f"{p[0]}-{p[1]}" for p in pairs if len(p)==2]
                        self.mc_swap_pairs_input = ", ".join(pair_strs)
                    
                else:
                    self.mc_enabled = False
                
                if "ensemble" in expl:
                    self.ensemble = expl["ensemble"]
                if "pressure" in expl:
                    self.pressure = expl["pressure"]
                
                if "device" in expl:
                    self.device = expl["device"]

                if "zbl_config" in expl:
                    zbl = expl["zbl_config"]
                    self.zbl_enabled = zbl.get("enabled", False)
                    if "cutoff" in zbl:
                        self.zbl_cutoff = zbl["cutoff"]
                    if "skin" in zbl:
                        self.zbl_skin = zbl["skin"]
                
            if "sampling" in data:
                samp = data["sampling"]
                if "strategy" in samp:
                    self.sampling_strategy = samp["strategy"]
                if "descriptor_type" in samp:
                    self.sampling_descriptor = samp["descriptor_type"]
                if "n_samples" in samp:
                    self.n_samples = samp["n_samples"]

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

    def stop_pipeline(self, event=None):
        if self._last_job_id:
            if self.job_manager.stop_job(self._last_job_id):
                self.status_message = f"Job {self._last_job_id} stopping..."
            else:
                self.status_message = "Failed to stop job (or not running)."

    def update_logs(self):
        if self._last_job_id:
            content = self.job_manager.get_log_content(self._last_job_id)
            if content != self.logs:
                # Intelligently filter TQDM progress updates to show only meaningful information
                import re
                
                lines = content.split('\n')
                filtered_lines = []
                last_tqdm_line = None
                
                for line in lines:
                    # Check if this is a TQDM progress line (contains progress bar characters)
                    if re.search(r'\d+%\|[▏▎▍▌▋▊▉█\s]+\|', line):
                        # Keep only the last TQDM line for each type
                        last_tqdm_line = line
                    elif 'MD Exploration Steps:' in line:
                        # This is a TQDM line, keep only the latest
                        last_tqdm_line = line
                    else:
                        # Not a TQDM line - keep it
                        # But first, add the last TQDM line if we have one
                        if last_tqdm_line and last_tqdm_line not in filtered_lines:
                            filtered_lines.append(last_tqdm_line)
                            last_tqdm_line = None
                        
                        # Keep important log lines
                        if line.strip():  # Non-empty lines
                            filtered_lines.append(line)
                
                # Add final TQDM line if exists
                if last_tqdm_line:
                    filtered_lines.append(last_tqdm_line)
                
                clean_content = '\n'.join(filtered_lines)
                self.logs = clean_content

            status = self.job_manager.get_status(self._last_job_id)
            if status == "running":
                import re
                
                # Extract current stage from log content
                if "Step 1: Structure Generation" in content or "Generating initial structures" in content:
                    self.status_message = "Generating initial structures..."
                    self.progress_value = max(self.progress_value, 25)
                elif "Minimization complete" in content or "Step 2: Exploration" in content:
                    self.status_message = "Running MD exploration..."
                    self.progress_value = max(self.progress_value, 50)
                elif "Step 3: Sampling" in content:
                    self.status_message = "Sampling structures..."
                    self.progress_value = max(self.progress_value, 75)
                elif "Step 4: Saving" in content:
                    self.status_message = "Saving results..."
                    self.progress_value = max(self.progress_value, 90)
                
                # Extract MD progress from TQDM lines
                md_prog_match = re.findall(r'MD Exploration Steps:\s+(\d+)%\|[^|]+\|\s+(\d+)/(\d+)', content)
                if md_prog_match:
                    percent, current, total = md_prog_match[-1]
                    try:
                        # Use the percentage from TQDM directly
                        perc = float(percent) / 100.0
                        overall_perc = 50 + (perc * 25)  # MD is 50-75% of overall progress
                        self.progress_value = max(self.progress_value, int(overall_perc))
                        self.status_message = f"MD Exploration: {current}/{total} steps ({percent}%)"
                    except (ValueError, ZeroDivisionError):
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
        # run_btn = pn.widgets.Button(name="Run Pipeline", button_type="primary")
        # run_btn.on_click(self.vm.run_pipeline)

        # stop_btn = pn.widgets.Button(name="Stop", button_type="danger")
        # stop_btn.on_click(self.vm.stop_pipeline)

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

        # Auto-scroll to bottom when logs update
        log_area.jscallback(value="""
            var el = document.getElementById(cb_obj.id);
            if (el) {
                if (el.shadowRoot) {
                    var ta = el.shadowRoot.querySelector('textarea');
                    if (ta) ta.scrollTop = ta.scrollHeight;
                } else if (el.tagName === 'TEXTAREA') {
                    el.scrollTop = el.scrollHeight;
                } else {
                    var ta = el.querySelector('textarea');
                    if (ta) ta.scrollTop = ta.scrollHeight;
                }
            }
        """)

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
                pn.Param(self.vm.param.output_dir, name="Output Directory"),
                pn.Param(self.vm.param.system_type),
                pn.Param(self.vm.param.elements_input),
                pn.Param(self.vm.param.seed, widgets={'seed': {'type': pn.widgets.EditableIntSlider}}),
                self.vm.system_settings_panel,
                
                pn.Card(
                    pn.Param(self.vm.param.n_initial_structures, widgets={'n_initial_structures': {'type': pn.widgets.EditableIntSlider}}),
                    pn.Param(self.vm.param.n_surface_samples, widgets={'n_surface_samples': {'type': pn.widgets.EditableIntSlider}}),
                    pn.Param(self.vm.param.supercell_input),
                    pn.Param(self.vm.param.pbc_input),
                    pn.Param(self.vm.param.rattle_std, widgets={'rattle_std': {'type': pn.widgets.EditableFloatSlider, 'format': '0.000'}}),
                    pn.Row(pn.Param(self.vm.param.vol_scale_min, name="Vol Min"), pn.Param(self.vm.param.vol_scale_max, name="Vol Max")),
                    title="Advanced System Settings", collapsed=True
                ),

                "### Advanced Physics",
                pn.Param(self.vm.param.vacancy_concentration, widgets={'vacancy_concentration': {'type': pn.widgets.EditableFloatSlider, 'format': '0.000'}}),
                pn.Param(self.vm.param.max_atoms, widgets={'max_atoms': {'type': pn.widgets.EditableIntSlider}}),
                pn.Param(self.vm.param.min_distance, widgets={'min_distance': {'type': pn.widgets.EditableFloatSlider, 'format': '0.00'}}),
                
                pn.Card(
                    pn.Param(self.vm.param.min_density, widgets={'min_density': {'type': pn.widgets.EditableFloatSlider, 'format': '0.00'}}),
                    pn.Param(self.vm.param.r_cut, widgets={'r_cut': {'type': pn.widgets.EditableFloatSlider, 'format': '0.0f'}}),
                    pn.Param(self.vm.param.min_cell_length_factor, widgets={'min_cell_length_factor': {'type': pn.widgets.EditableFloatSlider, 'format': '0.00'}}),
                    title="Extra Constraints", collapsed=True
                ),

                "### MD Exploration",
                pn.Param(self.vm.param.model_name),
                pn.Param(self.vm.param.device, widgets={'device': pn.widgets.RadioButtonGroup}),
                pn.Param(self.vm.param.temperature_mode, widgets={'temperature_mode': pn.widgets.RadioButtonGroup}),
                self.vm.exploration_settings_panel,
                pn.Param(self.vm.param.timestep, widgets={'timestep': {'type': pn.widgets.EditableFloatSlider, 'format': '0.0f'}}),
                pn.Param(self.vm.param.steps, widgets={'steps': {'type': pn.widgets.EditableIntSlider}}),
                pn.Param(self.vm.param.snapshot_interval, widgets={'snapshot_interval': {'type': pn.widgets.EditableIntSlider}}),
                pn.Param(self.vm.param.ensemble),
                self.vm.ensemble_settings_panel,
                pn.Param(self.vm.param.mc_enabled, widgets={'mc_enabled': pn.widgets.Toggle}),
                self.vm.mc_settings_panel,
                pn.Param(self.vm.param.zbl_enabled, widgets={'zbl_enabled': pn.widgets.Toggle}, name="Enable ZBL"),
                self.vm.zbl_settings_panel,
                "### Sampling",
                pn.Param(self.vm.param.sampling_strategy),
                pn.Param(self.vm.param.sampling_descriptor),
                pn.Param(self.vm.param.n_samples, widgets={'n_samples': {'type': pn.widgets.EditableIntSlider}}),
                
                pn.Row(
                    pn.widgets.Button(
                        name="Run Job", 
                        button_type="primary", 
                        on_click=self.vm.run_pipeline,
                        disabled=self.vm.param.progress_active,
                        width=150
                    ),
                    pn.widgets.Button(
                        name="Stop Job", 
                        button_type="danger", 
                        on_click=self.vm.stop_pipeline,
                        disabled=self.vm.param.progress_idle,
                        width=150
                    ),
                    pn.indicators.LoadingSpinner(value=self.vm.param.progress_active, width=30, height=30, align='center')
                ),
                progress_bar,
                pn.Param(self.vm.param.status_message, widgets={'status_message': {'type': pn.widgets.StaticText, 'styles': {'color': 'black', 'font-weight': 'bold'}}}),
            ),
            pn.Column(
                "## Logs",
                log_area
            )
        )
