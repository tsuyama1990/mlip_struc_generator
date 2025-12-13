import os
import ase.db
import param
import panel as pn
import numpy as np
import threading
import concurrent.futures
import tempfile
from typing import List, Optional
from ase import Atoms
from ase.io import write
from io import StringIO
from sklearn.decomposition import PCA
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, TapTool, HoverTool
from scipy.spatial import distance
from omegaconf import OmegaConf

from nnp_gen.web_ui.job_manager import JobManager, JobStatus
from nnp_gen.web_ui.components.structure_viewer import generate_3dmol_html, get_3dmol_header
from nnp_gen.samplers.selector import DescriptorManager

class VizViewModel(param.Parameterized):
    # Job Selection
    job_selector = param.Selector(objects=[], doc="Select Completed Job")
    external_db_path = param.String(default="", doc="External DB Path")

    # Data State
    structures = param.List(default=[], doc="Loaded structures")
    metadata_list = param.List(default=[], doc="Corresponding metadata")

    # Visualization State
    # Visualization State
    selected_idx = param.Integer(default=0, doc="Index of selected structure")
    pca_plot = param.ClassSelector(class_=figure, is_instance=True)
    entropy_plot = param.ClassSelector(class_=figure, is_instance=True)
    viewer_html = param.String(default="")

    # FPS & Advanced Viz
    fps_n_samples = param.Integer(default=10, bounds=(2, 5000), doc="Number of FPS samples")
    pca_x_axis = param.Selector(objects=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], default='PC1')
    pca_y_axis = param.Selector(objects=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], default='PC2')
    
    # Export Config
    export_filename = param.String(default="selected_fps.db", doc="Filename for export")
    is_computing = param.Boolean(default=False, doc="Is background task running")
    status_msg = param.String(default="")
    warning_msg = param.String(default="") # For truncation warnings

    def __init__(self, **params):
        super().__init__(**params)
        self.job_manager = JobManager()
        self._cache_pca = {} # Stores (coords, valid_indices) where coords is (N, 5)
        self._cache_fps = {} # Stores {n_samples: (selected_indices, entropy_curve)}
        self.current_db_path = None

        # Dedicated executor for UI tasks to avoid blocking main loop
        # and to avoid blocking the JobManager's single worker
        self.ui_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        # Init plots
        self._init_plot()

    def _init_plot(self):
        # PCA Plot
        p = figure(title="PCA of Structures", tools="pan,wheel_zoom,box_select,reset,save",
                   active_scroll="wheel_zoom", width=500, height=400)
        self.pca_source = ColumnDataSource(data=dict(x=[], y=[], color=[], idx=[], alpha=[]))

        renderer = p.scatter('x', 'y', color='color', source=self.pca_source, size=8, alpha='alpha')

        # Add TapTool
        tap = TapTool(renderers=[renderer])
        p.add_tools(tap)
        p.add_tools(HoverTool(tooltips=[("ID", "@idx")]))

        self.pca_plot = p

        # Entropy/Coverage Plot
        p2 = figure(title="Coverage (MaxMin Dist) vs N", tools="pan,wheel_zoom,reset,save",
                    active_scroll="wheel_zoom", width=500, height=300,
                    x_axis_label="Number of Samples", y_axis_label="MaxMin Dist")
        self.entropy_source = ColumnDataSource(data=dict(n=[], dist=[]))
        p2.line('n', 'dist', source=self.entropy_source, line_width=2, color='green')
        p2.scatter('n', 'dist', source=self.entropy_source, size=6, color='green')
        p2.add_tools(HoverTool(tooltips=[("N", "@n"), ("Dist", "@dist")]))
        
        self.entropy_plot = p2

    def update_job_list(self):
        """Called periodically to refresh job list."""
        jobs = self.job_manager.get_all_jobs()
        completed = [j.job_id for j in jobs if j.status == JobStatus.COMPLETED]

        # Update selector objects
        self.param.job_selector.objects = completed

        # Auto-select newest if nothing selected or list grew
        if completed and (not self.job_selector or self.job_selector not in completed):
            self.job_selector = completed[0]

    @param.depends("job_selector", watch=True)
    def load_job_data(self):
        if not self.job_selector:
            return

        job = self.job_manager.get_job(self.job_selector)
        if not job:
            return

        db_path = os.path.join(job.output_dir, "dataset.db")
        if not os.path.exists(db_path):
             # Fallback to checkpoints if job incomplete/failed
             ckpt_path = os.path.join(job.output_dir, "checkpoints.db")
             if os.path.exists(ckpt_path):
                 self.status_msg = "Job incomplete. Loading checkpoints..."
                 db_path = ckpt_path
             else:
                 self.status_msg = f"No DB found in {job.output_dir}"
                 return
        
        self.load_db(db_path)

    def load_db(self, db_path: str):

        self.current_db_path = db_path
        self.status_msg = f"Loading {db_path}..."
        self.warning_msg = ""

        try:
            structs = []
            metas = []
            LIMIT = 5000

            with ase.db.connect(db_path) as db:
                count = db.count()
                if count > LIMIT:
                    self.warning_msg = f"Warning: Dataset has {count} structures. Displaying first {LIMIT} only."

                # Fetch limited rows
                rows = list(db.select(limit=LIMIT))
                for row in rows:
                    atoms = row.toatoms()
                    # Restore descriptor if in data
                    if hasattr(row, 'data') and 'descriptor' in row.data:
                        atoms.info['descriptor'] = row.data['descriptor']

                    structs.append(atoms)
                    # Extract minimal metadata for coloring
                    is_sampled = row.get("is_sampled", False)
                    metas.append({"is_sampled": is_sampled})

            self.structures = structs
            self.metadata_list = metas
            self.status_msg = f"Loaded {len(structs)} structures."

            # Update PCA asynchronously
            self.compute_pca_async()

            # Select first
            if structs:
                self.selected_idx = 0
                self.update_viewer()

        except Exception as e:
            self.status_msg = f"Error loading DB: {e}"

    def compute_pca_async(self):
        if not self.current_db_path:
            self.status_msg = "No database loaded."
            return

        # Explicitly managing descriptor calculation state
        if self.current_db_path not in self._cache_pca:
            # Check if we need to calculate descriptors
            needs_calc = False
            if self.structures:
                # simple check first element
                if self.structures[0].info.get("descriptor") is None:
                    needs_calc = True
            
            if needs_calc:
                self.status_msg = "Calculating descriptors (SOAP/RDF)..."
                # Determine job output dir for config
                base_dir = os.path.dirname(self.current_db_path)
                if self.job_selector:
                    job = self.job_manager.get_job(self.job_selector)
                    if job:
                        base_dir = job.output_dir
                
                token = self.current_db_path
                doc = pn.state.curdoc
                future = self.ui_executor.submit(self._calculate_and_run_pca, self.structures, base_dir, token)
                future.add_done_callback(lambda f: self._on_pca_complete(f, doc))
                return

            # If we have structures but no descriptors and needs_calc is False (e.g. empty list),
            # or if logic falls through (e.g. mixed state), try standard path
            if self.structures and not any(atoms.info.get("descriptor") is not None for atoms in self.structures):
                 self.status_msg = "Cannot run FPS: No descriptors found in DB."
                 return

            self.status_msg = "Calculating PCA..."
            # Standard path: descriptors already in atoms
            X, valid_indices = self._extract_descriptors_from_atoms()
            if X is None:
                 return # status updated in helper

            # Capture current token & doc
            token = self.current_db_path
            doc = pn.state.curdoc

            future = self.ui_executor.submit(self._run_pca, X, valid_indices, token)
            future.add_done_callback(lambda f: self._on_pca_complete(f, doc))
            return

        # Already cached
        pass

    def _extract_descriptors_from_atoms(self):
        descriptors = []
        valid_indices = []
        for i, atoms in enumerate(self.structures):
            desc = atoms.info.get("descriptor")
            if desc is not None and len(desc) > 0:
                d = np.array(desc)
                if d.ndim == 2:
                    d = d.mean(axis=0)
                descriptors.append(d)
                valid_indices.append(i)
        
        if not descriptors:
            self.status_msg = "No descriptors found."
            return None, None
            
        X = np.array(descriptors)
        X = np.nan_to_num(X)
        return X, valid_indices

    def _calculate_and_run_pca(self, structures, base_dir, token):
        """Worker: Calculate descriptors on-the-fly then run PCA."""
        # 1. Load config for params
        rcut, nmax, lmax = 5.0, 4, 3 # defaults
        config_path = os.path.join(base_dir, "config.yaml")
        if os.path.exists(config_path):
            try:
                cfg = OmegaConf.load(config_path)
                if 'sampling' in cfg:
                    params = cfg.sampling.get('descriptor_params', {})
                    rcut = params.get('rcut', 5.0)
                    nmax = params.get('nmax', 4)
                    lmax = params.get('lmax', 3)
            except Exception as e:
                print(f"Failed to load config for descriptors: {e}")
        
        # 2. Calculate
        mgr = DescriptorManager(rcut=rcut, nmax=nmax, lmax=lmax)
        features = mgr.calculate(structures) 
        
        # 3. Store back in atoms.info
        # Note: This modifies the shared Atoms objects. Thread safety? 
        # Atoms are just objects in memory. GIL protects python ops.
        valid_indices = []
        descriptors = []
        for i, feat in enumerate(features):
            # Check for validity
            if feat is not None and feat.size > 0:
                structures[i].info['descriptor'] = feat
                descriptors.append(feat)
                valid_indices.append(i)
        
        X = np.array(descriptors)
        X = np.nan_to_num(X)
        
        # 4. Run PCA
        return self._run_pca(X, valid_indices, token)

    def _run_pca(self, X, valid_indices, token):
        """Runs in worker thread."""
        npc = min(5, X.shape[1], X.shape[0])
        pca = PCA(n_components=npc)
        coords = pca.fit_transform(X) # (N, npc)
        
        # Pad with zeros if less than 5 components
        if coords.shape[1] < 5:
            padding = np.zeros((coords.shape[0], 5 - coords.shape[1]))
            coords = np.hstack([coords, padding])
            
        return coords, valid_indices, token

    def _on_pca_complete(self, future, doc=None):
        """Callback when PCA is done."""
        try:
            coords, valid_indices, token = future.result()

            # CRITICAL CHECK: Discard result if user has switched DBs
            if token != self.current_db_path:
                return

            # ALWAYS Cache full 5-component coords
            self._cache_pca[token] = (coords, valid_indices)

            # Schedule UI update
            if doc:
                doc.add_next_tick_callback(lambda: self.update_plots())
            elif pn.state.curdoc:
                pn.state.execute(lambda: self.update_plots())
            else:
                self.update_plots()

        except Exception as e:
            msg = f"PCA Failed: {e}"
            if doc:
                doc.add_next_tick_callback(lambda: setattr(self, 'status_msg', msg))
            elif pn.state.curdoc:
                pn.state.execute(lambda: setattr(self, 'status_msg', msg))

    def update_plots(self):
        """Redraw plots based on current state (PCA axes, FPS selection)."""
        if not self.current_db_path or self.current_db_path not in self._cache_pca:
            return

        coords, valid_indices = self._cache_pca[self.current_db_path]
        
        # Determine axes
        x_idx = int(self.pca_x_axis[2]) - 1 # PC1 -> 0
        y_idx = int(self.pca_y_axis[2]) - 1 # PC2 -> 1
        
        xs = coords[:, x_idx]
        ys = coords[:, y_idx]

        # Determine coloring based on FPS
        selected_indices_set = set()
        
        # Check if FPS is computed for current N
        fps_key = (self.current_db_path, self.fps_n_samples)
        if fps_key in self._cache_fps:
            selected_indices_in_valid, entropy_curve = self._cache_fps[fps_key]
            
            # Map back to global indices
            for local_idx in selected_indices_in_valid:
                selected_indices_set.add(valid_indices[local_idx])
            
            # Update Entropy Plot
            ns, dists = zip(*entropy_curve)
            self.entropy_source.data = dict(n=ns, dist=dists)
        else:
            # Do NOT auto-run FPS on simple update; wait for user or explicit call
            pass

        colors = []
        alphas = []
        
        for idx in valid_indices:
            if idx in selected_indices_set:
                colors.append("red")
                alphas.append(1.0)
            else:
                colors.append("gray")
                alphas.append(0.3 if selected_indices_set else 0.8) # Dim others if selection active

        self.pca_source.data = dict(
            x=xs,
            y=ys,
            color=colors,
            idx=valid_indices,
            alpha=alphas
        )
        self.pca_plot.xaxis.axis_label = self.pca_x_axis
        self.pca_plot.yaxis.axis_label = self.pca_y_axis
        self.status_msg = "Plots Updated."

    def compute_fps_async(self):
        if not self.current_db_path:
            self.status_msg = "No database loaded."
            return

        if self.current_db_path not in self._cache_pca:
            # Check if we have structures but no descriptors
            if self.structures and not any(atoms.info.get("descriptor") for atoms in self.structures):
                 self.status_msg = "Cannot run FPS: No descriptors found in DB."
            else:
                 self.status_msg = "PCA data not ready. Please wait..."
            return
        
        _, valid_indices = self._cache_pca[self.current_db_path]
        if not valid_indices:
            self.status_msg = "Cannot run FPS: No valid descriptors found."
            return

        # Prepare for worker
        coords = self._cache_pca[self.current_db_path][0]
        token = self.current_db_path
        n = self.fps_n_samples
        
        self.is_computing = True
        self.status_msg = f"Computing FPS N={n}..."
        doc = pn.state.curdoc
        future = self.ui_executor.submit(self._run_fps, coords, n, token)
        future.add_done_callback(lambda f: self._on_fps_complete(f, doc))

    def _run_fps(self, points, n_samples, token):
        """
        Farthest Point Sampling.
        points: (N, D) array
        dataset size N
        """
        N = points.shape[0]
        n_samples = min(n_samples, N)
        
        selected_indices = [0] # Start with first point (could assume random or 0)
        distances = distance.cdist(points[selected_indices], points, metric='euclidean') # (1, N)
        min_distances = distances[0] # (N,) array of min dist to selected set
        
        entropy_curve = [] # (n, max_min_dist)
        
        # already have 1 sample. max_min is just max(min_distances) but irrelevant for 1 point coverage?
        # Actually coverage is defined by the point furthest away.
        entropy_curve.append((1, np.max(min_distances)))

        for _ in range(1, n_samples):
            # Select point with largest minimum distance to current set
            new_idx = np.argmax(min_distances)
            selected_indices.append(new_idx)
            
            # Update min distances: min(existing_min, dist_to_new_point)
            new_dists = distance.cdist([points[new_idx]], points, metric='euclidean')[0]
            min_distances = np.minimum(min_distances, new_dists)
            
            entropy_curve.append((len(selected_indices), np.max(min_distances)))
            
        return selected_indices, entropy_curve, n_samples, token

    def _on_fps_complete(self, future, doc=None):
        try:
            selected_indices, entropy_curve, n, token = future.result()
            
            if token != self.current_db_path:
                return
                
            self._cache_fps[(token, n)] = (selected_indices, entropy_curve)
            
            if doc:
                doc.add_next_tick_callback(lambda: self._finalize_fps())
            elif pn.state.curdoc:
                pn.state.execute(lambda: self._finalize_fps())
                
        except Exception as e:
            msg = f"FPS Error: {e}"
            print(msg)
            if doc:
                doc.add_next_tick_callback(lambda: setattr(self, 'status_msg', msg))
                doc.add_next_tick_callback(lambda: setattr(self, 'is_computing', False))
            elif pn.state.curdoc:
                pn.state.execute(lambda: setattr(self, 'status_msg', msg))
                pn.state.execute(lambda: setattr(self, 'is_computing', False))

    def _finalize_fps(self):
        self.is_computing = False
        self.update_plots()

    def export_selected_structures(self):
        """Export current FPS selection to a new DB."""
        if not self.current_db_path or self.current_db_path not in self._cache_pca:
             self.status_msg = "No data loaded."
             return

        fps_key = (self.current_db_path, self.fps_n_samples)
        if fps_key not in self._cache_fps:
            self.status_msg = "Please run FPS first."
            return

        selected_indices_in_valid, _ = self._cache_fps[fps_key]
        _, valid_indices = self._cache_pca[self.current_db_path]
        
        # Map back to real structure indices
        target_indices = [valid_indices[i] for i in selected_indices_in_valid]
        
        # Prepare subset in worker to avoid UI freeze
        self.is_computing = True
        self.status_msg = f"Exporting {len(target_indices)} structures..."
        
        # Get absolute path for export
        # Default to current job dir if available, else working dir
        if self.job_selector:
             # Try to find the job object to get output_dir
             job = self.job_manager.get_job(self.job_selector)
             base_dir = job.output_dir if job else os.getcwd()
        else:
             base_dir = os.getcwd()
             
        out_path = os.path.join(base_dir, self.export_filename)
        
        doc = pn.state.curdoc
        future = self.ui_executor.submit(self._run_export, target_indices, out_path)
        future.add_done_callback(lambda f: self._on_export_complete(f, doc))

    def _run_export(self, target_indices, out_path):
        try:
            # We need to access self.structures, but we are in a thread. 
            # self.structures is list of Atoms, which is picklable/shareable.
            # But safer to pass only the indices and have main thread pass data? 
            # Actually, threading shares memory. So self.structures is accessible.
            # BUT ase.db.connect might have issues if not careful.
            
            # Simple write:
            subset = [self.structures[i] for i in target_indices]
            
            with ase.db.connect(out_path, append=False) as db:
                for atoms in subset:
                    db.write(atoms, data=atoms.info.get("data", {}))
            
            return out_path
        except Exception as e:
            raise e

    def _on_export_complete(self, future, doc=None):
        try:
            out_path = future.result()
            msg = f"Exported to {out_path}"
        except Exception as e:
            msg = f"Export Failed: {e}"
            
        if doc:
            doc.add_next_tick_callback(lambda: setattr(self, 'status_msg', msg))
            doc.add_next_tick_callback(lambda: setattr(self, 'is_computing', False))
        elif pn.state.curdoc:
            pn.state.execute(lambda: setattr(self, 'status_msg', msg))
            pn.state.execute(lambda: setattr(self, 'is_computing', False))

    @param.depends("pca_x_axis", "pca_y_axis", watch=True)
    def on_viz_param_change(self):
        self.update_plots()

    @param.depends("selected_idx", watch=True)
    def update_viewer(self):
        if not self.structures or self.selected_idx >= len(self.structures):
            self.viewer_html = ""
            return

        atoms = self.structures[self.selected_idx]

        # Convert to XYZ string
        f = StringIO()
        write(f, atoms, format="xyz")
        xyz_data = f.getvalue()

        self.viewer_html = generate_3dmol_html(xyz_data)

class VizTab:
    def __init__(self):
        self.vm = VizViewModel()

        # Watcher for PCA selection
        def on_selection_change(attr, old, new):
            if new:
                selected_row_idx = new[0]
                real_idx = self.vm.pca_source.data['idx'][selected_row_idx]
                self.vm.selected_idx = real_idx

        self.vm.pca_source.selected.on_change("indices", on_selection_change)

    def view(self):
        # Register periodic callback when the view is loaded into the server
        if pn.state.curdoc:
             pn.state.onload(lambda: pn.state.add_periodic_callback(self.vm.update_job_list, period=2000))

        # File Input for external load
        file_input = pn.widgets.FileInput(accept=".db")

        def on_upload(event):
            if file_input.value:
                tmp_path = None
                try:
                    # Create temp file, write content, close it, but keep it on disk
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                        tmp.write(file_input.value)
                        tmp_path = tmp.name

                    # Load from the temp path
                    self.vm.load_db(tmp_path)
                finally:
                    # Ensure cleanup happens even if load_db fails
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError as e:
                            print(f"Error removing temp file: {e}")

        file_input.param.watch(on_upload, "value")

        return pn.Column(
            get_3dmol_header(),
            pn.Row(
                pn.Column(
                    "### Data Source",
                    pn.Param(self.vm.param.job_selector),
                    "Or Load External DB:",
                    file_input,
                    pn.Param(self.vm.param.status_msg, widgets={'status_msg': pn.widgets.StaticText}),
                    "### FPS Analysis",
                    pn.Row(
                        pn.widgets.IntSlider.from_param(self.vm.param.fps_n_samples, name="N Samples", width=200),
                        pn.widgets.Button(name="Run FPS", button_type="primary", on_click=lambda e: self.vm.compute_fps_async(), width=100),
                        pn.indicators.LoadingSpinner(value=self.vm.param.is_computing, width=30, height=30, align='center')
                    ),
                    "### Export Selection",
                    pn.Row(
                        pn.widgets.TextInput.from_param(self.vm.param.export_filename, name="Filename", placeholder="selected.db", width=200),
                        pn.widgets.Button(name="Export", button_type="success", on_click=lambda e: self.vm.export_selected_structures(), width=100)
                    ),
                    "### PCA Settings",
                    pn.Row(
                        pn.Param(self.vm.param.pca_x_axis, width=100),
                        pn.Param(self.vm.param.pca_y_axis, width=100)
                    ),
                    pn.pane.Bokeh(self.vm.pca_plot),
                    "### Descriptor Coverage",
                    pn.pane.Bokeh(self.vm.entropy_plot)
                ),
                pn.Column(
                    "### 3D Viewer",
                    pn.pane.HTML(self.vm.param.viewer_html, height=400, width=600),
                    pn.Param(self.vm.param.selected_idx)
                )
            )
        )
