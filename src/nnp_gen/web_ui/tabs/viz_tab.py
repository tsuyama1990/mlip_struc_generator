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
    selected_idx = param.Integer(default=0, doc="Index of selected structure")
    pca_plot = param.ClassSelector(class_=figure, is_instance=True)
    viewer_html = param.String(default="")

    status_msg = param.String(default="")
    warning_msg = param.String(default="") # For truncation warnings

    def __init__(self, **params):
        super().__init__(**params)
        self.job_manager = JobManager()
        self._cache_pca = {}
        self.current_db_path = None

        # Dedicated executor for UI tasks to avoid blocking main loop
        # and to avoid blocking the JobManager's single worker
        self.ui_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        # Init plots
        self._init_plot()

    def _init_plot(self):
        p = figure(title="PCA of Structures", tools="pan,wheel_zoom,box_select,reset,save",
                   active_scroll="wheel_zoom")
        self.pca_source = ColumnDataSource(data=dict(x=[], y=[], color=[], idx=[]))

        renderer = p.scatter('x', 'y', color='color', source=self.pca_source, size=8)

        # Add TapTool
        tap = TapTool(renderers=[renderer])
        p.add_tools(tap)
        p.add_tools(HoverTool(tooltips=[("ID", "@idx")]))

        self.pca_plot = p

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
        self.load_db(db_path)

    def load_db(self, db_path: str):
        if not os.path.exists(db_path):
            self.status_msg = f"DB not found: {db_path}"
            return

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
        """
        Offload PCA computation to thread pool.
        """
        if not self.structures:
            return

        self.status_msg = "Calculating PCA in background..."

        # Prepare data in main thread
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
            self.status_msg = "No descriptors found for PCA."
            return

        X = np.array(descriptors)
        X = np.nan_to_num(X)

        # Capture current token
        token = self.current_db_path

        # Submit to executor
        future = self.ui_executor.submit(self._run_pca, X, valid_indices, token)
        future.add_done_callback(self._on_pca_complete)

    def _run_pca(self, X, valid_indices, token):
        """Runs in worker thread."""
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        return coords, valid_indices, token

    def _on_pca_complete(self, future):
        """Callback when PCA is done."""
        try:
            coords, valid_indices, token = future.result()

            # CRITICAL CHECK: Discard result if user has switched DBs
            if token != self.current_db_path:
                return

            # Schedule UI update on main thread/loop
            if pn.state.curdoc:
                pn.state.execute(lambda: self._update_pca_plot(coords, valid_indices))
            else:
                # Fallback if no server context (e.g. testing)
                self._update_pca_plot(coords, valid_indices)

        except Exception as e:
            # Schedule error update
             if pn.state.curdoc:
                pn.state.execute(lambda: setattr(self, 'status_msg', f"PCA Failed: {e}"))

    def _update_pca_plot(self, coords, valid_indices):
        """Update Bokeh source. Must run on main thread/with doc lock."""
        colors = []
        for i in valid_indices:
            # Check bounds just in case metadata list changed (unlikely given flow)
            if i < len(self.metadata_list):
                meta = self.metadata_list[i]
                colors.append("red" if meta["is_sampled"] else "blue")
            else:
                colors.append("gray")

        self.pca_source.data = dict(
            x=coords[:, 0],
            y=coords[:, 1],
            color=colors,
            idx=valid_indices
        )
        self.status_msg = "PCA Updated."

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

        try:
            pn.state.add_periodic_callback(self.vm.update_job_list, period=2000)
        except RuntimeError:
             pass

    def view(self):
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
                    pn.Param(self.vm.param.warning_msg, widgets={'warning_msg': {'type': pn.widgets.StaticText, 'styles': {'color': 'red', 'font-weight': 'bold'}}}),
                    "### PCA Projection",
                    pn.pane.Bokeh(self.vm.pca_plot)
                ),
                pn.Column(
                    "### 3D Viewer",
                    pn.pane.HTML(self.vm.param.viewer_html, height=400, width=600),
                    pn.Param(self.vm.param.selected_idx)
                )
            )
        )
