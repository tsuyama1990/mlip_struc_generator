import panel as pn
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
import asyncio
from typing import Optional

from logic.config_manager import ConfigManager, SimulationConfig
from logic.simulation_mock import MockSimulation
from logic.analysis import AnalysisData

# Initialize Panel extension with Plotly
pn.extension('plotly')

class Dashboard:
    def __init__(self):
        # Logic Components
        self.config_manager = ConfigManager()
        self.simulation = MockSimulation()
        self.analysis = AnalysisData()

        # Load initial config
        try:
            self.current_config = self.config_manager.load_config()
        except Exception as e:
            pn.state.notifications.error(f"Error loading config: {e}")
            self.current_config = SimulationConfig(composition="FeNi")

        # --- Widgets: Configuration ---
        self.w_composition = pn.widgets.Select(
            name="Composition",
            options=["FeNi", "CuZr", "AlLi", "SiGe"],
            value=self.current_config.composition
        )
        self.w_system_type = pn.widgets.Select(
            name="System Type",
            options=["Alloy", "Covalent Crystal", "Ionic", "Molecule"],
            value=self.current_config.system_type
        )
        self.w_temperature = pn.widgets.FloatSlider(
            name="Temperature (K)",
            start=300, end=2000, step=10,
            value=self.current_config.temperature
        )
        self.w_atom_limit = pn.widgets.IntInput(
            name="Atom Limit",
            start=10, end=10000,
            value=self.current_config.atom_limit
        )
        self.w_run_button = pn.widgets.Button(
            name="Run Simulation",
            button_type="primary"
        )
        self.w_run_status = pn.widgets.StaticText(value="Ready")

        # --- Widgets: Visualization ---
        self.w_filter_extracted = pn.widgets.IntSlider(
            name="Limit Extracted Structures",
            start=1, end=100, value=20
        )
        self.w_player = pn.widgets.Player(
            name="Structure Player",
            start=0, end=10,
            value=0, loop_policy="loop", interval=500
        )

        # --- Panes (Placeholders) ---
        self.pane_scatter = pn.pane.Plotly()
        self.pane_structure = pn.pane.Plotly()
        self.pane_coverage = pn.pane.Plotly()

        # Data State
        self.pca_data = self.analysis.get_pca_data()
        self.extracted_data = self.pca_data[self.pca_data['is_extracted']].reset_index(drop=True)

        # Update Player range based on data
        self.w_player.end = len(self.extracted_data) - 1

        # --- Callbacks ---
        self.w_run_button.on_click(self.run_simulation_callback)

        # Linking visualization updates
        # Update scatter plot when filter changes
        self.w_filter_extracted.param.watch(self.update_scatter_plot, 'value')

        # We need to capture click events on the scatter plot.
        # Panel's Plotly pane exposes 'click_data'.
        self.pane_scatter.param.watch(self.on_scatter_click, 'click_data')

        # Watch player value to update structure
        self.w_player.param.watch(self.on_player_change, 'value')

        # Initial Render
        self.update_scatter_plot(None)
        self.update_coverage_plot()
        # Show first extracted structure by default
        if not self.extracted_data.empty:
            self.update_structure_view(self.extracted_data.iloc[0]['structure_id'])

    async def run_simulation_callback(self, event):
        """
        Handles the 'Run' button click asynchronously.
        """
        self.w_run_button.disabled = True
        self.w_run_status.value = "Running..."

        try:
            # 1. Update Config Object from Widgets
            new_config = SimulationConfig(
                composition=self.w_composition.value,
                system_type=self.w_system_type.value,
                temperature=self.w_temperature.value,
                atom_limit=self.w_atom_limit.value
            )

            # 2. Save Config
            self.config_manager.save_config(new_config)

            # 3. Run Simulation (Mock) in Executor
            # This prevents blocking the Tornado event loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.simulation.run, new_config.model_dump())

            self.w_run_status.value = "Simulation Complete!"
            pn.state.notifications.success("Simulation finished successfully.")

        except Exception as e:
            self.w_run_status.value = "Error"
            pn.state.notifications.error(f"Simulation failed: {e}")
        finally:
            self.w_run_button.disabled = False

    def update_scatter_plot(self, event=None):
        """
        Generates the PCA scatter plot.
        """
        # Create a copy to avoid SettingWithCopy warnings on self.pca_data
        df = self.pca_data.copy()

        # Apply filter: Only top N extracted structures are marked as "Extracted"
        limit = self.w_filter_extracted.value

        # Identify indices of extracted structures
        extracted_indices = df[df['is_extracted']].index

        # If we have more extracted than limit, demote the rest to "Others" (or a different category if needed)
        # For this requirement, we'll just treat them as "Others" effectively removing them from the "Extracted" group visual
        if len(extracted_indices) > limit:
            indices_to_hide = extracted_indices[limit:]
            df.loc[indices_to_hide, 'is_extracted'] = False

        # Map boolean is_extracted to symbol
        # Plotly Express symbol map
        df['type'] = df['is_extracted'].map({True: 'Extracted', False: 'Others'})

        fig = px.scatter(
            df,
            x="pc1", y="pc2",
            color="composition",
            symbol="type",
            symbol_map={'Extracted': 'circle', 'Others': 'x'},
            hover_data=["structure_id"],
            title="PCA Structure Map"
        )
        fig.update_layout(clickmode='event+select')

        self.pane_scatter.object = fig

    def update_coverage_plot(self):
        """
        Generates the coverage plot.
        """
        df = self.analysis.get_coverage_data()
        fig = px.line(
            df,
            x="num_extracted", y="coverage_rate",
            title="Coverage Rate vs Extracted Structures"
        )
        self.pane_coverage.object = fig

    def update_structure_view(self, structure_id: str):
        """
        Updates the 3D structure viewer for a given ID.
        """
        df = self.analysis.get_structure(structure_id)

        if df.empty:
            self.pane_structure.object = None
            return

        # Map elements to colors/sizes (simple mapping)
        # We can use discrete color map in plotly

        fig = px.scatter_3d(
            df,
            x='x', y='y', z='z',
            color='element',
            title=f"Structure: {structure_id}",
            size_max=20,
            opacity=1.0
        )
        # Enforce fixed aspect ratio for atoms
        fig.update_layout(scene_aspectmode='data')

        self.pane_structure.object = fig

    def on_scatter_click(self, event):
        """
        Callback when a point on the scatter plot is clicked.
        """
        # event.new contains the click data dict from Plotly
        click_data = event.new
        if click_data and 'points' in click_data:
            point = click_data['points'][0]
            # customdata is not explicitly set in px.scatter unless we use hover_data carefully,
            # but we can try to find the index or use point_number/curve_number if data is aligned.
            # px.scatter usually keeps data order if not sorted.

            # Robust way: retrieve structure_id from customdata if available,
            # or match by x/y coords.
            # Let's see... px.scatter adds hover_data to customdata?
            # Actually, standard way with px is pointIndex.

            try:
                point_index = point['pointIndex']
                # But wait, if we have multiple traces (due to color/symbol grouping),
                # pointIndex is relative to the trace.
                # Use customdata.

                # To ensure customdata is present, I need to check how I built the fig.
                # px.scatter(..., hover_data=["structure_id"]) puts structure_id in customdata[0] usually.

                if 'customdata' in point:
                     structure_id = point['customdata'][0]
                     self.update_structure_view(structure_id)

                     # Also try to sync player if this ID is in extracted list
                     extracted_idx = self.extracted_data.index[self.extracted_data['structure_id'] == structure_id].tolist()
                     if extracted_idx:
                         self.w_player.value = extracted_idx[0]

            except Exception as e:
                print(f"Error handling click: {e}")

    def on_player_change(self, event):
        """
        Callback when the player advances.
        """
        idx = event.new
        if 0 <= idx < len(self.extracted_data):
            struct_id = self.extracted_data.iloc[idx]['structure_id']
            self.update_structure_view(struct_id)

    def layout(self):
        """
        Constructs the main dashboard layout.
        """
        # Tab 1: Configuration
        config_tab = pn.Column(
            pn.pane.Markdown("## MD Simulation Configuration"),
            self.w_composition,
            self.w_system_type,
            self.w_temperature,
            self.w_atom_limit,
            pn.layout.Divider(),
            self.w_run_button,
            self.w_run_status
        )

        # Tab 2: Visualization
        # Top: Scatter | Coverage
        # Bottom: 3D View | Player

        viz_tab = pn.Row(
            pn.Column(
                self.pane_scatter,
                self.w_filter_extracted,
                self.pane_coverage,
                width=600
            ),
            pn.Column(
                self.pane_structure,
                self.w_player,
                width=600
            )
        )

        tabs = pn.Tabs(
            ("Configuration", config_tab),
            ("Visualization", viz_tab)
        )

        return pn.template.MaterialTemplate(
            title="MD Simulation & Analysis Dashboard",
            main=[tabs]
        )

# Serve the app
if __name__.startswith("bokeh"):
    dashboard = Dashboard()
    dashboard.layout().servable()

if __name__ == "__main__":
    # For running locally via 'python dashboard/app.py'
    dashboard = Dashboard()
    pn.serve(
        dashboard.layout(),
        port=5006,
        show=False,
        address="0.0.0.0",
        allow_websocket_origin=["*"]
    )
