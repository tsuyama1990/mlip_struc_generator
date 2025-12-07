# NNP Generation Pipeline

## Overview

This project provides a pipeline for generating dataset structures for Neural Network Potentials (NNP).
It supports Alloy, Ionic, Covalent, and Molecular systems, as well as Adsorbates on Surfaces.
Key features:
- **Structure Generation:**
  - `ase`, `pymatgen`, `icet`, `pyxtal` integration.
  - **File Loading:** Import custom structures (CIF, XYZ) with optional repetition and vacancy injection.
  - **Vacancies:** Inject random vacancies (0-25%) into Ionic, Alloy, and Covalent systems.
- **Physics-Based Validation:**
  - Strict checks for density, atomic overlap, and vacuum detection.
- **Hybrid MD/MC Engine:**
  - **Auto Ensemble Switching:** Automatically selects NVT (Langevin) for slabs/vacuum systems and NPT for bulk.
  - **Monte Carlo Moves:** Interleave MD with MC swaps or "Smart Rattle" (Vacancy Hopping).
  - **Charge Safety:** Prevents aliovalent swaps in ionic systems to avoid Coulomb explosions.
- **Sampling Strategies:** Random, FPS (Farthest Point Sampling) with SOAP descriptors.
- **Web UI:** Panel-based dashboard for configuration and visualization.

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (Recommended)

### Setup

1.  **Install uv** (if not installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Sync Environment**:
    ```bash
    uv sync
    ```
    This will create a virtual environment and install all dependencies.

3.  **Activate Environment**:
    ```bash
    source .venv/bin/activate
    ```

## Usage

### Command Line Interface (CLI)

The pipeline is managed via `hydra`. The main entry point is `main.py`.

Run with default configuration:
```bash
python main.py
```

Override configuration parameters:
```bash
python main.py system=alloy system.elements=["Cu","Au"] exploration.steps=100
```

#### Example: Hybrid MD/MC with File Loading
```yaml
# config.yaml snippet
system:
  type: user_file
  path: "./start.cif"
  repeat: 2
  vacancy_concentration: 0.05

exploration:
  method: hybrid_mc_md
  ensemble: AUTO
  steps: 5000
  mc_config:
    enabled: true
    strategy: [SWAP, VACANCY_HOP]
    swap_interval: 100
    swap_pairs: [["Fe", "Ni"]]
```

### Web UI

To launch the dashboard:
```bash
python main_gui.py
```
Open your browser at `http://localhost:5006`.

## Features Logic

### Smart Doping & MC
- **Vacancy Hop (Smart Rattle):** Moves a random atom by ~2.5Å in a random direction. If it lands in a void (low energy), it is accepted.
- **Charge Safety:** When swapping ions (e.g. Na <-> Cl), the engine checks charges. Mismatches > 0.1e are rejected by default.

### Auto Ensemble
- **Bulk:** Uses NPT (Nose-Hoover) to allow cell relaxation.
- **Slab/Vacuum:** Uses NVT (Langevin) to prevent vacuum expansion artifacts.

## Development

### Running Tests

```bash
uv run pytest
```

### Project Structure

- `src/nnp_gen`: Source code.
  - `core`: Core logic, interfaces, configuration models.
  - `generators`: Structure generators (Alloy, Ionic, etc.).
  - `explorers`: MD exploration engine (Hybrid MD/MC).
  - `samplers`: Sampling strategies (FPS, Random).
  - `pipeline`: Pipeline orchestration.
  - `web_ui`: Panel-based dashboard.
- `config`: Hydra configuration files.
- `tests`: Unit and integration tests.

## Dependency Management

Dependencies are managed in `pyproject.toml`.
To add a new dependency:
```bash
uv add <package_name>
```

To update dependencies:
```bash
uv lock
uv sync
```
