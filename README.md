# NNP Generation Pipeline

## Overview

This project provides a pipeline for generating dataset structures for Neural Network Potentials (NNP).
It supports Alloy, Ionic, Covalent, and Molecular systems.
Key features:
- Structure generation using `ase`, `pymatgen`, `icet`, `pyxtal`.
- Physics-based validation (density, atomic distance).
- MD exploration using `ase` (Langevin dynamics).
- Sampling strategies: Random, FPS (Farthest Point Sampling) with SOAP descriptors.
- Web UI dashboard for configuration and visualization.

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

### Web UI

To launch the dashboard:
```bash
python main_gui.py
```
Open your browser at `http://localhost:5006`.

## Development

### Running Tests

```bash
uv run pytest
```

### Project Structure

- `src/nnp_gen`: Source code.
  - `core`: Core logic, interfaces, configuration models.
  - `generators`: Structure generators (Alloy, Ionic, etc.).
  - `explorers`: MD exploration engine.
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
