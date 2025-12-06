# MLIP Structure Generator (nnp_gen)

A robust pipeline for generating training datasets for Neural Network Potentials (NNP/MLIP).

## Features
- **Structure Generation**: Support for Alloys, Ionic crystals, Covalent systems, and Molecules.
- **Physics Validation**: Enforces density, minimum distance, and cell size constraints.
- **MD Exploration**: Runs molecular dynamics (via MACE/SevenNet or fallback) to explore phase space.
- **Sampling**: Diverse sampling using FPS (SOAP descriptors) or Random selection.
- **Database**: Stores structures and provenance metadata in ASE Database (SQLite).

## Installation

Requires Python >= 3.10 and < 3.14.

```bash
pip install .
```

For heavy dependencies (optional but recommended for full functionality):
```bash
pip install .[all]
```

## Usage

Run the pipeline using the CLI:

```bash
python main.py
```

### Configuration
The pipeline is configured via `config/config.yaml`. You can override parameters from command line using Hydra syntax:

```bash
# Run for a different system (e.g. Ionic)
python main.py system.type=ionic system.elements=[Li,F] system.oxidation_states="{Li:1, F:-1}"

# Change exploration settings
python main.py exploration.steps=1000 exploration.temperature=500
```

## Architecture
- `src/nnp_gen/core`: Core physics logic, config models, and interfaces.
- `src/nnp_gen/generators`: Structure generators (Alloy, Ionic, Covalent, Molecule).
- `src/nnp_gen/explorers`: MD engine.
- `src/nnp_gen/samplers`: Sampling logic (FPS, Random).
- `src/nnp_gen/pipeline`: Orchestration.

## Testing
Run tests with:
```bash
pytest tests
```
