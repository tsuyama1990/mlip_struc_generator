# NNP Generation Tutorials

This directory contains tutorials for beginners to get started with the NNP Generation Pipeline.

## Prerequisites

Ensure you have installed the project dependencies as described in the main `README.md`.

## Tutorials

### 1. Fe-Pt Alloy System (`01_alloy_fe_pt`)

This tutorial demonstrates how to generate structures for a simple Fe-Pt alloy system.

**Configuration:** `tutorials/01_alloy_fe_pt/config.yaml`

**To Run:**

From the root of the repository:

```bash
python main.py --config-path $(pwd)/tutorials/01_alloy_fe_pt --config-name config
```

### 2. SiO2 Ionic System (`02_ionic_sio2`)

This tutorial covers generating structures for an ionic system (Silicon Dioxide), specifying oxidation states.

**Configuration:** `tutorials/02_ionic_sio2/config.yaml`

**To Run:**

From the root of the repository:

```bash
python main.py --config-path $(pwd)/tutorials/02_ionic_sio2 --config-name config
```

## Note on MD Exploration and Calculators

These tutorials are configured to use the `mace` calculator for Molecular Dynamics (MD) exploration.
If `mace-torch` is not installed (which is an optional dependency), the MD exploration step will gracefully fail/skip, and the pipeline will proceed with the initially generated structures.
To enable full MD exploration, install the optional ML calculators:

```bash
uv pip install "nnp_gen[ml-calculators]"
```
