import pandas as pd
import numpy as np
from pathlib import Path
import random

class AnalysisData:
    """
    Handles data loading and analysis for the dashboard.
    Uses dummy CSV/XYZ files for now.
    """
    def __init__(self, data_dir: str = "dashboard/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_dummy_data()

    def _ensure_dummy_data(self):
        """
        Generates dummy data if it doesn't exist.
        """
        pca_path = self.data_dir / "pca.csv"

        if not pca_path.exists():
            print("Generating dummy PCA data...")
            # Generate 10 points
            n_points = 10
            np.random.seed(42)

            data = {
                "pc1": np.random.normal(0, 1, n_points),
                "pc2": np.random.normal(0, 1, n_points),
                "composition": np.random.choice(["FeNi", "CuZr", "AlLi"], n_points),
                "is_extracted": np.random.choice([True, False], n_points, p=[0.2, 0.8]),
                "structure_id": [f"struct_{i}" for i in range(n_points)]
            }
            df = pd.DataFrame(data)
            df.to_csv(pca_path, index=False)

            # Generate dummy XYZ files for extracted structures
            structures_dir = self.data_dir / "structures"
            structures_dir.mkdir(exist_ok=True)

            for i in range(n_points):
                struct_id = f"struct_{i}"
                self._generate_dummy_xyz(structures_dir / f"{struct_id}.xyz")

    def _generate_dummy_xyz(self, filepath: Path):
        """
        Generates a simple cubic-like structure with random noise.
        """
        atoms = []
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    symbol = random.choice(["Fe", "Ni", "Cu", "Zr"])
                    # Add some noise to positions
                    px = x * 2.5 + random.uniform(-0.1, 0.1)
                    py = y * 2.5 + random.uniform(-0.1, 0.1)
                    pz = z * 2.5 + random.uniform(-0.1, 0.1)
                    atoms.append(f"{symbol} {px:.3f} {py:.3f} {pz:.3f}")

        with open(filepath, "w") as f:
            f.write(f"{len(atoms)}\n")
            f.write("Dummy structure\n")
            f.write("\n".join(atoms))

    def get_pca_data(self) -> pd.DataFrame:
        """
        Returns the PCA dataframe.
        """
        return pd.read_csv(self.data_dir / "pca.csv")

    def get_structure(self, structure_id: str) -> pd.DataFrame:
        """
        Returns XYZ coordinates for a structure as a DataFrame.
        Columns: [element, x, y, z]
        """
        filepath = self.data_dir / "structures" / f"{structure_id}.xyz"
        if not filepath.exists():
            # Fallback if file missing
            return pd.DataFrame(columns=["element", "x", "y", "z"])

        # Simple XYZ parser: skip first 2 lines
        try:
            df = pd.read_csv(
                filepath,
                skiprows=2,
                sep=r'\s+',
                names=["element", "x", "y", "z"]
            )
            return df
        except Exception as e:
            print(f"Error reading XYZ {filepath}: {e}")
            return pd.DataFrame(columns=["element", "x", "y", "z"])

    def get_coverage_data(self) -> pd.DataFrame:
        """
        Returns dummy coverage statistics.
        """
        # Linear growth of coverage
        n_extracted = np.arange(1, 21)
        coverage = 1 - np.exp(-0.1 * n_extracted)

        return pd.DataFrame({
            "num_extracted": n_extracted,
            "coverage_rate": coverage
        })
