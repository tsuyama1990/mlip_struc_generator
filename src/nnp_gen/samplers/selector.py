import numpy as np
from typing import List, Optional
from ase import Atoms
from scipy.spatial.distance import cdist
import logging
from nnp_gen.core.interfaces import ISampler

logger = logging.getLogger(__name__)

class DescriptorManager:
    def __init__(self, rcut: float = 5.0, nmax: int = 4, lmax: int = 3):
        self.rcut = rcut
        self.nmax = nmax
        self.lmax = lmax
        self._check_dscribe()

    def _check_dscribe(self):
        try:
            import dscribe
            self.has_dscribe = True
        except ImportError:
            self.has_dscribe = False
            logger.warning("dscribe not found. Using RDF fallback.")

    def calculate(self, structures: List[Atoms]) -> np.ndarray:
        if not structures:
            return np.array([])

        if self.has_dscribe:
            try:
                return self._calculate_soap(structures)
            except Exception as e:
                logger.error(f"SOAP calculation failed: {e}. Falling back to RDF.")
                return self._calculate_rdf(structures)
        else:
            return self._calculate_rdf(structures)

    def _calculate_soap(self, structures: List[Atoms]) -> np.ndarray:
        # Determine species from all structures
        species = set()
        for atoms in structures:
            species.update(atoms.get_chemical_symbols())
        species = sorted(list(species))

        from dscribe.descriptors import SOAP
        # Ensure correct periodic setting.
        # If structures mixed, might be issue. Assuming consistent.
        periodic = any(structures[0].pbc)

        soap = SOAP(
            species=species,
            periodic=periodic,
            rcut=self.rcut,
            nmax=self.nmax,
            lmax=self.lmax,
            average="inner"
        )

        return soap.create(structures, n_jobs=1)

    def _calculate_rdf(self, structures: List[Atoms]) -> np.ndarray:
        """
        Calculate simple RDF histogram as descriptor.
        """
        bins = np.linspace(0, self.rcut, 20) # 20 bins for simple descriptor
        features = []
        for atoms in structures:
            if len(atoms) > 1:
                dists = atoms.get_all_distances(mic=any(atoms.pbc))
                # Ignore zero distances (diagonal)
                # Using mask
                mask = dists > 1e-6
                valid_dists = dists[mask]

                if len(valid_dists) > 0:
                    hist, _ = np.histogram(valid_dists, bins=bins, density=True)
                    features.append(hist)
                else:
                    features.append(np.zeros(len(bins)-1))
            else:
                features.append(np.zeros(len(bins)-1))

        return np.array(features)

class FPSSampler(ISampler):
    def __init__(self, descriptor_manager: DescriptorManager):
        self.desc_mgr = descriptor_manager

    def sample(self, structures: List[Atoms], n_samples: int) -> List[Atoms]:
        n_total = len(structures)
        if n_samples >= n_total:
            return structures

        features = self.desc_mgr.calculate(structures)
        if len(features) == 0:
            return []

        selected_indices = []

        # 1. Random first choice
        first_idx = np.random.randint(0, n_total)
        selected_indices.append(first_idx)

        # Initialize min_distances
        current_features = features[first_idx].reshape(1, -1)
        min_dists = cdist(current_features, features, metric='euclidean').flatten()

        for _ in range(n_samples - 1):
            # 3. Find point with max min_dist
            next_idx = np.argmax(min_dists)
            selected_indices.append(next_idx)

            # Update min_dists
            new_feat = features[next_idx].reshape(1, -1)
            dists_new = cdist(new_feat, features, metric='euclidean').flatten()
            min_dists = np.minimum(min_dists, dists_new)

        return [structures[i] for i in selected_indices]

class RandomSampler(ISampler):
    def sample(self, structures: List[Atoms], n_samples: int) -> List[Atoms]:
        n_total = len(structures)
        if n_samples >= n_total:
            return structures
        indices = np.random.choice(n_total, n_samples, replace=False)
        return [structures[i] for i in indices]
