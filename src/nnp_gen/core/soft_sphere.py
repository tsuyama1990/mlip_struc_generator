import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list

class SoftSphereCalculator(Calculator):
    """
    A simple pair-potential calculator that only includes a soft repulsive term.
    
    Potential: V(r) = (k/2) * (rc - r)^2  for r < rc
    Force:     F(r) = k * (rc - r) * (r_vec / r)
    
    Used for resolving atomic overlaps.
    """
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, rc=2.0, k=50.0, **kwargs):
        super().__init__(**kwargs)
        self.rc = rc
        self.k = k

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        
        # Use get_all_distances(mic=True) for robustness against PBC edge cases matching MD engine
        dists = self.atoms.get_all_distances(mic=True)
        # Efficient approach:
        np.fill_diagonal(dists, np.inf)
        
        mask = dists < self.rc
        
        positions = self.atoms.get_positions()
        forces = np.zeros_like(positions)
        energy = 0.0
        
        # DEBUG: Direct output to confirm running and detecting
        # import sys
        # min_found = np.min(dists)
        # sys.stderr.write(f"SoftSphere: N={len(self.atoms)} RC={self.rc} MinDist={min_found:.4f}\n")

        if np.any(mask):
            # EMERGENCY LOG: If we see dangerous overlap, shout about it.
            min_d_found = np.min(dists[mask])
            if min_d_found < 1.0:
                 # import sys
                 # sys.stderr.write(f"SoftSphere WARNING: Critical Overlap d={min_d_found:.4f} A. Applying massive repulsion.\n")
                 # sys.stderr.flush()
                 pass

            # We iterate over pairs
            # Note: dists is symmetric. We want to avoid double counting for energy,
            # but for forces we need both F_ij and F_ji.
            
            # Get indices
            rows, cols = np.where(mask)
            
            for r, c in zip(rows, cols):
                if r == c: continue
                d_val = dists[r, c]
                
                # Energy: sum 0.5 * k * (rc - r)^2
                # We calculate full sum and divide by 2 later (since we visit r,c and c,r)
                e_pair = 0.5 * self.k * (self.rc - d_val)**2
                energy += e_pair

                # Force on r due to c
                vec_cr = self.atoms.get_distance(r, c, vector=True, mic=True) # Vector from r to c
                
                # Force on r is away from c. Direction is -(r_c - r_r) = r_r - r_c.
                # F_vector = (Magnitude) * (-vec_cr) / dist
                
                f_mag = self.k * (self.rc - d_val)
                f_vec = -vec_cr / d_val * f_mag
                
                forces[r] += f_vec
                
                # Stress Calculation (Virial)
                r_rel = -vec_cr
                vol = self.atoms.get_volume()
                pref = -0.5 / vol 
                
                s_xx = pref * r_rel[0] * f_vec[0]
                s_yy = pref * r_rel[1] * f_vec[1]
                s_zz = pref * r_rel[2] * f_vec[2]
                s_yz = pref * r_rel[1] * f_vec[2]
                s_xz = pref * r_rel[0] * f_vec[2]
                s_xy = pref * r_rel[0] * f_vec[1]
                
                if 'stress' not in locals(): # Initialize stress if not exists in loop scope (safe)
                     # Actually stress variable needs to be outside loop.
                     pass 

            # Initialize stress if we found overlaps (re-loop or just do it inside)
            # Optimization: Move stress init outside
            pass
            
            # Re-calculating stress cleanly
        
        # --- Clean Re-implementation of Loop ---
        stress = np.zeros(6)
        
        if np.any(mask):
             rows, cols = np.where(mask)
             for r, c in zip(rows, cols):
                if r == c: continue
                d_val = dists[r, c]
                
                e_pair = 0.5 * self.k * (self.rc - d_val)**2
                energy += e_pair
                
                vec_cr = self.atoms.get_distance(r, c, vector=True, mic=True)
                f_mag = self.k * (self.rc - d_val)
                f_vec = -vec_cr / d_val * f_mag
                
                forces[r] += f_vec
                
                r_rel = -vec_cr
                vol = self.atoms.get_volume()
                pref = -0.5 / vol 
                
                stress[0] += pref * r_rel[0] * f_vec[0]
                stress[1] += pref * r_rel[1] * f_vec[1]
                stress[2] += pref * r_rel[2] * f_vec[2]
                stress[3] += pref * r_rel[1] * f_vec[2]
                stress[4] += pref * r_rel[0] * f_vec[2]
                stress[5] += pref * r_rel[0] * f_vec[1]

             energy *= 0.5
        
        self.results['energy'] = energy
        self.results['forces'] = forces
        self.results['stress'] = stress
