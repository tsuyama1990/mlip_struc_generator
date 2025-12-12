import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list
from ase.data import atomic_numbers

class ZBLCalculator(Calculator):
    """
    Ziegler-Biersack-Littmark (ZBL) repulsive potential calculator.
    
    Potential V(r) = (1/(4*pi*epsilon_0)) * (Z1*Z2*e^2/r) * phi(r/a)
    
    where phi(x) is the screening function:
    phi(x) = 0.1818*exp(-3.2*x) + 0.5099*exp(-0.9423*x) + 0.2802*exp(-0.4029*x) + 0.02817*exp(-0.2016*x)
    
    and a is the screening length:
    a = 0.8854 * a0 / (Z1^0.23 + Z2^0.23)
    
    CONSTANTS (in eV and Angstrom):
    Coulomb constant k_e * e^2 = 14.3996 eV*A
    a0 (Bohr radius) = 0.529177 A
    
    """
    implemented_properties = ['energy', 'forces', 'stress']
    
    # Constants
    KE_E2 = 14.3996  # eV * A
    BOHR = 0.529177  # A

    def __init__(self, cutoff=2.0, **kwargs):
        """
        Args:
            cutoff (float): Cutoff distance in Angstrom. ZBL is short-ranged, 
                            so a small cutoff (e.g. 2.0 A) is usually sufficient.
        """
        super().__init__(**kwargs)
        self.cutoff = cutoff

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        
        positions = self.atoms.get_positions()
        numbers = self.atoms.numbers
        forces = np.zeros_like(positions)
        energy = 0.0
        
        # Use ASE neighbor list
        # i, j are unique pairs if self_interaction=False? No, it returns both i->j and j->i
        # But for pair potentials we usually iterate unique pairs or divide energy by 2.
        # i, j = neighbor indeces
        # d = distance
        # D = vector r_j - r_i
        i_list, j_list, d_list, D_list = neighbor_list('ijdD', self.atoms, self.cutoff, self_interaction=False)
        
        # Filter for non-zero distance (should be handled by cutoff, but safety for d=0)
        mask = d_list > 1e-4 
        
        if np.any(mask):
            i_idx = i_list[mask]
            j_idx = j_list[mask]
            r = d_list[mask]
            r_vec = D_list[mask] # r_j - r_i
            
            # DEBUG: Log if very close atoms found
            min_r = np.min(r)
            if min_r < 1.0:
                 import logging
                 logger = logging.getLogger(__name__)
                 logger.warning(f"ZBL: Found close atoms d={min_r:.4f} A. Pair count: {len(r)}")
            
            # Atomic numbers
            Z1 = numbers[i_idx]
            Z2 = numbers[j_idx]
            
            # Screening length a
            # a = 0.8854 * a0 / (Z1^0.23 + Z2^0.23)
            a = (0.8854 * self.BOHR) / (Z1**0.23 + Z2**0.23)
            
            x = r / a
            
            # Screening function phi(x) and its derivative phi'(x)
            # phi(x) = sum( c_i * exp(-d_i * x) )
            c = np.array([0.1818, 0.5099, 0.2802, 0.02817])
            d_exp = np.array([3.2, 0.9423, 0.4029, 0.2016])
            
            # Vectorized calculation for phi and dphi/dx
            # shape of x is (N_pairs,)
            # shape of c, d is (4,)
            # We want (N_pairs, 4)
            
            x_expanded = x[:, None]
            phi_terms = c * np.exp(-d_exp * x_expanded)
            phi = np.sum(phi_terms, axis=1)
            
            dphi_dx_terms = -c * d_exp * np.exp(-d_exp * x_expanded)
            dphi_dx = np.sum(dphi_dx_terms, axis=1)
            
            # Potential V(r) = (Ke / r) * phi(x)
            # Energy
            v_coulomb = (self.KE_E2 * Z1 * Z2) / r
            e_pair = v_coulomb * phi
            
            # We sum over all pairs (i,j) and (j,i). Total energy is 1/2 sum.
            energy = np.sum(e_pair) / 2.0
            
            # Force F(r) = -dV/dr
            # V(r) = C * (1/r) * phi(r/a) where C = Ke*Z1*Z2
            # dV/dr = C * [ (-1/r^2)*phi + (1/r)*phi' * (1/a) ]
            #       = (C/r) * [ -phi/r + phi'/a ]
            #       = (V / r) * [ -1 + (r/phi)*(phi'/a) ]  <-- careful with phi=0
            # Let's use direct:
            # dV/dr = (V_coulomb * phi)' = V_coulomb' * phi + V_coulomb * phi'
            #       = (-V_coulomb/r) * phi + V_coulomb * (dphi/dx * dx/dr)
            #       = (-e_pair / r) + (v_coulomb * dphi_dx * (1/a))
            
            dV_dr = (-e_pair / r) + (v_coulomb * dphi_dx / a)
            
            # Force vector on i due to j: F_ij = - (dV/dr) * (r_i - r_j) / r
            # Vector r_vec is (r_j - r_i). So (r_i - r_j) = -r_vec.
            # F_ij = - (dV/dr) * (-r_vec / r) = (dV/dr) * (r_vec / r)
            
            # Wait check sign:
            # Repulsive => Energy decreases as r increases => dV/dr < 0.
            # Force pushes i away from j. Direction is (r_i - r_j).
            # If dV/dr is negative, then -dV/dr is positive.
            # So Force = Positive * (r_i - r_j)/r.
            # r_vec = r_j - r_i.
            # So Force = Positive * (-r_vec)/r.
            
            # My dV/dr calculation:
            # e_pair > 0, r > 0 => -e_pair/r < 0.
            # dphi_dx < 0 (decaying). v_coulomb > 0, a > 0 => 2nd term < 0.
            # So dV/dr < 0. Correct (repulsive).
            
            # F_i_vec = - (dV/dr) * (r_i - r_j)/r
            #         = - (dV/dr) * (-r_vec)/r
            #         = (dV/dr/r) * r_vec
            
            # Note: dV/dr is negative. So (dV/dr/r) is negative.
            # r_vec is (r_j - r_i).
            # So force is along (r_j - r_i) but negative => along (r_i - r_j). Away from neighbor. Correct.
            
            f_factor = (dV_dr / r)[:, None]
            f_contribution = f_factor * r_vec
            
            # Add to forces
            for k, idx in enumerate(i_list[mask]):
                forces[idx] += f_contribution[k]
            
            # --- Stress Calculation ---
            # Virial Stress: S = -1/V * Sum( r_ij * F_ij )
            # ASE Convention: Positive stress = repulsive (wants to expand).
            # My f_factor is negative for repulsion.
            # My r_vec is (r_j - r_i).
            # Force on i is F_i = f_factor * r_vec.
            # Term r_ij * F_ij where r_ij = r_i - r_j = -r_vec.
            # Term = (-r_vec) * (f_factor * r_vec) = -f_factor * (r_vec^2)
            # Since f_factor < 0, Term > 0.
            # Formula S = -1/V * Term => S < 0 ?
            # Wait, ASE: Stress = 1/V * dE/dEps.
            # Repulsive => Expansion lowers energy => dE/dEps < 0?
            # NO. "Positive means system is under pressure".
            # If under pressure, it pushes out.
            # Let's trust the standard virial sign:
            # Stress = -1/V * Virial (where Virial < 0 for repulsion?)
            # Actually, let's look at a reference result.
            # Ideally S_xx should be positive for ZBL.
            
            # Let's try: Stress_contribution = -0.5 * (r_i - r_j) tensor F_ij
            # r_i - r_j = -r_vec. F_ij implies force on i = f_contribution.
            # S_ab = -0.5 * (-r_vec)_a * (f_contribution)_b
            #      = +0.5 * r_vec_a * f_contribution_b
            # f_contribution is parallel to r_vec (with neg scalar).
            # So S_ab has sign of f_factor. (Negative).
            
            # If I get negative stress, that implies tension (wants to contract).
            # But ZBL is repulsive! It should want to expand!
            # So I likely need a minus sign.
            # S_ab = -0.5 * r_vec_a * f_contribution_b
            
            # Calculation:
            # volume
            vol = self.atoms.get_volume()
            
            # s_k = -0.5 * r_vec * f_contribution / vol
            # We construct Voigt [xx, yy, zz, yz, xz, xy]
            
            # f_contribution = f_factor * r_vec
            # So r_vec * f_contribution = f_factor * r_vec * r_vec
            
            # Reshape for broadcasting
            rx = r_vec[:, 0]
            ry = r_vec[:, 1]
            rz = r_vec[:, 2]
            
            fx = f_contribution[:, 0]
            fy = f_contribution[:, 1]
            fz = f_contribution[:, 2]
            
            # Note: factor 0.5 because neighbor_list double counts pairs
            prefactor = -0.5 / vol
            
            s_xx = np.sum(prefactor * rx * fx)
            s_yy = np.sum(prefactor * ry * fy)
            s_zz = np.sum(prefactor * rz * fz)
            s_yz = np.sum(prefactor * ry * fz)
            s_xz = np.sum(prefactor * rx * fz)
            s_xy = np.sum(prefactor * rx * fy)
            
            stress = np.array([s_xx, s_yy, s_zz, s_yz, s_xz, s_xy])
        else:
            stress = np.zeros(6)

        self.results['energy'] = energy
        self.results['forces'] = forces
        self.results['stress'] = stress
