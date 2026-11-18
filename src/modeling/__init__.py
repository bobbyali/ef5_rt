"""
Mathematical modelling package for EF5 PET/CT-driven tumour growth and hypoxia radiotherapy response.

Modules:
  - physics: Shared oxygen-dependent radiobiology (Alper-Howard-Flanders OER, LQ survival,
             pO2-dependent cellular kinetics, and bioreductive EF5 PET <-> pO2 mappings).
  - legacy_spheroid: Faithful & bug-audited Python ports of the 2013 MATLAB spheroid ODE
                     (my_spheroid_07/08) and Dr. Alex Fletcher's CodeForBobby.m (+ Byrne analytical).
  - spheroid_rt_model: Coupled 1D radially-symmetric quasi-steady oxygen PDE + moving-boundary
                       viable/damaged/necrotic spheroid RT model (Sub-goal 1).
  - voxel_initializer: 3D CT/EF5-PET voxel bundle loader, isotropic resampler, and vascular
                       supply anchoring (Sub-goal 2).
  - voxel_rt_model: 3D continuum oxygen PDE + Darcy pressure-velocity tumour growth/shrinkage
                    and virtual EF5 PET forward model (Sub-goal 2).
"""

from src.modeling.physics import RadiobiologyParams

__all__ = ["RadiobiologyParams"]
