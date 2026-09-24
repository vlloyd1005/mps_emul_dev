"""
compute_delta_chi2.py

For a batch of cosmological parameter sets, run cobaya's `evaluate` sampler
(N=1) twice per row -- once with the CAMB theory pipeline and once with the
emulator (emulrdrag/emulbaosn/emulmps) pipeline -- writing real cobaya
chain output for each. This does NOT compute chi2 in Python; each row is
formatted into a `sampler: evaluate: N: 1, override: {...}` block (the
same mechanism your two original example yamls used) and passed to
`cobaya.run()`, so the resulting output files are exactly what cobaya
itself reports (including the `chi2_roman_real.cosmic_shear = ...` line),
for you to read/parse yourself.

Follows the same MPI batching pattern as datageneratormps.py (rank 0 loads
and scatters the parameter array, each rank runs its slice).

USAGE (mirrors datageneratormps.py):
    mpirun -n <N> python compute_delta_chi2.py -f <n>

OUTPUT:
    For row i (global index across the full parameter file), two output
    prefixes are written under projects/roman_real/chains/:
        DELTA_CHI2_CAMB_<i>.1.txt   (+ .updated.yaml, etc.)
        DELTA_CHI2_EMUL_<i>.1.txt   (+ .updated.yaml, etc.)
    Each .1.txt has a header row and a data row including a
    chi2_roman_real.cosmic_shear column -- that's the number to read.
"""

import numpy as np
import sys
import os
import copy
import time
from mpi4py import MPI
from cobaya.yaml import yaml_load
from cobaya.run import run

# ---------------------------------------------------------------------------
# CLI args (mirrors the "-f <n>" convention from datageneratormps.py)
# ---------------------------------------------------------------------------
if "-f" not in sys.argv:
    sys.exit(
        "compute_delta_chi2.py: missing required '-f <n>' argument.\n"
        "Usage: python compute_delta_chi2.py -f <n>\n"
        "Check the launch command in your job script -- it needs to pass -f."
    )
idx = sys.argv.index("-f")
try:
    n = int(sys.argv[idx + 1])
except (IndexError, ValueError):
    sys.exit("compute_delta_chi2.py: '-f' must be followed by an integer, e.g. '-f 0'.")

# ---------------------------------------------------------------------------
# YAML configs
#
# Same roman_real params/theory blocks as before (mnu and HMCode_logT_AGN
# sampled -- under params: -- in BOTH configs, so both are valid override
# targets for the evaluate sampler; roman_A2_2 aligned to 0.0 in both).
# No `sampler:` or `output:` block baked in here -- both get set
# programmatically per row below, exactly the way your two original
# example yamls had a `sampler.evaluate.override` block, just filled in
# from each row instead of hardcoded fiducial values.
# ---------------------------------------------------------------------------

CAMB_YAML = r"""
timing: True
debug: False
stop_at_error: False

likelihood:
  roman_real.cosmic_shear:
    use_emulator: 0
    path: ./external_modules/data/roman_real
    data_file: example1.dataset # that assumes lens = source
    print_datavector: False
    print_datavector_file: "./projects/roman_real/chains/theory.modelvector"
    accuracyboost: 1.0
    integration_accuracy: 0
    lmax: 75000
    kmax_boltzmann: 7.5
    non_linear_emul: 2
    IA_model: 0
    IA_code: 0
    IA_redshift_evolution: 3
    ggl_exclude: [[6,0],[7,0],[7,1]]
    debug: false
    use_baryon_pca: false
    create_baryon_pca: false
    baryon_pca_select_sims: "antilles-2-99/antilles-101-378/antilles-380-400"
    filename_baryon_pca: "./projects/lsst_y1/chains/pca.txt"
params:
  As_1e9:
    prior:
      min: 0.5
      max: 5
    ref:
      dist: norm
      loc: 2.1
      scale: 0.2
    proposal: 0.2
    latex: 10^9 A_\mathrm{s}
    drop: true
    renames: A
  ns:
    prior:
      min: 0.87
      max: 1.07
    ref:
      dist: norm
      loc: 0.96605
      scale: 0.01
    proposal: 0.01
    latex: n_\mathrm{s}
  H0:
    prior:
      min: 55
      max: 91
    ref:
      dist: norm
      loc: 67.32
      scale: 3
    proposal: 3
    latex: H_0
  omegab:
    prior:
      min: 0.03
      max: 0.07
    ref:
      dist: norm
      loc: 0.0495
      scale: 0.004
    proposal: 0.004
    latex: \Omega_\mathrm{b}
    drop: true
  omegam:
    prior:
      min: 0.1
      max: 0.9
    ref:
      dist: norm
      loc: 0.316
      scale: 0.015
    proposal: 0.015
    latex: \Omega_\mathrm{m}
    drop: true
  w:
    prior:
      min: -3
      max: -0.01
    ref:
      dist: norm
      loc: -0.99
      scale: 0.05
    proposal: 0.05
    latex: w_{0,\mathrm{DE}}
  w0pwa:
    prior:
      min: -5
      max: -0.01
    ref:
      dist: norm
      loc: -0.99
      scale: 0.05
    proposal: 0.05
    latex: w_{0,\mathrm{DE}}+w_{a,\mathrm{DE}}
    drop: true
  wa:
    value: 'lambda w0pwa, w: w0pwa - w'
    latex: w_{a,\mathrm{DE}}
  mnu:
    prior:
      min: 0.06
      max: 0.6
    ref:
      dist: norm
      loc: 0.25
      scale: 0.1
    proposal: 0.05
  HMCode_logT_AGN:
    prior:
      min: 6.0
      max: 9.0
    ref:
      dist: norm
      loc: 7.8
      scale: 0.3
    proposal: 0.3
    latex: \log_{10}(T_\mathrm{AGN}/\mathrm{K})
  tau:
    value: 0.0697186
    latex: \tau_\mathrm{reio}
  As:
    value: 'lambda As_1e9: 1e-9 * As_1e9'
    latex: A_\mathrm{s}
  omegabh2:
    value: 'lambda omegab, H0: omegab*(H0/100)**2'
    latex: \Omega_\mathrm{b} h^2
  omegach2:
    value: 'lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2-(mnu*(3.046/3)**0.75)/94.0708'
    latex: \Omega_\mathrm{c} h^2
  omegal:
    latex: \Omega_\Lambda
  omegamh2:
    derived: 'lambda omegam, H0: omegam*(H0/100)**2'
    latex: \Omega_\mathrm{m} h^2
  sigma8:
    latex: \sigma_8
  s8h5:
    derived: 'lambda sigma8, H0: sigma8*(H0*1e-2)**(-0.5)'
    latex: \sigma_8/h^{0.5}
  s8omegamp5:
    derived: 'lambda sigma8, omegam: sigma8*omegam**0.5'
    latex: \sigma_8 \Omega_\mathrm{m}^{0.5}
  s8omegamp25:
    derived: 'lambda sigma8, omegam: sigma8*omegam**0.25'
    latex: \sigma_8 \Omega_\mathrm{m}^{0.25}
  age:
    latex: '{\rm{Age}}/\mathrm{Gyr}'
  rdrag:
    latex: r_\mathrm{drag}
  yheused:
    latex: Y_P^\mathrm{BBN}
  omegan2:
    latex: \Omega_\mathrm{\\nu} h^2
  omegan:
    derived: 'lambda omegan2, H0: omegan2/((H0/100)**2)'
    latex: \Omega_\mathrm{\\nu}
  roman_DZ_S1:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^1
  roman_DZ_S2:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^2
  roman_DZ_S3:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^3
  roman_DZ_S4:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^4
  roman_DZ_S5:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^5
  roman_DZ_S6:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^6
  roman_DZ_S7:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^7
  roman_DZ_S8:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^8
  roman_M1:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^1
  roman_M2:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^2
  roman_M3:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^3
  roman_M4:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^4
  roman_M5:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^5
  roman_M6:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^6
  roman_M7:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^7
  roman_M8:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^8
  roman_A1_1:
    prior:
      min: -5
      max: 5
    ref:
      dist: norm
      loc: 0.7
      scale: 0.5
    proposal: 0.5
    latex: A_\mathrm{1-IA,roman}^1
  roman_A1_2:
    prior:
      min: -5
      max: 5
    ref:
      dist: norm
      loc: -1.7
      scale: 0.5
    proposal: 0.5
  roman_A2_1:
    prior:
      min: -5
      max: 5
    ref:
      dist: norm
      loc: -1.36
      scale: 0.5
    proposal: 0.5
    latex: A_\mathrm{2-IA,Roman}^1
  roman_A2_2:
    prior:
      min: -5
      max: 5
    ref:
      dist: norm
      loc: -2.5
      scale: 0.5
    proposal: 0.5
    latex: A_\mathrm{2-IA,Roman}^2
  roman_BTA_1:
    prior:
      min: 0
      max: 2
    ref:
      dist: norm
      loc: 1
      scale: 0.15
    proposal: 0.1
    latex: A_\mathrm{BTA-IA,Roman}^1

theory:
  camb:
    path: ./external_modules/code/CAMB
    stop_at_error: False
    use_renames: True
    extra_args:
      halofit_version: mead2020_feedback
      AccuracyBoost: 1.6
      lens_potential_accuracy: 1.0
      dark_energy_model: ppf
      accurate_massive_neutrino_transfers: false
      k_per_logint: 30
      kmax: 100
"""

EMUL_YAML = r"""
timing: True
debug: False
stop_at_error: False

likelihood:
  roman_real.cosmic_shear:
    use_emulator: 2
    path: ./external_modules/data/roman_real
    data_file: example1.dataset # that assumes lens = source
    print_datavector: False
    print_datavector_file: "./projects/roman_real/chains/theory.modelvector"
    accuracyboost: 1.0
    integration_accuracy: 0
    lmax: 75000
    kmax_boltzmann: 7.5
    non_linear_emul: 2
    IA_model: 0
    IA_code: 0
    IA_redshift_evolution: 3
    ggl_exclude: [[6,0],[7,0],[7,1]]
    debug: false
    use_baryon_pca: false
    create_baryon_pca: false
    baryon_pca_select_sims: "antilles-2-99/antilles-101-378/antilles-380-400"
    filename_baryon_pca: "./projects/lsst_y1/chains/pca.txt"
params:
  As_1e9:
    prior:
      min: 0.5
      max: 5
    ref:
      dist: norm
      loc: 2.1
      scale: 0.25
    proposal: 0.2
    latex: 10^9 A_\mathrm{s}
    renames: A
  ns:
    prior:
      min: 0.87
      max: 1.07
    ref:
      dist: norm
      loc: 0.96605
      scale: 0.01
    proposal: 0.01
    latex: n_\mathrm{s}
  H0:
    prior:
      min: 55
      max: 91
    ref:
      dist: norm
      loc: 67.32
      scale: 5
    proposal: 3
    latex: H_0
  omegab:
    prior:
      min: 0.03
      max: 0.07
    ref:
      dist: norm
      loc: 0.0495
      scale: 0.004
    proposal: 0.004
    latex: \Omega_\mathrm{b}
  omegam:
    prior:
      min: 0.1
      max: 0.9
    ref:
      dist: norm
      loc: 0.316
      scale: 0.01
    proposal: 0.01
    latex: \Omega_\mathrm{m}
  w:
    prior:
      min: -3
      max: -0.01
    ref:
      dist: norm
      loc: -0.99
      scale: 0.05
    proposal: 0.05
    latex: w_{0,\mathrm{DE}}
  w0pwa:
    prior:
      min: -5
      max: -0.01
    ref:
      dist: norm
      loc: -0.99
      scale: 0.05
    proposal: 0.05
    latex: w_{0,\mathrm{DE}}+w_{a,\mathrm{DE}}
  wa:
    value: 'lambda w0pwa, w: w0pwa - w'
    derived: false
    latex: 'w_{a,\mathrm{DE}}'
  mnu:
    prior:
      min: 0.06
      max: 0.6
    ref:
      dist: norm
      loc: 0.25
      scale: 0.1
    proposal: 0.05
    latex: \Sigma m_\nu
  HMCode_logT_AGN:
    prior:
      min: 6.0
      max: 9.0
    ref:
      dist: norm
      loc: 7.8
      scale: 0.3
    proposal: 0.3
    latex: \log_{10}(T_\mathrm{AGN}/\mathrm{K})
  omegabh2:
    value: 'lambda omegab, H0: omegab*(H0/100)**2'
    latex: \Omega_\mathrm{b} h^2
  omegach2:
    value: 'lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2-(mnu*(3.046/3)**0.75)/94.0708'
    latex: \Omega_\mathrm{c} h^2
  As:
    value: 'lambda As_1e9: 1e-9 * As_1e9'
    latex: A_\mathrm{s}
  roman_DZ_S1:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^1
  roman_DZ_S2:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^2
  roman_DZ_S3:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^3
  roman_DZ_S4:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^4
  roman_DZ_S5:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^5
  roman_DZ_S6:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^6
  roman_DZ_S7:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^7
  roman_DZ_S8:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^8
  roman_M1:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^1
  roman_M2:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^2
  roman_M3:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^3
  roman_M4:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^4
  roman_M5:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^5
  roman_M6:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^6
  roman_M7:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^7
  roman_M8:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^8
  roman_A1_1:
    prior:
      min: -5
      max: 5
    ref:
      dist: norm
      loc: 0.7
      scale: 0.5
    proposal: 0.5
    latex: A_\mathrm{1-IA,roman}^1
  roman_A1_2:
    prior:
      min: -5
      max: 5
    ref:
      dist: norm
      loc: -1.7
      scale: 0.5
    proposal: 0.5
  roman_A2_1:
    value: 0.0
    latex: A_\mathrm{2-IA,Roman}^1
  roman_A2_2:
    value: 0.0
    latex: A_\mathrm{2-IA,Roman}^2
  roman_BTA_1:
    value: 0.0
    latex: A_\mathrm{BTA-IA,Roman}^1

theory:
  emulrdrag:
    path: ./cobaya/cobaya/theories/
    provides: ['rdrag']
    extra_args:
      file: ['external_modules/data/emultrf/BAO_SN_RES/emul_lcdm_rdrag_GP.joblib']
      extra: ['external_modules/data/emultrf/BAO_SN_RES/extra_lcdm_rdrag.npy']
      ord: [['omegabh2','omegach2']]
  emulbaosn:
    path: ./cobaya/cobaya/theories/
    stop_at_error: True
    provides: ['comoving_radial_distance', 'angular_diameter_distance', 'Hubble']
    extra_args:
      device: "cuda"
      file:  [None, 'external_modules/data/emultrf/BAO_SN_RES/w0wa/emul_w0wa_H.pt']
      extra: [None, 'external_modules/data/emultrf/BAO_SN_RES/w0wa/extra_w0wa_H.npy']
      ord: [None, ['omegam','H0','w','wa']]
      extrapar: [{'MLA': 'INT', 'ZMIN' : 0.0001, 'ZMAX' : 3, 'NZ' : 600},
                 {'MLA': 'ResMLP', 'offset' : 0.0, 'INTDIM' : 4, 'NLAYER' : 6,
                  'TMAT': 'external_modules/data/emultrf/BAO_SN_RES/w0wa/PCA_w0wa_H.npy',
                  'ZLIN': 'external_modules/data/emultrf/BAO_SN_RES/w0wa/z_lin_w0wa.npy'}]
  emulmps:
    path: ./cobaya/cobaya/theories/
    stop_at_error: True
    extra_args:
      # model_file: "/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/mps_emu/models/emulator_npce_w0wacdm_expanded_mead2020_Tfree_mnufree_lin_tfreemnufree_nTrain500_v11.keras"
      # metadata_file: "/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/mps_emu/metadata/metadata_w0wacdm_expanded_mead2020_Tfree_mnufree_lin_tfreemnufree_nTrain500_v11/metadata.joblib"
      # nl_model_file: "/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/mps_emu/models/emulator_npce_w0wacdm_expanded_mead2020_Tfree_mnufree_boost_tfreemnufree_nTrain500_v11.keras"
      # nl_metadata_file: "/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/mps_emu/metadata/metadata_w0wacdm_expanded_mead2020_Tfree_mnufree_boost_tfreemnufree_nTrain500_v11/metadata.joblib"
      model_file: "/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/mps_emu/models/emulator_npce_w0wacdm_expanded_mead2020_Tfree_mnufree_lin_tfreemnufree_nTrain750_v13.keras"
      metadata_file: "/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/mps_emu/metadata/metadata_w0wacdm_expanded_mead2020_Tfree_mnufree_lin_tfreemnufree_nTrain750_v13/metadata.joblib"
      nl_model_file: "/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/mps_emu/models/emulator_npce_w0wacdm_expanded_mead2020_Tfree_mnufree_boost_tfreemnufree_nTrain750_v13.keras"
      nl_metadata_file: "/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/mps_emu/metadata/metadata_w0wacdm_expanded_mead2020_Tfree_mnufree_boost_tfreemnufree_nTrain750_v13/metadata.joblib"
      emul_folder: "/lustre/nvwulf/home/vlloyd/emulator_out_nTrain50_cut2_LNL"
      use_syren: False
      param_order: ["As_1e9", "ns", "H0", "omegab", "omegam", 'w', 'wa', 'HMCode_logT_AGN', 'mnu']
"""


# ---------------------------------------------------------------------------
# Parameter file format
#
# cosmos_med_hypersphere.npy is a plain numpy array of shape (N, 9) -- NOT
# a dict/structured array. Each row is a cosmology, with columns in this
# fixed order:
PARAM_NAMES = ["As", "ns", "H0", "Omega_b", "Omega_m", "w", "wa", "Tagn", "mnu"]

# Fixed nuisance parameter values used in BOTH override blocks.
# roman_A2_1/roman_A2_2/roman_BTA_1 are only included for CAMB, since the
# emulator yaml fixes them via `value: 0.0` (not sampled) -- overriding a
# non-sampled parameter isn't valid for the evaluate sampler.
COMMON_NUISANCE = {
    "roman_DZ_S1": 0.0, "roman_DZ_S2": 0.0, "roman_DZ_S3": 0.0, "roman_DZ_S4": 0.0,
    "roman_DZ_S5": 0.0, "roman_DZ_S6": 0.0, "roman_DZ_S7": 0.0, "roman_DZ_S8": 0.0,
    "roman_M1": 0.0, "roman_M2": 0.0, "roman_M3": 0.0, "roman_M4": 0.0,
    "roman_M5": 0.0, "roman_M6": 0.0, "roman_M7": 0.0, "roman_M8": 0.0,
    "roman_A1_1": 0.6, "roman_A1_2": -1.5,
}
CAMB_ONLY_NUISANCE = {"roman_A2_1": 0.0, "roman_A2_2": 0.0, "roman_BTA_1": 0.0}


def row_to_cosmo_dict(row):
    """row: 1D array-like of length len(PARAM_NAMES), in PARAM_NAMES order."""
    if len(row) != len(PARAM_NAMES):
        raise ValueError(
            f"row has {len(row)} entries, expected {len(PARAM_NAMES)} "
            f"matching PARAM_NAMES={PARAM_NAMES}"
        )
    return dict(zip(PARAM_NAMES, row))


def cosmo_override(row):
    """
    Translate one row into the override dict entries shared by both
    CAMB and emulator sampler.evaluate blocks. "As" passes straight
    through as As_1e9 (already pre-scaled). w0pwa = w + wa, since
    w0pwa (not wa) is the model's actual sampled dimension.
    """
    cosmo = row_to_cosmo_dict(row)
    return {
        "As_1e9": float(cosmo["As"]),
        "ns": float(cosmo["ns"]),
        "H0": float(cosmo["H0"]),
        "omegab": float(cosmo["Omega_b"]),
        "omegam": float(cosmo["Omega_m"]),
        "w": float(cosmo["w"]),
        "w0pwa": float(cosmo["w"]) + float(cosmo["wa"]),
        "mnu": float(cosmo["mnu"]),
        "HMCode_logT_AGN": float(cosmo["Tagn"]),
    }


def build_camb_override(row):
    o = cosmo_override(row)
    o.update(COMMON_NUISANCE)
    o.update(CAMB_ONLY_NUISANCE)
    return o


def build_emul_override(row):
    o = cosmo_override(row)
    o.update(COMMON_NUISANCE)
    return o


if __name__ == "__main__":

    BASE = "/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/"

    parameters_file = "/lustre/nvwulf/home/vlloyd/near_fiducial_samples.npy"

    # camb_output_prefix = BASE + "projects/roman_real/chains/emul_test/DELTA_CHI2_CAMB"
    emul_output_prefix = BASE + "projects/roman_real/chains/emul_test_new/DELTA_CHI2_EMUL5"
    os.makedirs(os.path.dirname(emul_output_prefix), exist_ok=True)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    # Base yaml dicts, parsed once per rank; deep-copied per row before
    # adding that row's sampler.evaluate.override block and output path.
    # camb_base_info = yaml_load(CAMB_YAML)
    emul_base_info = yaml_load(EMUL_YAML)

    print("rank", rank, "yamls loaded")
    start = time.time()

    if rank == 0:
        samples = np.load(parameters_file, allow_pickle=True)
        total_num_dvs = len(samples)

        param_info = samples[0:total_num_dvs:num_ranks]
        for i in range(1, num_ranks):
            comm.send(samples[i:total_num_dvs:num_ranks], dest=i, tag=1)
    else:
        param_info = comm.recv(source=0, tag=1)

    num_local = len(param_info)
    n_camb_ok = n_emul_ok = 0

    for i in range(num_local):
        row = param_info[i]
        # matches the slicing pattern samples[rank:total:num_ranks] used
        # above, so this is the row's index in the ORIGINAL parameter file.
        global_idx = rank + i * num_ranks

        # camb_info_i = copy.deepcopy(camb_base_info)
        # camb_info_i["sampler"] = {"evaluate": {"N": 1, "override": build_camb_override(row)}}
        # camb_info_i["output"] = f"{camb_output_prefix}_{global_idx}"
        # try:
        #     run(camb_info_i)
        #     n_camb_ok += 1
        # except Exception as e:
        #     print(f"[rank {rank}] CAMB run failed at global idx {global_idx}: {e}")

        emul_info_i = copy.deepcopy(emul_base_info)
        emul_info_i["sampler"] = {"evaluate": {"N": 1, "override": build_emul_override(row)}}
        emul_info_i["output"] = f"{emul_output_prefix}_{global_idx}"
        try:
            run(emul_info_i)
            n_emul_ok += 1
        except Exception as e:
            print(f"[rank {rank}] Emulator run failed at global idx {global_idx}: {e}")

    print(f"[rank {rank}] done in {time.time() - start:.1f}s: "
          # f"{n_camb_ok}/{num_local} CAMB runs, "
          f" {n_emul_ok}/{num_local} emulator runs written.")


# mpirun -n 5 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
#     --bind-to core --map-by core --report-bindings --mca mpi_yield_when_idle 1 \
#     python compute_delta_chi2.py -f 0