import numpy as np
import cobaya
from cobaya.yaml import yaml_load
from cobaya.model import get_model
import sys
import os
import platform
import yaml
from mpi4py import MPI
from scipy.stats import qmc
import copy
import functools, iminuit, copy, argparse, random, time 
import emcee, itertools
from schwimmbad import MPIPool
from cobaya.likelihood import Likelihood

if "-f" in sys.argv:
    idx = sys.argv.index('-f')
n= int(sys.argv[idx+1])


class MyPkLikelihood(Likelihood):

    def initialize(self):
        self.kmax = 100.0
        self.linear_pk = None
        self.nonlinear_pk = None
        z1_mps = np.linspace(0,3,33,endpoint=False)
        z2_mps = np.linspace(3,10,7,endpoint=False)
        z3_mps = np.linspace(10,50,12)
        self.z_eval = np.concatenate((z1_mps,z2_mps,z3_mps),axis=0)
    
    def get_requirements(self):
        return {"omegabh2": None, "Pk_interpolator": {
        "z":self.z_eval,
        "k_max": 100,
        "nonlinear": (True,False),
        "vars_pairs": ([("delta_tot", "delta_tot")])
        }, "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
        'tt': 0
        },"omegach2":None, "H0": None, "ns": None, "As": None, "tau": None}

    def logp(self, **params):
        # Get linear P(k, z)

        self.linear_pk = self.provider.get_Pk_interpolator(
                ("delta_tot", "delta_tot"), nonlinear=False, extrap_kmin = 5e-6, extrap_kmax = self.kmax)

        # Get non-linear P(k, z)
        self.nonlinear_pk = self.provider.get_Pk_interpolator(
                ("delta_tot", "delta_tot"), nonlinear=True, extrap_kmin = 5e-6, extrap_kmax = self.kmax
            )

        # Choose k range and evaluate both
        k = np.logspace(-5.1, 2, 500)
        pk_lin = self.linear_pk.P(self.z_eval, k)
        pk_nonlin = self.nonlinear_pk.P(self.z_eval, k)



        # Return dummy log-likelihood
        return -0.5 


####add yaml here, and make all input paratemers passing in
yaml_string=r"""

likelihood:
  dummy:
    class: MyPkLikelihood

params:
  logA:
    prior:
      min: 1.61
      max: 3.91
    ref:
      dist: norm
      loc: 3.0448
      scale: 0.05
    proposal: 3
    latex: \log(10^{10} A_\mathrm{s})
    drop: true
  ns:
    prior:
      min: 0.6
      max: 1.3
    ref:
      dist: norm
      loc: 0.96605
      scale: 0.005
    proposal: 0.96
    latex: n_\mathrm{s}

  H0:
    prior:
      min: 20
      max: 120
    ref:
      dist: norm
      loc: 67
      scale: 2
    proposal: 67
    latex: H_0


  omegabh2:
    prior:
      min: 0.0
      max: 0.04
    ref:
      dist: norm
      loc: 0.022383
      scale: 0.005
    proposal: 0.005
    latex: \Omega_\mathrm{b} h^2
  omegach2:
    prior:
      min: 0.0
      max: 0.5
    ref:
      dist: norm
      loc: 0.12011
      scale: 0.03
    proposal: 0.03
    latex: \Omega_\mathrm{c} h^2
  
  tau:
    value: 0.0697186
    latex: \tau_\mathrm{reio}

  
  As:
    value: 'lambda logA: 1e-10*np.exp(logA)'
    latex: A_\mathrm{s}
  mnu:
    value: 0.06


  thetastar:
    derived: true
    latex: \Theta_\star
  rdrag:
    derived: True
    latex: r_\mathrm{drag}

  w:
    prior:
      min: -2.2
      max: 05
    ref:
      dist: norm
      loc: -1.0
      scale: 0.1
    proposal: 0.1
    latex: w_0

  wa:
    prior:
      min: -4
      max: 1
    ref:
      dist: norm
      loc: 0.0
      scale: 0.3
    proposal: 0.3
    latex: w_a

theory:
  # camb:
  #   path: ./external_modules/code/CAMB
  #   extra_args:
  #     halofit_version: mead2020
  #     dark_energy_model: ppf
  #     # w: 'lambda w0: w0'
  #     # wa: 'lambda wa: wa'
  #     lmax: 1000
  #     kmax: 100
  #     k_per_logint: 30
  #     AccuracyBoost: 1.2
  #     lAccuracyBoost: 1.0
  #     lens_margin: 2050
  #     lens_k_eta_reference: 18000.0
  #     nonlinear: NonLinear_both
  #     #recombination_model: CosmoRec
  #     Accuracy.AccurateBB: True
  #     DoLateRadTruncation: False
  camb:
    path: ./external_modules/code/CAMB
    stop_at_error: False
    use_renames: True
    extra_args:
      halofit_version: takahashi
      AccuracyBoost: 1.6
      dark_energy_model: ppf
      accurate_massive_neutrino_transfers: false
      k_per_logint: 30
      kmax: 50.0




output: ./projects/axions/chains/EXAMPLE_EVALUATE0

"""


####add main function.


#===================================================================================================
# datavectors


if __name__ == '__main__':

    f = yaml_load(yaml_string)
    sys.modules["MyPkLikelihood"] = sys.modules[__name__]

    model = get_model(f)
    
    prior_params = list(model.parameterization.sampled_params())
    sampling_dim = len(prior_params)

    BASE = "/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/"

    datavectors_file_path = BASE+'mps/output/train_w0wacdm_'+str(n)+'_expanded_uniform'
    parameters_file  = BASE+'mps/input/train_w0wacdm_mps_datagen_'+str(n)+'_expanded_uniform.npy'

    # datavectors_file_path = BASE+'mps/output/one_eval_wcdm_500ks'
    # parameters_file  = BASE+'mps/input/one_eval_datagen_params.npy'

    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    print('rank',rank,'is at barrier')

        
    start = time.time()
        
    z1_mps = np.linspace(0,3,33,endpoint=False)
    z2_mps = np.linspace(3,10,7,endpoint=False)
    z3_mps = np.linspace(10,50,12)
    z_mps = np.concatenate((z1_mps,z2_mps,z3_mps),axis=0)
    len_z = len(z_mps)
    k = np.logspace(-5.1, 2, 500)
    len_k = len(k)
    PK_LIN_DIR = datavectors_file_path + '_pklin.npy'
    PK_NONLIN_DIR = datavectors_file_path + '_pknonlin.npy'
    if rank == 0:
        samples = np.load(parameters_file,allow_pickle=True)
        total_num_dvs = len(samples)

        param_info = samples[0:total_num_dvs:num_ranks]#reading for 0th rank input
        for i in range(1,num_ranks):#sending other ranks' data
            comm.send(
                samples[i:total_num_dvs:num_ranks], 
                dest = i, 
                tag  = 1
            )
                
    else:
            
        param_info = comm.recv(source = 0, tag = 1)

            
    num_datavector = len(param_info)


    PKLIN = np.zeros(
            (num_datavector, len_z, len_k), dtype = "float32"
        )
    PKNONLIN = np.zeros(
            (num_datavector, len_z, len_k), dtype = "float32"
        ) 


    for i in range(num_datavector):
        input_params = model.parameterization.to_input(param_info[i])
        #print(input_params)
        input_params.pop("As", None)

        try:
            model.logposterior(input_params)
            theory = list(model.theory.values())[1]
            lin_pk = theory.get_Pk_interpolator(
                ("delta_tot", "delta_tot"), nonlinear=False, extrap_kmin = 5e-6, extrap_kmax = 100)
            nonlin_pk = theory.get_Pk_interpolator(
                ("delta_tot", "delta_tot"), nonlinear=True, extrap_kmin = 5e-6, extrap_kmax = 100)

                
        except:
            print('fail')
        else:
            PKLIN[i] = lin_pk.P(z_mps, k)
            PKNONLIN[i] = nonlin_pk.P(z_mps, k)

    if rank == 0:
        result_pklin   = np.zeros((total_num_dvs, len_z, len_k), dtype="float32")
        result_pknonlin   = np.zeros((total_num_dvs, len_z, len_k), dtype="float32")

            
        result_pklin[0:total_num_dvs:num_ranks] = PKLIN
        result_pknonlin[0:total_num_dvs:num_ranks] = PKNONLIN

        for i in range(1,num_ranks):        
            result_pklin[i:total_num_dvs:num_ranks] = comm.recv(source = i, tag = 10)
            result_pknonlin[i:total_num_dvs:num_ranks] = comm.recv(source = i, tag = 11)

        np.save(PK_LIN_DIR, result_pklin)
        np.save(PK_NONLIN_DIR, result_pknonlin)
            
    else:    
        comm.send(PKLIN, dest = 0, tag = 10)
        comm.send(PKNONLIN, dest = 0, tag = 11)




#mpirun -n 5 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
#     --bind-to core --map-by core --report-bindings --mca mpi_yield_when_idle 1 \
#    python datageneratormps.py \

