import sys
import numpy as np
from jax import jit
# import arviz
import mici

from models import matlabmodel_sin as KOHmodel
from data.dataloadermatlab import DataLoader
# from dataloader import DataLoader
from kohgpjax.mappings import mapRto01, map01toR, mapRto0inf, map0inftoR
from MATLAB_mappings import ell2rho, beta2ell

from time import process_time


################## Parse arguments ##################
n_warm_up_iter = int(sys.argv[1])
n_main_iter = int(sys.argv[2])


################## Load data ##################
### Use dataloadermatlab
# dataloader = DataLoader('data/matlab/simple_field.csv', 'data/matlab/simple_comp.csv')
### Use dataloader
dataloader = DataLoader('data/toy/field_sin.csv', 'data/toy/sim_sin.csv')
data = dataloader.get_data() # loads normalised/standardised data
model = KOHmodel.MatlabModel(*data)

tmax = dataloader.t_max
tmin = dataloader.t_min



################## MCMC setup ##################
theta_0 = 0.5

ell_eta_0_0 = 0.2 # np.sqrt(np.var(dataloader.xf))/3
ell_eta_1_0 = 0.2 # np.sqrt(np.var(dataloader.tc))/3
# ell_delta_0_0 = 1 # np.sqrt(np.var(dataloader.xf))/5

lambda_eta_0 = 0.1
lambda_delta_0 = 30
lambda_epsilon_0 = 400
lambda_epsilon_eta_0 = 180000

init_states = np.array([[
    map01toR(theta_0), 
    map0inftoR(ell_eta_0_0),
    map0inftoR(ell_eta_1_0),
    # map0inftoR(ell_delta_0_0), # Not used in MATLAB model
    map0inftoR(lambda_eta_0),
    map0inftoR(lambda_delta_0),
    map0inftoR(lambda_epsilon_0),
    map0inftoR(lambda_epsilon_eta_0),
]])

param_transform_mici_to_gpjax = lambda x: [
    [ # theta (calibration) parameters
        mapRto01(x[0])
    ],
    [ # lengthscale parameters
        mapRto0inf(x[1]), 
        mapRto0inf(x[2]), 
    ],
    [ # lambda (variance) parameters
        mapRto0inf(x[3]), 
        mapRto0inf(x[4]), 
        mapRto0inf(x[5]), 
        mapRto0inf(x[6])
    ]
]

jitted_neg_log_posterior_density = jit(
    model.get_KOH_neg_log_pos_dens_func(
        param_transform_mici_to_gpjax
    )
)

def neg_log_pos_dens(x):
    return np.asarray(jitted_neg_log_posterior_density(x))

##### Mici #####
system = mici.systems.EuclideanMetricSystem(
    neg_log_dens=neg_log_pos_dens,
    grad_neg_log_dens=lambda q: q * 0,
)
integrator = mici.integrators.LeapfrogIntegrator(system)



################## Run MCMC ##################
seed = 1234
n_chain = 1 # only 1 works on MacOS
# n_warm_up_iter = 5000
# n_main_iter = 5000
rng = np.random.default_rng(seed)

##### Mici sampler and adapters #####
sampler = mici.samplers.StaticMetropolisHMC(system, integrator, rng, n_step=1)
adapters = [
    mici.adapters.DualAveragingStepSizeAdapter(0.234),
    mici.adapters.OnlineCovarianceMetricAdapter()
]

def trace_func(state):
    return {
        'm_theta': state.pos[0], 
        'm_ell_eta_0': state.pos[1], 
        'm_ell_eta_1': state.pos[2],
        'm_lambda_eta': state.pos[3],
        'm_lambda_delta': state.pos[4],
        'm_lambda_epsilon': state.pos[5],
        'm_lambda_epsilon_eta': state.pos[6],
        'hamiltonian': system.h(state)
    }

start_time = process_time()
final_states, traces, stats = sampler.sample_chains(
    n_warm_up_iter, 
    n_main_iter, 
    init_states, 
    adapters=adapters, 
    n_process=n_chain, # only 1 works on MacOS
    trace_funcs=[trace_func]
)
end_time = process_time()
elapsed_time = end_time - start_time

################## Save results ##################
with open("elapsed_time_mcmc.txt", "a") as f:
    f.write(f"Warmup: {n_warm_up_iter}\nChain length: {n_main_iter}\nElapsed time: {elapsed_time:.3f} seconds\n\n")

np.savez(f"traces_{n_warm_up_iter}_{n_main_iter}.npz", **traces)

# for var, trace in traces.items():
#     print(var, ": ", np.mean(trace[0]), '±', np.std(trace[0]))