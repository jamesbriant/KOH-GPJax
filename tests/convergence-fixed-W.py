import sys
from models import toymodel as KOHmodel
from data.dataloader import DataLoader
from kohgpjax.mappings import mapRto01, map01toR, mapRto0inf, map0inftoR
from jax import jit, grad

import numpy as np
# import matplotlib.pyplot as plt

import mici

n_warm_up_iter = 200

################## Parse arguments ##################
if int(sys.argv[1]) == 0:
    sampler_name = 'dynamic'
    n_main_iters = [30, 50, 75, 100, 150, 200, 300, 500, 700, 900, 1200]
elif int(sys.argv[1]) == 1:
    if len(sys.argv) < 3:
        raise ValueError('Please provide number of leapfrog steps')
    n_steps = int(sys.argv[2])
    sampler_name = f'static-{n_steps}'
    n_main_iters = [30, 50, 75, 100, 150, 200, 300, 500, 700, 900, 1200]
else:
    raise ValueError('Invalid sampler')


################## Load data ##################
dataloader = DataLoader('data/toy/field_sin.csv', 'data/toy/sim_sin.csv')
data = dataloader.get_data()
model = KOHmodel.Model(*data)

tmax = dataloader.t_max
tmin = dataloader.t_min

################## MCMC setup ##################
theta_0 = 0.5

ell_eta_0_0 = 1 # np.sqrt(np.var(dataloader.xf))/3
ell_eta_1_0 = 0.3 # np.sqrt(np.var(dataloader.tc))/3
ell_delta_0_0 = 1 # np.sqrt(np.var(dataloader.xf))/5

lambda_eta_0 = 1
lambda_delta_0 = 30
lambda_epsilon_0 = 400
lambda_epsilon_eta_0 = 10000

init_states = np.array([[
    map01toR(theta_0), 
    map0inftoR(ell_eta_0_0),
    map0inftoR(ell_eta_1_0),
    map0inftoR(ell_delta_0_0),
    map0inftoR(lambda_eta_0),
    map0inftoR(lambda_delta_0),
    map0inftoR(lambda_epsilon_0),
    map0inftoR(lambda_epsilon_eta_0),
]])

param_transform_mici_to_gpjax = lambda x: [
    [ # theta (calibration) parameters
        mapRto01(x[0]),
    ],
    [ # lengthscale parameters
        mapRto0inf(x[1]), 
        mapRto0inf(x[2]), 
        mapRto0inf(x[3]),
    ],
    [ # lambda (variance) parameters
        mapRto0inf(x[4]), 
        mapRto0inf(x[5]), 
        mapRto0inf(x[6]), 
        mapRto0inf(x[7]),
    ]
]

jitted_neg_log_posterior_density = jit(
    model.get_KOH_neg_log_pos_dens_func(
        param_transform_mici_to_gpjax
    )
)
grad_neg_log_posterior_density = jit(grad(
    model.get_KOH_neg_log_pos_dens_func(
        param_transform_mici_to_gpjax
    )
))

def neg_log_pos_dens(x):
    return np.asarray(jitted_neg_log_posterior_density(x))

def grad_neg_log_pos_dens(x):
    return np.asarray(grad_neg_log_posterior_density(x))

##### Mici #####
system = mici.systems.EuclideanMetricSystem(
    neg_log_dens=neg_log_pos_dens,
    grad_neg_log_dens=grad_neg_log_pos_dens,
)
integrator = mici.integrators.LeapfrogIntegrator(system)


################## Run MCMC ##################
seed = 1234
n_chain = 1
rng = np.random.default_rng(seed)

##### Mici sampler and adapters #####
if sampler_name == 'dynamic':
    sampler = mici.samplers.DynamicMultinomialHMC(system, integrator, rng)
else:
    sampler = mici.samplers.StaticMetropolisHMC(system, integrator, rng, n_step=n_steps)

adapters = [
    mici.adapters.DualAveragingStepSizeAdapter(0.8),
    mici.adapters.OnlineCovarianceMetricAdapter()
]

def trace_func(state):
    return {
        'm_theta': state.pos[0], 
        'm_ell_eta_0': state.pos[1], 
        'm_ell_eta_1': state.pos[2],
        'm_ell_delta_0': state.pos[3],
        'm_lambda_eta': state.pos[4],
        'm_lambda_delta': state.pos[5],
        'm_lambda_epsilon': state.pos[6],
        'm_lambda_epsilon_eta': state.pos[7],
        'hamiltonian': system.h(state)
    }


params_transformed_mean = {}
params_transformed_std = {}

for N in n_main_iters:
    final_states, traces, stats = sampler.sample_chains(
        n_warm_up_iter, 
        N, 
        init_states, 
        adapters=adapters, 
        n_process=n_chain, # only 1 works on MacOS
        trace_funcs=[trace_func]
    )

    for var, trace in traces.items():
        if var == 'hamiltonian':
            continue
        var_name = var.split('m_')[1]
        if var_name == 'theta':
            params_transformed_mean.setdefault(var_name, []).append(np.mean(mapRto01(trace[0])*(tmax-tmin) + tmin))
            params_transformed_std.setdefault(var_name, []).append(np.std(mapRto01(trace[0])*(tmax-tmin) + tmin))
        elif var_name.startswith('ell'):
            params_transformed_mean.setdefault(var_name, []).append(np.mean(mapRto0inf(trace[0])))
            params_transformed_std.setdefault(var_name, []).append(np.std(mapRto0inf(trace[0])))
        elif var_name.startswith('lambda'):
            params_transformed_mean.setdefault(var_name, []).append(np.mean(mapRto0inf(trace[0])))
            params_transformed_std.setdefault(var_name, []).append(np.std(mapRto0inf(trace[0])))

    # print(params_transformed_mean)

np.savez(
    f"convergence/params-transformed-fixed-W-{n_warm_up_iter}-{sampler_name}.npz", 
    n_main_iters=n_main_iters,
    params_transformed_mean=params_transformed_mean, 
    params_transformed_std=params_transformed_std
)