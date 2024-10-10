from models import toymodel as KOHmodel
from data.dataloader import DataLoader
from kohgpjax.mappings import mapRto01, map01toR, mapRto0inf, map0inftoR
from jax import jit, grad

import numpy as np
import matplotlib.pyplot as plt

import mici


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
samplers = {
    'static_3': mici.samplers.StaticMetropolisHMC(system, integrator, rng, n_step=3),
    'dynamic': mici.samplers.DynamicMultinomialHMC(system, integrator, rng)
}

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


# n_warm_up_iter = [30, 50, 75, 100, 150, 200, 300, 500, 700, 900, 1200]
n_warm_up_iter = [50, 70]
n_main_iters = 500

################## Iterate over samplers ##################
for sampler_name, sampler in samplers.items():
    params_transformed_mean = {}
    params_transformed_std = {}

    for W in n_warm_up_iter:
        final_states, traces, stats = sampler.sample_chains(
            W, 
            n_main_iters, 
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

        print(params_transformed_mean)


    ################## Save results ##################
    fig, axes = plt.subplots(4, 2, figsize=(8, 12))

    for i, var in enumerate(params_transformed_mean):
        ax = axes.flatten()[i]
        ax.set_xscale('log')
        ax.plot(n_warm_up_iter, params_transformed_mean[var], 'o-', label='mean')
        ax.fill_between(
            n_warm_up_iter, 
            np.array(params_transformed_mean[var])-np.array(params_transformed_std[var]), 
            np.array(params_transformed_mean[var])+np.array(params_transformed_std[var]), 
            alpha=0.3
        )
        ax.set_xlabel('Number of warm-up iterations, W')
        ax.set_ylabel('mean')

        ax2 = ax.twinx()
        ax2.plot(n_warm_up_iter, params_transformed_std[var], 'x--', color='tab:orange', label='std')
        ax2.set_ylabel('std')

        ax.legend(loc=2)
        ax2.legend(loc=1)

        ax.set_title(var)

    axes[0,0].axhline(0.4, color='k', linestyle='--')
    axes[3,0].axhline(400, color='k', linestyle='--')
    fig.suptitle(f'Convergence of parameters for {sampler_name} sampler\nN={n_warm_up_iter}')
    plt.tight_layout()
    plt.savefig(f'convergence/params-convergence-fixed-N-{sampler_name}.png')
    plt.close()