from simulations.pomdp import POMDPSimulation
from utils.constants import *
from utils.helpers import *
from utils.visualization import *
import matplotlib.pyplot as plt
from inference.sampling import *
from scripts.run_inference_mp import get_complete_df_Q, save_csvs, save_policy

import yaml
import os

if __name__ == "__main__":
    run_folder = create_folder_for_experiment(folder_name='../_data/debug')

    with open('../configs/inference_mp.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # GENERATE DATA
    np.random.seed(config[SEED])
    pomdp_sim = POMDPSimulation(config)
    print(pomdp_sim.parent_ctbn.Q)

    np.random.seed(config[SEED])
    df_traj = pomdp_sim.sample_trajectory()
    df_Q = pomdp_sim.df_Q_agent
    df_b = pomdp_sim.df_belief

    # SAVE AND VISUALIZE DATA
    save_policy(pomdp_sim, run_folder)
    save_csvs(df_b, df_Q, run_folder, df_traj=df_traj)
    visualize_pomdp_simulation(df_traj, df_b, df_Q, path_to_save=run_folder, tag='',
                               belief_method=config[B_UPDATE_METHOD])

    # LIKELIHOOD OF INHOMOGENOUS PROCESS
    llh_X3_true = llh_inhomogenous_mp(df_traj, df_Q)
    print(llh_X3_true)

    # LIKELIHOOD OF HOMOGENOUS PROCESS
    llh_X1 = llh_homogenous_mp(df_traj, pomdp_sim.parent_ctbn.Q[parent_list_[0]], node=parent_list_[0])
    print(llh_X1)
    llh_X2 = llh_homogenous_mp(df_traj, pomdp_sim.parent_ctbn.Q[parent_list_[1]], node=parent_list_[1])
    print(llh_X2)

    # MARGINAL LIKELIHOOD OF HOMOGENOUS PROCESS
    marg_llh_X1 = marginalized_llh_homogenous_mp(df_traj, params=pomdp_sim.config[GAMMA_PARAMS], node=parent_list_[0])
    marg_llh_X2 = marginalized_llh_homogenous_mp(df_traj, params=pomdp_sim.config[GAMMA_PARAMS], node=parent_list_[1])

    # INFERENCE
    df_Q_inferred = get_complete_df_Q(pomdp_sim, df_traj, 1, path_to_save=run_folder)

    llh_X3_inf = llh_inhomogenous_mp(df_traj, df_Q_inferred)
    llh_inference = llh_X3_inf + marg_llh_X1 + marg_llh_X2
    print(llh_X3_inf)
