from simulations.pomdp import POMDPSimulation
from utils.constants import *
from utils.helpers import *
from utils.visualization import *
import matplotlib.pyplot as plt
from inference.sampling import *
from scripts.run_inference_mp import get_complete_df_Q

import yaml
import os


if __name__ == "__main__":
    IMPORT = None  #'1590537929'  #
    run_folder = create_folder_for_experiment(folder_name='../_data/debug')

    with open('../configs/inference_mp.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # GENERATE DATA
    np.random.seed(config[SEED])
    pomdp_sim = POMDPSimulation(config)
    print(pomdp_sim.parent_ctbn.Q)

    np.random.seed(config[SEED])
    if IMPORT:
        df_traj = pd.read_csv(f'../_data/debug/{IMPORT}/df_traj.csv', index_col=0)
        df_Q = pd.read_csv(f'../_data/debug/{IMPORT}/df_Qz.csv', index_col=0)
    else:
        df_traj = pomdp_sim.sample_trajectory()
        df_Q = pomdp_sim.df_Qz

    # SAVE AND VISUALIZE DATA
    df_traj.to_csv(os.path.join(run_folder, 'df_traj.csv'))
    pomdp_sim.belief_updater.df_belief.to_csv(os.path.join(run_folder, 'df_belief.csv'))
    if pomdp_sim.POLICY_TYPE == DET_FUNC:
        np.save(os.path.join(run_folder, 'policy.npy'), pomdp_sim.policy)
        b_debug = generate_belief_grid(step=0.01, cols=['00', '01', '10', '11'])
        plt.figure()
        plt.plot((b_debug * pomdp_sim.policy).sum(axis=1).round())
        plt.plot((b_debug * pomdp_sim.policy).sum(axis=1))
        plt.savefig(run_folder + '/policy_debug.png')
    else:
        pomdp_sim.policy.to_csv(os.path.join(run_folder, 'policy.csv'))
    pomdp_sim.df_Qz.to_csv(os.path.join(run_folder, 'df_Qz.csv'))
    visualize_pomdp_simulation(df_traj, pomdp_sim.belief_updater.df_belief[pomdp_sim.S], pomdp_sim.df_Qz[['01', '10']],
                               path_to_save=run_folder, tag='', belief_method=config[B_UPDATE_METHOD])

    # LIKELIHOOD OF INHOMOGENOUS PROCESS
    llh_X3_true = llh_inhomogenous_mp(df_traj, df_Q)

    # LIKELIHOOD OF HOMOGENOUS PROCESS
    llh_X1 = llh_homogenous_mp(df_traj, pomdp_sim.parent_ctbn.Q[parent_list_[0]], node=parent_list_[0])
    llh_X2 = llh_homogenous_mp(df_traj, pomdp_sim.parent_ctbn.Q[parent_list_[1]], node=parent_list_[1])

    # MARGINAL LIKELIHOOD OF HOMOGENOUS PROCESS
    marg_llh_X1 = marginalized_llh_homogenous_mp(df_traj, params=pomdp_sim.config[GAMMA_PARAMS], node=parent_list_[0])
    marg_llh_X2 = marginalized_llh_homogenous_mp(df_traj, params=pomdp_sim.config[GAMMA_PARAMS], node=parent_list_[1])

    # INFERENCE
    df_Q_inferred = get_complete_df_Q(pomdp_sim, df_traj, 1, path_to_save=run_folder)

    llh_X3_inf = llh_inhomogenous_mp(df_traj, df_Q_inferred)
    llh_inference = llh_X3_inf + marg_llh_X1 + marg_llh_X2
