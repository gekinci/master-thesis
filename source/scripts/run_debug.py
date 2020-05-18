from simulations.pomdp import POMDPSimulation
from utils.constants import *
from utils.helpers import *
import matplotlib.pyplot as plt

import yaml
import os


if __name__ == "__main__":
    run_folder = create_folder_for_experiment(folder_name='../_data/debug/')

    with open('../configs/inference_mp.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(run_folder, 'cfg.yaml'), 'w') as f:
        yaml.dump(config, f)

    n_samples = config[N_TRAIN] + config[N_TEST]

    np.random.seed(config[SEED])
    pomdp_sim = POMDPSimulation(config)
    print(pomdp_sim.parent_ctbn.Q)

    df_traj = pomdp_sim.sample_trajectory()

    df_traj.to_csv(os.path.join(run_folder, 'df_traj.csv'))
    pomdp_sim.df_b.to_csv(os.path.join(run_folder, 'df_belief.csv'))
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
