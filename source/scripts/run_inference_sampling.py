from simulations.pomdp import POMDPSimulation
from utils.visualization import *
from utils.constants import *
from utils.helpers import *

import logging
import yaml
import os

if __name__ == "__main__":
    n_traj = 20

    folder = create_folder_for_experiment(folder_name='../_data/inference_sampling/')

    with open('../configs/pomdp_sim.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(folder, 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    np.random.seed(cfg[SEED])
    pomdp_sim = POMDPSimulation(cfg, save_folder=folder, import_data='1582930774')

    df_all_traj = pd.DataFrame()

    for i in range(1, n_traj + 1):
        run_folder = folder + f'/{i}'
        os.makedirs(run_folder, exist_ok=True)

        df_traj = pomdp_sim.sample_trajectory()
        df_traj.loc[:, TRAJ_ID] = i

        df_traj.to_csv(os.path.join(run_folder, 'df_traj.csv'))
        pomdp_sim.df_b.to_csv(os.path.join(run_folder, 'df_belief.csv'))
        pomdp_sim.policy.to_csv(os.path.join(run_folder, 'df_policy.csv'))
        pomdp_sim.df_Qz.to_csv(os.path.join(run_folder, 'df_Qz.csv'))

        visualize_pomdp_simulation(df_traj, pomdp_sim.df_b[pomdp_sim.S], pomdp_sim.df_Qz[['01', '10']],
                                   path_to_save=run_folder)

        df_all_traj = df_all_traj.append(df_traj)

        pomdp_sim.reset()

    df_all_traj.to_csv(os.path.join(folder, 'df_all_traj.csv'))
