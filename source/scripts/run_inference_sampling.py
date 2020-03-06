from simulations.pomdp import POMDPSimulation
from utils.visualization import *
from utils.constants import *
from utils.helpers import *
from inference.sampling import *

import logging
import yaml
import os


def get_complete_df_Q(pomdp_, df_orig):
    df_par = pomdp_sim.sample_parent_trajectory()
    pomdp_sim.get_belief_traj(df_par)
    pomdp_sim.update_cont_Q()

    for t_ in df_orig.loc[abs(df_orig['Z'].diff()) > 0, TIME]:
        pomdp_.append_event(t_)

    pomdp_.df_Qz[T_DELTA] = np.append(to_decimal(pomdp_.time_grain),
                                      np.diff(pomdp_.df_Qz.index)).astype(float)
    pomdp_.df_Qz.fillna(method='ffill', inplace=True)
    return pomdp_.df_Qz


if __name__ == "__main__":
    n_traj = 20
    IMPORT_TRAJ = True

    folder = create_folder_for_experiment(folder_name='../_data/inference_sampling/')

    with open('../configs/pomdp_sim.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(folder, 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    t0 = time.time()

    np.random.seed(cfg[SEED])
    pomdp_sim = POMDPSimulation(cfg, save_folder=folder)

    if IMPORT_TRAJ:
        df_all_traj = pd.read_csv('../_data/inference_sampling/1583515823_20x10/df_all_traj.csv', index_col=0)
    else:
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

    np.random.seed(1)

    phi_set = obs_model_set(len(cfg[STATES])**2, len(cfg[OBS_SPACE]))
    L_list = np.zeros((len(phi_set)))

    for i, obs_model in enumerate(phi_set):
        pomdp_sim.reset_obs_model(obs_model)

        L = 0
        grouped_by_traj = df_all_traj.groupby(by=TRAJ_ID)

        for id, df_traj in list(grouped_by_traj):
            pomdp_sim.reset()
            df_Q = get_complete_df_Q(pomdp_sim, df_traj)
            L += log_likelihood_inhomogeneous_ctmc(df_traj, df_Q)

        L_list[i] = L

    print(phi_set[np.argmax(L_list)])
    np.save(os.path.join(folder, 'phi_set.npy'), phi_set)
    np.save(os.path.join(folder, 'llh.npy'), L_list)

    t1 = time.time()
    print(t1 - t0)
