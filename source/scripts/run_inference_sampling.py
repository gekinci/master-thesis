from simulations.pomdp import POMDPSimulation
from ctbn.parameter_learning import *
from utils.visualization import *
from utils.constants import *
from utils.helpers import *
from inference.sampling import *

import matplotlib.pyplot as plt
import logging
import yaml
import os


def generate_dataset(pomdp_, n_samples, path_to_save, IMPORT_DATA=None):
    if IMPORT_DATA:
        df_all = pd.read_csv(f'../_data/inference_sampling/{IMPORT_DATA}/dataset.csv', index_col=0)
        df_all = df_all[df_all[TRAJ_ID] <= n_samples]
    else:
        df_all = pd.DataFrame()

        for k in range(1, n_samples + 1):
            data_folder = path_to_save + f'/dataset/traj_{k}'
            os.makedirs(data_folder, exist_ok=True)

            df_traj = pomdp_.sample_trajectory()
            df_traj.loc[:, TRAJ_ID] = k

            df_traj.to_csv(os.path.join(data_folder, 'traj.csv'))
            pomdp_.df_b.to_csv(os.path.join(data_folder, 'belief_traj.csv'))
            pomdp_.df_Qz.to_csv(os.path.join(data_folder, 'Q_traj.csv'))

            visualize_pomdp_simulation(df_traj, pomdp_.df_b[pomdp_.S], pomdp_.df_Qz[['01', '10']],
                                       path_to_save=data_folder)

            df_all = df_all.append(df_traj)

            pomdp_.reset()

    return df_all, pomdp_


def get_complete_df_Q(pomdp_, df_orig, path_to_save=None):
    df_orig.loc[:, OBS] = df_orig.apply(pomdp_.get_observation, axis=1)

    pomdp_.get_belief_traj(df_orig)
    pomdp_.update_cont_Q()

    pomdp_.df_Qz[T_DELTA] = np.append(to_decimal(pomdp_.time_grain),
                                      np.diff(pomdp_.df_Qz.index)).astype(float)
    pomdp_.df_Qz.fillna(method='ffill', inplace=True)

    if path_to_save:
        pomdp_.df_b.to_csv(os.path.join(path_to_save, 'belief_traj.csv'))
        pomdp_.df_Qz.to_csv(os.path.join(path_to_save, 'Q_traj.csv'))

        visualize_pomdp_simulation(df_orig, pomdp_.df_b[pomdp_.S], pomdp_.df_Qz[['01', '10']],
                                   node_list=['X', 'Y', 'o'], path_to_save=path_to_save)
    return pomdp_.df_Qz


if __name__ == "__main__":
    IMPORT_TRAJ = None
    t0 = time.time()

    # READING AND SAVING CONFIG
    with open('../configs/inference_sampling.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    n_traj = cfg[N_TRAJ]
    n_obs_model = cfg[N_OBS_MODEL]
    t_max = cfg[T_MAX]
    policy_type = cfg[POLICY]

    if cfg[MARGINALIZE]:
        tag = f'_{t_max}sec_{n_traj}traj_{n_obs_model}model_{policy_type}Policy_marginalized'
    else:
        tag = f'_{t_max}sec_{n_traj}traj_{n_obs_model}model_{policy_type}Policy'

    folder = create_folder_for_experiment(folder_name='../_data/inference_sampling/',
                                          tag=tag)

    np.random.seed(cfg[SEED])

    pomdp_sim = POMDPSimulation(cfg, save_folder=folder)

    if pomdp_sim.policy_type == 'function':
        np.save(os.path.join(folder, 'policy.npy'), pomdp_sim.policy_func)
    else:
        pomdp_sim.df_policy.to_csv(os.path.join(folder, 'policy.csv'))

    cfg['T'] = pomdp_sim.T.tolist()
    cfg['Qz'] = pomdp_sim.Qz

    with open(os.path.join(folder, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    # Generate (or read) dataset
    df_all_traj, pomdp_sim = generate_dataset(pomdp_sim, n_traj, path_to_save=folder, IMPORT_DATA=IMPORT_TRAJ)
    df_all_traj.to_csv(os.path.join(folder, 'dataset.csv'))

    # np.random.seed(0)

    # phi_subset = get_downsampled_obs_set(n_obs_model, pomdp_sim.Z)
    phi_subset = np.load('../_data/inference_sampling/phi_set_3.npy')
    np.save(os.path.join(folder, 'phi_set.npy'), phi_subset)

    df_L = pd.DataFrame()

    for i, obs_model in enumerate(phi_subset):
        run_folder = folder + f'/inference/obs_model_{i}'
        os.makedirs(run_folder, exist_ok=True)

        pomdp_sim.reset_obs_model(obs_model)

        L = []
        grouped_by_traj = df_all_traj.groupby(by=TRAJ_ID)

        for traj_id, df_traj in list(grouped_by_traj):
            run_folder_ = run_folder + f'/traj_{traj_id}'
            os.makedirs(run_folder_, exist_ok=True)

            pomdp_sim.reset()
            df_Q = get_complete_df_Q(pomdp_sim, df_traj, path_to_save=run_folder_)
            llh_Z = llh_inhomogenous_ctbn(df_traj, df_Q)
            if cfg[MARGINALIZE]:
                marg_log_p = marginalized_log_prob_of_homogenous_ctbn(df_traj, params=cfg[GAMMA_PARAMS])
                llh_data = llh_Z + marg_log_p
            else:
                llh_data = llh_Z
            L += [llh_Z]

        df_L[f'obs_{i}'] = L
        df_L_norm = df_L.cumsum().div((df_L.index + 1), axis=0)

        plt.figure()
        df_L_norm.plot()
        plt.savefig(os.path.join(folder, 'llh.png'))
        plt.close()

        print(obs_model, np.sum(L))
        df_L.to_csv(os.path.join(folder, 'llh.csv'))

    print('Maximum likely obs model:')
    print(phi_subset[int(df_L.sum(axis=0).idxmax().split('_')[-1])])
    t1 = time.time()
    print(f'It has been {(t1 - t0) / 3600} hours...PHEW!')
