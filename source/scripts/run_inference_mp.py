import os, sys, getpass

sys.path.append(f'/home/{getpass.getuser()}/master_thesis/source/')

from simulations.pomdp import POMDPSimulation
from ctbn.parameter_learning import *
from utils.visualization import *
from utils.constants import *
from utils.helpers import *
from inference.sampling import *

import seaborn as sns

sns.set()
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import logging
import yaml


def create_folder_tag(conf):
    n_train = conf[N_TRAIN]
    n_test = conf[N_TEST]
    n_obs_model = conf[N_OBS_MODEL]
    t_max = conf[T_MAX]
    policy_type = conf[POLICY]
    b_type = conf[B_UPDATE]
    n_par = conf[N_PARTICLE] if b_type == 'particle_filter' else ''
    seed = conf[SEED]
    tag = f'_{t_max}sec_{n_train}train_{n_test}test_{n_obs_model}model_{policy_type}Policy_{b_type}{n_par}_seed{seed}'
    return tag


def generate_dataset(pomdp_, n_samples, path_to_save):
    def generate_trajectory(pomdp_, k, path_to_csv, path_to_plot):
        np.random.seed(k)
        df_traj = pomdp_.sample_trajectory()
        df_traj.loc[:, TRAJ_ID] = k

        df_traj.to_csv(os.path.join(path_to_csv, f'traj_{k}.csv'))
        pomdp_.df_b.to_csv(os.path.join(path_to_csv, f'belief_traj_{k}.csv'))
        pomdp_.df_Qz.to_csv(os.path.join(path_to_csv, f'Q_traj_{k}.csv'))
        visualize_pomdp_simulation(df_traj, pomdp_.df_b[pomdp_.S], pomdp_.df_Qz[['01', '10']],
                                   path_to_save=path_to_plot, tag=str(k))
        print(llh_inhomogenous_mp(df_traj, pomdp_.df_Qz),
              marginalized_llh_homogenous_mp(df_traj, params=pomdp_.cfg[GAMMA_PARAMS], node=parent_list_[0]),
              marginalized_llh_homogenous_mp(df_traj, params=pomdp_.cfg[GAMMA_PARAMS], node=parent_list_[1]))
        return df_traj

    data_folder = path_to_save + f'/dataset'
    csv_folder = data_folder + '/csv'
    os.makedirs(csv_folder, exist_ok=True)

    traj_list = Parallel(n_jobs=25)(
        delayed(generate_trajectory)(pomdp_, traj_id, csv_folder, data_folder) for traj_id in range(1, n_samples + 1))
    df_all = pd.concat(traj_list)
    return df_all


def inference_per_obs_model(pomdp_, df_all_, path_to_save):
    def infer_trajectory(pomdp_, df_traj, path_to_save):
        traj_id = df_traj.loc[0, TRAJ_ID]

        df_Q = get_complete_df_Q(pomdp_, df_traj, traj_id, path_to_save=path_to_save)

        llh_X3 = llh_inhomogenous_mp(df_traj, df_Q)
        if pomdp_.cfg[B_UPDATE] == "particle_filter":
            marg_llh_X1 = marginalized_llh_homogenous_mp(df_traj, params=pomdp_.cfg[GAMMA_PARAMS], node=parent_list_[0])
            marg_llh_X2 = marginalized_llh_homogenous_mp(df_traj, params=pomdp_.cfg[GAMMA_PARAMS], node=parent_list_[1])
            llh_data = llh_X3 + marg_llh_X1 + marg_llh_X2
        else:
            llh_X1 = llh_homogenous_mp(df_traj, pomdp_.parent_ctbn.Q[parent_list_[0]], node=parent_list_[0])
            llh_X2 = llh_homogenous_mp(df_traj, pomdp_.parent_ctbn.Q[parent_list_[1]], node=parent_list_[1])
            llh_data = llh_X3 + llh_X1 + llh_X2
        return llh_data

    llh_list = Parallel(n_jobs=20)(
        delayed(infer_trajectory)(pomdp_, df_traj, path_to_save) for _, df_traj in list(df_all_.groupby(by=TRAJ_ID)))
    return llh_list


def get_complete_df_Q(pomdp_, df_orig, traj_id, path_to_save=None):
    df_orig.loc[:, OBS] = df_orig.apply(pomdp_.get_observation, axis=1)

    pomdp_.get_belief_traj(df_orig)
    pomdp_.update_cont_Q()

    pomdp_.df_Qz[T_DELTA] = np.append(to_decimal(pomdp_.time_grain),
                                      np.diff(pomdp_.df_Qz.index)).astype(float)
    pomdp_.df_Qz.fillna(method='ffill', inplace=True)

    if path_to_save:
        csv_path = os.path.join(path_to_save, 'csv')
        os.makedirs(csv_path, exist_ok=True)
        pomdp_.df_b.to_csv(os.path.join(csv_path, f'belief_traj_{traj_id}.csv'))
        pomdp_.df_Qz.to_csv(os.path.join(csv_path, f'Q_traj_{traj_id}.csv'))
        visualize_pomdp_simulation(df_orig, pomdp_.df_b[pomdp_.S], pomdp_.df_Qz[['01', '10']],
                                   node_list=[r'$X_{1}$', r'$X_{2}$', r'y'], path_to_save=path_to_save,
                                   tag=str(traj_id))
    return pomdp_.df_Qz


def run_test(df_llh, phi_set, n_train, n_test, path_to_save):
    df_train = df_llh.head(n_train)
    df_test = df_llh.tail(n_test)

    df_test_result = pd.DataFrame(columns=['Number of trajectories', 'Test log-likelihood'])
    for i in range(1, n_train + 1):
        df_test_run = pd.DataFrame(columns=['Number of trajectories', 'Test log-likelihood'])
        df_train_run = df_train.head(i)
        # pred = phi_set[int(df_train_run.sum(axis=0).idxmax().split('_')[-1][0])]
        pred_tag = r'$\psi_{}$'.format(int(df_train_run.sum(axis=0).idxmax().split('_')[-1][0]))
        print(f'Number of trajectories:{i}, prediction:{pred_tag}')
        df_test_run['Test log-likelihood'] = df_test[pred_tag]
        df_test_run['Number of trajectories'] = i
        df_test_result = pd.concat([df_test_result, df_test_run])
        df_test_result.reset_index(drop=True, inplace=True)

    plt.figure()
    ax = sns.lineplot(x="Number of trajectories", y="Test log-likelihood", data=df_test_result)
    plt.savefig(os.path.join(path_to_save + '/test_likelihood.png'))


def run(pomdp_, psi_set, n_samp, run_folder, IMPORT_DATA=None):
    conf = pomdp_.cfg
    # Generate (or read) dataset
    if IMPORT_DATA:
        df_all = pd.read_csv(f'{IMPORT_DATA}/dataset.csv', index_col=0)
        df_all = df_all[df_all[TRAJ_ID] <= n_samp]
    else:
        df_all = generate_dataset(pomdp_, n_samp, path_to_save=run_folder)
    df_all.to_csv(os.path.join(run_folder, 'dataset.csv'))

    df_L = pd.DataFrame()

    for i, obs_model in enumerate(psi_set):
        inference_folder = run_folder + f'/inference/obs_model_{i}'
        os.makedirs(inference_folder, exist_ok=True)

        pomdp_.reset()
        pomdp_.reset_obs_model(obs_model)
        np.random.seed(i)
        L = inference_per_obs_model(pomdp_, df_all, path_to_save=inference_folder)

        df_L[r'$\psi_{}$'.format(i)] = L
        df_L_norm = df_L.cumsum().div((df_L.index + 1), axis=0)

        plt.figure()
        df_L_norm.head(conf[N_TRAIN]).plot()
        plt.xlabel('Number of trajectories')
        plt.ylabel('Average log-likelihood')
        plt.savefig(os.path.join(run_folder, 'llh.png'))
        plt.close()

        print(obs_model, np.sum(L))
        df_L.to_csv(os.path.join(run_folder, 'llh.csv'))
    return df_L


if __name__ == "__main__":
    t0 = time.time()
    config_file = '../configs/inference_mp.yaml'
    main_folder = '../_data/inference_mp/'

    # READING AND SAVING CONFIG
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    IMPORT_DATA = config['import_data']
    IMPORT_PSI = config['import_psi']

    n_samples = config[N_TRAIN] + config[N_TEST]
    run_folder = create_folder_for_experiment(folder_name=main_folder, tag=create_folder_tag(config))

    np.random.seed(config[SEED])
    pomdp_sim = POMDPSimulation(config, save_folder=run_folder)
    print(pomdp_sim.parent_ctbn.Q)

    if pomdp_sim.policy_type == 'function':
        np.save(os.path.join(run_folder, 'policy.npy'), pomdp_sim.policy)
    else:
        pomdp_sim.policy.to_csv(os.path.join(run_folder, 'policy.csv'))

    config['T'] = pomdp_sim.T.tolist()
    config['Q3'] = pomdp_sim.Qz
    config['parent_Q'] = pomdp_sim.parent_ctbn.Q

    with open(os.path.join(run_folder, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    if IMPORT_PSI:
        psi_subset = np.load('../configs/psi_set_3.npy')
    else:
        psi_subset = get_downsampled_obs_set(config[N_OBS_MODEL], pomdp_sim.Z)
    np.save(os.path.join(run_folder, 'psi_set.npy'), psi_subset)

    import_folder = main_folder + str(IMPORT_DATA) if IMPORT_DATA else None
    df_L = run(pomdp_sim, psi_subset, n_samples, run_folder, IMPORT_DATA=import_folder)

    run_test(df_L, psi_subset, config[N_TRAIN], config[N_TEST], run_folder)

    t1 = time.time()
    print(f'It has been {np.round((t1 - t0) / 3600, 3)} hours...PHEW!')
