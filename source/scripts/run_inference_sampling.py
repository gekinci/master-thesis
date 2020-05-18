import os, sys, getpass

sys.path.append(f'/home/{getpass.getuser()}/master_thesis/source/')

from simulations.pomdp import POMDPSimulation
from ctbn.parameter_learning import *
from utils.visualization import *
from utils.constants import *
from utils.helpers import *
from inference.sampling import *

import seaborn as sns; sns.set()
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
    n_par = conf[N_PARTICLE] if b_type == PART_FILT else ''
    seed = conf[SEED]

    if conf[MARGINALIZE]:
        tag = f'_{t_max}sec_{n_train}train_{n_test}test_{n_obs_model}model_{policy_type}Policy_{b_type}{n_par}_marginalized_seed{seed}'
    else:
        tag = f'_{t_max}sec_{n_train}train_{n_test}test_{n_obs_model}model_{policy_type}Policy_{b_type}{n_par}_seed{seed}'

    return tag


def generate_dataset(pomdp_, n_samples, path_to_save, IMPORT_DATA=None):
    if IMPORT_DATA:
        df_all = pd.read_csv(f'../_data/inference_sampling/{IMPORT_DATA}/dataset.csv', index_col=0)
        df_all = df_all[df_all[TRAJ_ID] <= n_samples]
    else:
        df_all = pd.DataFrame()
        data_folder = path_to_save + f'/dataset'
        csv_folder = data_folder + '/csv'
        os.makedirs(csv_folder, exist_ok=True)

        for k in range(1, n_samples + 1):
            np.random.seed(k)
            df_traj = pomdp_.sample_trajectory()
            df_traj.loc[:, TRAJ_ID] = k

            df_traj.to_csv(os.path.join(csv_folder, f'traj_{k}.csv'))
            pomdp_.df_b.to_csv(os.path.join(csv_folder, f'belief_traj_{k}.csv'))
            pomdp_.df_Qz.to_csv(os.path.join(csv_folder, f'Q_traj_{k}.csv'))
            visualize_pomdp_simulation(df_traj, pomdp_.df_b[pomdp_.S], pomdp_.df_Qz[['01', '10']],
                                       path_to_save=data_folder, tag=str(k))
            df_all = df_all.append(df_traj)
            print(llh_inhomogenous_mp(df_traj, pomdp_.df_Qz),
                  llh_homogenous_mp(df_traj, pomdp_.parent_ctbn.Q[parent_list_[0]], node=parent_list_[0]),
                  llh_homogenous_mp(df_traj, pomdp_.parent_ctbn.Q[parent_list_[1]], node=parent_list_[1]))
    return df_all, pomdp_


def get_complete_df_Q(pomdp_, df_orig, traj_id, path_to_save=None):
    df_orig.loc[:, OBS] = df_orig.apply(pomdp_.get_observation, axis=1)

    pomdp_.get_belief_traj(df_orig)
    pomdp_.update_cont_Q()

    pomdp_.df_Qz[T_DELTA] = np.append(to_decimal(pomdp_.time_increment),
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
    for i in range(1, n_train+1):
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


if __name__ == "__main__":
    IMPORT_TRAJ = '1589127818_3sec_20train_0test_3model_deterministicPolicy_particle_filter100_seed3'  # None  #
    IMPORT_PSI = True
    t0 = time.time()

    # READING AND SAVING CONFIG
    with open('../configs/inference_sampling.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    folder = create_folder_for_experiment(folder_name='../_data/inference_sampling/', tag=create_folder_tag(cfg))

    np.random.seed(cfg[SEED])
    pomdp_sim = POMDPSimulation(cfg)

    if pomdp_sim.POLICY_TYPE == DET_FUNC:
        np.save(os.path.join(folder, 'policy.npy'), pomdp_sim.policy)
    else:
        pomdp_sim.policy.to_csv(os.path.join(folder, 'policy.csv'))

    cfg['T'] = pomdp_sim.T.tolist()
    cfg['Q3'] = pomdp_sim.Qset
    cfg['parent_Q'] = pomdp_sim.parent_ctbn.Q

    with open(os.path.join(folder, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    n_samples = cfg[N_TRAIN] + cfg[N_TEST]

    # Generate (or read) dataset
    df_all, pomdp_sim = generate_dataset(pomdp_sim, n_samples, path_to_save=folder, IMPORT_DATA=IMPORT_TRAJ)
    df_all.to_csv(os.path.join(folder, 'dataset.csv'))

    if IMPORT_PSI:
        psi_subset = np.load('../_data/inference_sampling/psi_set_3.npy')
    else:
        psi_subset = get_downsampled_obs_set(cfg[N_OBS_MODEL], pomdp_sim.PSI)
    np.save(os.path.join(folder, 'psi_set.npy'), psi_subset)

    df_L = pd.DataFrame()

    for i, obs_model in enumerate(psi_subset):
        np.random.seed(cfg[SEED])
        run_folder = folder + f'/inference/obs_model_{i}'
        os.makedirs(run_folder, exist_ok=True)

        pomdp_sim.reset_obs_model(obs_model)

        L = []
        grouped_by_traj = df_all.groupby(by=TRAJ_ID)

        for traj_id, df_traj in list(grouped_by_traj):
            run_folder_ = run_folder
            os.makedirs(run_folder_, exist_ok=True)

            pomdp_sim.reset()
            df_Q = get_complete_df_Q(pomdp_sim, df_traj, traj_id, path_to_save=run_folder_)
            llh_X3 = llh_inhomogenous_mp(df_traj, df_Q)
            llh_X1 = llh_homogenous_mp(df_traj, pomdp_sim.parent_ctbn.Q[parent_list_[0]], node=parent_list_[0])
            llh_X2 = llh_homogenous_mp(df_traj, pomdp_sim.parent_ctbn.Q[parent_list_[1]], node=parent_list_[1])
            if cfg[MARGINALIZE]:
                marg_llh_X1 = marginalized_llh_homogenous_mp(df_traj, params=cfg[GAMMA_PARAMS], node='X1')
                marg_llh_X2 = marginalized_llh_homogenous_mp(df_traj, params=cfg[GAMMA_PARAMS], node='X2')
                llh_data = llh_X3 + marg_llh_X1 + marg_llh_X2
            else:
                llh_data = llh_X3 + llh_X1 + llh_X2
            L += [llh_data]

        df_L[r'$\psi_{}$'.format(i)] = L
        df_L_norm = df_L.cumsum().div((df_L.index + 1), axis=0)

        plt.figure()
        df_L_norm.head(cfg[N_TRAIN]).plot()
        plt.xlabel('Number of trajectories')
        plt.ylabel('Average log-likelihood')
        plt.savefig(os.path.join(folder, 'llh.png'))
        plt.close()

        print(obs_model, np.sum(L))
        df_L.to_csv(os.path.join(folder, 'llh.csv'))

    run_test(df_L, psi_subset, cfg[N_TRAIN], cfg[N_TEST], folder)

    t1 = time.time()
    print(f'It has been {(t1 - t0) / 3600} hours...PHEW!')
