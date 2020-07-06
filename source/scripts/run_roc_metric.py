import os, sys, getpass

sys.path.append(f'/home/{getpass.getuser()}/master_thesis/source/')

import glob
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import math

from utils.constants import *
from utils.helpers import *
from utils.visualization import *
from inference.sampling import *
from joblib import Parallel, delayed
from simulations.pomdp import POMDPSimulation

N_TREADS = 25


def divisors(n):
    divs = [1]
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.extend([i, n / i])
    return np.sort(list(set(divs))).astype(int)


def create_folder_tag(conf):
    n_train = conf[N_TRAIN]
    n_test = conf[N_TEST]
    n_obs_model = conf[N_OBS_MODEL]
    t_max = conf[T_MAX]
    policy_type = conf[POLICY_TYPE]
    b_type = conf[B_UPDATE_METHOD]
    n_par = conf[N_PARTICLE] if b_type == PART_FILT else ''
    seed = conf[SEED]
    obs_model = conf[OBS_MODEL][0] if conf[OBS_MODEL] else ''
    prior = 'informative' if conf[PR_INFORM] else 'noninformative'
    p_e = conf['p_error']
    tag = f'_error{p_e}_{t_max}sec_{n_train}train_{n_test}test_{n_obs_model}model_{policy_type}Policy_{b_type}{n_par}_' \
          f'seed{seed}_{obs_model}_{prior}'
    return tag


def save_csvs(dict_b, dict_Q, path_to_save, tag='', df_traj=None):
    if df_traj is not None:
        df_traj.to_csv(os.path.join(path_to_save, f'df_traj{tag}.csv'))
    for m, df_ in dict_b.items():
        df_.to_csv(os.path.join(path_to_save, f'belief_{m}_{tag}.csv'))
    for m, df_ in dict_Q.items():
        df_.to_csv(os.path.join(path_to_save, f'Q_agent_{m}_{tag}.csv'))


def save_policy(pomdp_, path_to_save):
    if pomdp_.POLICY_TYPE == DET_FUNC:
        np.save(os.path.join(path_to_save, 'policy.npy'), pomdp_.policy)
        b_debug = generate_belief_grid(step=0.01, cols=['00', '01', '10', '11'])
        plt.figure()
        plt.plot((b_debug * pomdp_.policy).sum(axis=1).round())
        plt.plot((b_debug * pomdp_.policy).sum(axis=1))
        plt.savefig(path_to_save + '/policy_debug.png')
    else:
        pomdp_.policy.to_csv(os.path.join(path_to_save, 'policy.csv'))


def generate_dataset(pomdp_, n_samples, path_to_save, rnd_seed=0):
    def generate_trajectory(pomdp_, k, path_to_csv, path_to_plot):
        np.random.seed(int(rnd_seed) + k)
        # print('seed ', rnd_seed + k)
        df_traj = pomdp_.sample_trajectory()
        df_traj.loc[:, TRAJ_ID] = k

        dict_b = pomdp_.belief_dict
        dict_Q = {pomdp_.BELIEF_UPDATE_METHOD[0]: pomdp_.Q_agent_dict[pomdp_.BELIEF_UPDATE_METHOD[0]]}

        save_csvs(dict_b, dict_Q, path_to_csv, k, df_traj=df_traj)
        # visualize_pomdp_simulation(df_traj, dict_b, dict_Q, path_to_save=path_to_plot, tag=str(k))
        return df_traj

    data_folder = path_to_save + f'/dataset'
    csv_folder = data_folder + '/csv'
    os.makedirs(csv_folder, exist_ok=True)

    traj_list = Parallel(n_jobs=N_TREADS)(
        delayed(generate_trajectory)(pomdp_, traj_id, csv_folder, data_folder) for traj_id in range(1, n_samples + 1))
    df_all = pd.concat(traj_list)
    return df_all


def inference_per_obs_model(pomdp_, df_all_, obs_id, path_to_save, rnd_seed=0):
    def infer_trajectory(pomdp_, df_traj, path_to_save):
        traj_id = df_traj.loc[0, TRAJ_ID]
        np.random.seed(rnd_seed + int(1e3 * obs_id) + traj_id)
        # print('seed ', rnd_seed + int(1e3 * obs_id) + traj_id)

        df_Q = get_complete_df_Q(pomdp_, df_traj, traj_id, path_to_save=path_to_save)

        llh_X3 = llh_inhomogenous_mp(df_traj, df_Q)
        marg_llh_X1 = marginalized_llh_homogenous_mp(df_traj, params=pomdp_.config[GAMMA_PARAMS],
                                                     node=parent_list_[0])
        marg_llh_X2 = marginalized_llh_homogenous_mp(df_traj, params=pomdp_.config[GAMMA_PARAMS],
                                                     node=parent_list_[1])
        llh_data = {k: v + marg_llh_X1 + marg_llh_X2 for k, v in llh_X3.items()}
        return llh_data

    print('INFERENCE...')
    llh_list = Parallel(n_jobs=N_TREADS)(
        delayed(infer_trajectory)(pomdp_, df_traj, path_to_save) for _, df_traj in list(df_all_.groupby(by=TRAJ_ID)))
    return llh_list


def get_complete_df_Q(pomdp_, df_orig, traj_id, path_to_save=None):
    df_orig.loc[:, OBS] = df_orig.apply(pomdp_.get_observation, axis=1)

    pomdp_.get_belief_for_inference(df_orig)
    pomdp_.update_cont_Q()

    for m, _ in pomdp_.Q_agent_dict.items():
        pomdp_.Q_agent_dict[m][T_DELTA] = np.append(np.diff(pomdp_.Q_agent_dict[m].index), 0).astype(float)
        pomdp_.Q_agent_dict[m].fillna(method='ffill', inplace=True)

    dict_b = pomdp_.belief_dict.copy()
    dict_Q = pomdp_.Q_agent_dict.copy()

    if path_to_save:
        csv_path = os.path.join(path_to_save, 'csv')
        os.makedirs(csv_path, exist_ok=True)
        save_csvs(dict_b, dict_Q, csv_path, tag=traj_id)
        # visualize_pomdp_simulation(df_orig, dict_b, dict_Q, node_list=[r'$X_{1}$', r'$X_{2}$', r'y'],
        #                            path_to_save=path_to_save, tag=str(traj_id))
    return dict_Q


def run(pomdp_, psi_set, n_samp, run_folder, rnd_seed=0):
    print('GENERATING DATA...')
    df_all = generate_dataset(pomdp_, n_samp, path_to_save=run_folder, rnd_seed=rnd_seed)
    df_all.to_csv(os.path.join(run_folder, 'dataset.csv'))

    dict_L = {m: pd.DataFrame() for m in pomdp_.BELIEF_UPDATE_METHOD}
    rnd_seed_inference = rnd_seed + int(1e3)
    for psi_id, obs_model in enumerate(psi_set):
        inference_folder = run_folder + f'/inference/obs_model_{psi_id}'
        os.makedirs(inference_folder, exist_ok=True)

        pomdp_.reset()
        pomdp_.reset_obs_model(obs_model)
        L = inference_per_obs_model(pomdp_, df_all, psi_id,  path_to_save=inference_folder, rnd_seed=rnd_seed_inference)

        for m in pomdp_.BELIEF_UPDATE_METHOD:
            dict_L[m][r'$\psi_{}$'.format(psi_id)] = [l[m] for l in L]
        print(obs_model, {m: dict_L[m].sum()[r'$\psi_{}$'.format(psi_id)] for m in dict_L.keys()})

        for m in pomdp_.BELIEF_UPDATE_METHOD:
            dict_L[m].to_csv(os.path.join(run_folder, f'llh_{m}.csv'))
    return dict_L


if __name__ == "__main__":
    t0 = time.time()
    main_folder = '../_data/roc_analysis'
    config_file = '../configs/roc_analysis.yaml'

    psi_set = np.load('../configs/psi_set_3.npy')
    n_classes = len(psi_set)

    with open(config_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    n_sample_per_class = cfg[N_TRAIN]
    p_e = cfg['p_error']
    run_folder = create_folder_for_experiment(folder_name=main_folder, tag=create_folder_tag(cfg))
    L_list = []

    np.random.seed(cfg[SEED])
    pomdp = POMDPSimulation(cfg)
    print(pomdp.parent_ctbn.Q)

    if pomdp.POLICY_TYPE == DET_FUNC:
        np.save(os.path.join(run_folder, 'policy.npy'), pomdp.policy)
    else:
        pomdp.policy.to_csv(os.path.join(run_folder, 'policy.csv'))

    cfg['Q3'] = pomdp.Qset
    cfg['parent_Q'] = pomdp.parent_ctbn.Q

    with open(os.path.join(run_folder, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    # Generating all the data
    for i, obs_model in enumerate(psi_set):
        print('psi_', i)
        rnd_seed_run = int(i*1e6)
        obs_model += (np.ones(obs_model.shape)*(0.5)-obs_model*1.5)*p_e
        pomdp.reset_obs_model(obs_model)

        psi_folder = run_folder + f'/psi_{i}'
        os.makedirs(psi_folder, exist_ok=True)

        L_list += [
            run(pomdp, psi_set, n_sample_per_class, psi_folder, rnd_seed=rnd_seed_run)[cfg[B_UPDATE_METHOD][0]]]

    for n in divisors(n_sample_per_class):
        df_scores = pd.DataFrame()
        y_labels = None

        for i, df_loglh in enumerate(L_list):
            # Concatenate likelihoods from different datasets
            df_lh = np.exp(df_loglh)
            df_lh = df_lh.divide(df_lh.values.sum(axis=1), axis=0)  # Normalizing likelihoods
            for k in range(n):
                df_shuffled_ = df_lh.sample(frac=1).reset_index(drop=True)
                df_scores = df_scores.append(df_shuffled_.groupby(df_shuffled_.index // n).mean())

            # Create and concatenate labels for different classes
            n_class_samples = int(len(df_loglh))
            y_class_labels = np.zeros((n_class_samples, n_classes))
            y_class_labels[:, i] = 1
            if y_labels is None:
                y_labels = y_class_labels
            else:
                y_labels = np.concatenate((y_labels, y_class_labels))

        df_scores.reset_index(drop=True, inplace=True)

        n_all_samples = len(df_scores)
        y_scores = df_scores.values

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for m in range(n_classes):
            fpr[m], tpr[m], _ = roc_curve(y_labels[:, m], y_scores[:, m])
            roc_auc[m] = auc(fpr[m], tpr[m])

        plt.figure()
        c = 0
        plt.plot(fpr[c], tpr[c], color='darkorange',
                 lw=2, label='AUROC = %0.2f' % roc_auc[c])
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'n={n}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(run_folder + f'/AUROC_{n_sample_per_class * n_classes}samples_class{c}_llh_n{n}.pdf')
        # plt.show()

    t1 = time.time()
    print(f'It has been {np.round((t1 - t0) / 3600, 3)} hours...PHEW!')
