import os, sys, getpass

sys.path.append(f'/home/{getpass.getuser()}/master_thesis/source/')

import glob
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scripts.run_inference_mp import *
import math


def divisors(n):
    divs = [1]
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.extend([i, n / i])
    return np.sort(list(set(divs))).astype(int)


if __name__ == "__main__":
    t0 = time.time()
    main_folder = '../_data/roc_analysis'
    config_file = '../configs/roc_analysis.yaml'

    psi_set = np.load('../configs/psi_set_3.npy')
    n_classes = len(psi_set)

    with open(config_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    n_samples = cfg[N_TRAIN]
    IMPORT_DATA = cfg['import_data']
    run_folder = create_folder_for_experiment(folder_name=main_folder, tag=create_folder_tag(cfg))
    L_list = []

    np.random.seed(cfg[SEED])
    pomdp = POMDPSimulation(cfg, save_folder=run_folder)
    print(pomdp.parent_ctbn.Q)

    if pomdp.policy_type == 'function':
        np.save(os.path.join(run_folder, 'policy.npy'), pomdp.policy)
    else:
        pomdp.policy.to_csv(os.path.join(run_folder, 'policy.csv'))

    cfg['T'] = pomdp.T.tolist()
    cfg['Q3'] = pomdp.Qz
    cfg['parent_Q'] = pomdp.parent_ctbn.Q

    with open(os.path.join(run_folder, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    # Generating all the data
    for i, obs_model in enumerate(psi_set):
        print('psi_', i)

        pomdp.reset_obs_model(obs_model)

        psi_folder = run_folder + f'/psi_{i}'
        os.makedirs(psi_folder, exist_ok=True)

        L_list += [run(pomdp, psi_set, n_samples, psi_folder, IMPORT_DATA=IMPORT_DATA)]

    for n in divisors(n_samples):
        df_scores = pd.DataFrame()
        y_labels = None

        for i, df_loglh in enumerate(L_list):
            # Concatenate likelihoods from different datasets
            df_lh = np.exp(df_loglh)
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

        n_samples = len(df_scores)
        y_scores = df_scores.divide(df_scores.values.sum(axis=1), axis=0).values # Normalizing likelihoods

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for m in range(n_classes):
            fpr[m], tpr[m], _ = roc_curve(y_labels[:, m], y_scores[:, m])
            roc_auc[m] = auc(fpr[m], tpr[m])

        plt.figure()
        c = 0
        plt.plot(fpr[c], tpr[c], color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' % roc_auc[c])
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(r'ROC curve $\psi_{0}$ vs. ' + f'all (n={n})')
        plt.legend(loc="lower right")
        plt.savefig(run_folder + f'/AUROC_{n_samples*n_classes}samples_class{c}_llh_n{n}.png')
        # plt.show()

    t0 = time.time()
    print(f'It has been {np.round((t1 - t0) / 3600, 3)} hours...PHEW!')
