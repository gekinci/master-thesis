import os
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
    main_folder = '../_data/roc_analysis'
    config_file = '../configs/inference_mp.yaml'

    psi_set = np.load(os.path.join(main_folder, 'psi_set_3_2.npy'))
    n_classes = len(psi_set)

    with open(config_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    n_samples = 200
    run_folder = create_folder_for_experiment(folder_name=main_folder)

    np.random.seed(cfg[SEED])
    pomdp_sim = POMDPSimulation(cfg, save_folder=run_folder)

    for i, obs_model in enumerate(psi_set):
        model_folder = run_folder + f'/psi_{i}'
        os.makedirs(model_folder, exist_ok=True)

        pomdp_sim.reset()
        pomdp_sim.reset_obs_model(obs_model)

        df_all = generate_dataset(pomdp_sim, n_samples, path_to_save=run_folder)
        df_all.to_csv(os.path.join(run_folder, 'dataset.csv'))

        for i, obs_model in enumerate(psi_subset):
            inference_folder = run_folder + f'/inference/obs_model_{i}'
            os.makedirs(inference_folder, exist_ok=True)

            np.random.seed(i)
            pomdp_sim.reset()
            pomdp_sim.reset_obs_model(obs_model)
            L = inference_per_obs_model(pomdp_sim, obs_model, df_all, path_to_save=inference_folder)

            df_L[r'$\psi_{}$'.format(i)] = L
            df_L_norm = df_L.cumsum().div((df_L.index + 1), axis=0)

            plt.figure()
            df_L_norm.head(cfg[N_TRAIN]).plot()
            plt.xlabel('Number of trajectories')
            plt.ylabel('Average log-likelihood')
            plt.savefig(os.path.join(run_folder, 'llh.png'))
            plt.close()

            print(obs_model, np.sum(L))
            df_L.to_csv(os.path.join(run_folder, 'llh.csv'))

    # # List of datasets of different classes
    # list_folders = np.sort(glob.glob(path_to_data + '/*/'))
    #
    # for n in divisors(100):
    #     df_scores = pd.DataFrame()
    #     y_labels = None
    #
    #     for i, folder in enumerate(list_folders):
    #         # Concatenate likelihoods from different datasets
    #         df_loglh = pd.read_csv(os.path.join(folder, 'llh.csv'), index_col=0)
    #         df_lh = np.exp(df_loglh)
    #         # df_scores = df_scores.append(df_lh)
    #         for k in range(n*mult):
    #             df_shuffled_ = df_lh.sample(frac=1).reset_index(drop=True)
    #             df_scores = df_scores.append(df_shuffled_.groupby(df_shuffled_.index // n).mean())
    #
    #         # Create and concatenate labels for different classes
    #         n_class_samples = int(len(df_loglh)*mult)
    #         y_class_labels = np.zeros((n_class_samples, n_classes))
    #         y_class_labels[:, i] = 1
    #         if y_labels is None:
    #             y_labels = y_class_labels
    #         else:
    #             y_labels = np.concatenate((y_labels, y_class_labels))
    #
    #     df_scores.reset_index(drop=True, inplace=True)
    #     # df_scores = df_scores.groupby(df_scores.index // n).mean()
    #
    #     n_samples = len(df_scores)*mult
    #     y_scores = df_scores.divide(df_scores.values.sum(axis=1), axis=0).values # Normalizing likelihoods
    #
    #     fpr = dict()
    #     tpr = dict()
    #     roc_auc = dict()
    #     for m in range(n_classes):
    #         fpr[m], tpr[m], _ = roc_curve(y_labels[:, m], y_scores[:, m])
    #         roc_auc[m] = auc(fpr[m], tpr[m])
    #
    #     plt.figure()
    #     c = 0
    #     plt.plot(fpr[c], tpr[c], color='darkorange',
    #              lw=2, label='ROC curve (area = %0.2f)' % roc_auc[c])
    #     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title(r'ROC curve $\psi_{0}$ vs. ' + f'all (n={n})')
    #     plt.legend(loc="lower right")
    #     plt.savefig(path_to_data + f'/AUROC_300samples_class{c}_llh_n{n}.png')
    #     plt.show()
