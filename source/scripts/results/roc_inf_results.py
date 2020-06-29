import numpy as np
import pandas as pd
import os
# import seaborn as sns;
#
# sns.set()
import math
import matplotlib.pyplot as plt
from utils.constants import *
from utils.helpers import *
from sklearn.metrics import roc_curve, auc


def divisors(n):
    divs = [1]
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.extend([i, n / i])
    return np.sort(list(set(divs))).astype(int)


def visualize_llh(dict_L, n_train, path_to_save, tag=''):
    for m, df_L in dict_L.items():
        df_L_norm = df_L.cumsum().div((df_L.index + 1), axis=0)
        plt.figure()
        df_L_norm.head(n_train).plot()
        plt.xlabel('Number of trajectories')
        plt.ylabel('Average log-likelihood')
        plt.tight_layout()
        plt.savefig(os.path.join(path_to_save, f'llh_{m}_{tag}.pdf'))
        plt.close()


def run_ml(obs_model, method):
    path_to_data = '/home/gizem/DATA/ROC_10MODEL_' + method + '/'
    path_to_exp = os.path.join(path_to_data, obs_model)
    path_to_thesis = '/home/gizem/master_thesis/docs/thesis/figures/roc_' + method

    df_L = pd.read_csv(path_to_exp + '/llh_' + method + '.csv', index_col=0)
    dict_L = {method: df_L}
    visualize_llh(dict_L, 100, path_to_save=path_to_thesis, tag=obs_model)


def run_roc_results(c, n_classes,  method):
    n_sample_per_class = 200
    path_to_data = '/home/gizem/DATA/ROC_' + str(n_classes) + 'MODEL_' + method + '/'
    path_to_thesis = '/home/gizem/master_thesis/docs/thesis/figures/roc_' + method

    obs_model_list = [f'psi_{i}' for i in range(n_classes)]
    L_list = []
    for obs in obs_model_list:
        path_to_exp = os.path.join(path_to_data, obs)
        df_L = pd.read_csv(path_to_exp + '/llh_' + method + '.csv', index_col=0)
        L_list += [df_L]

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

        title_font = 22
        label_font = 20
        legend_font = 20
        tick_font=18
        plt.figure()
        plt.plot(fpr[c], tpr[c], color='darkorange',
                 lw=2, label='AUROC = %0.2f' % roc_auc[c])
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.xticks(fontsize=tick_font)
        plt.yticks(fontsize=tick_font)
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=label_font)
        plt.ylabel('True Positive Rate', fontsize=label_font)
        plt.title(f'n={n}', fontsize=title_font)
        plt.legend(loc="lower right", fontsize=legend_font)
        plt.tight_layout()
        plt.savefig(path_to_thesis + f'/AUROC_{n_sample_per_class * n_classes}samples_class{c}_llh_n{n}.pdf')
        plt.close('all')


if __name__ == "__main__":
    num_classes = 10
    obs_list = [f'psi_{i}' for i in range(num_classes)]
    method_list = [EXACT, PART_FILT]
    for obs_model in obs_list:
        for method in method_list:
            run_ml(obs_model, method)
            # run_roc_results(int(obs_model.split('_')[-1]), num_classes, method)
