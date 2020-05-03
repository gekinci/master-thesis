import os
import glob
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scripts.run_inference_sampling import *
import math


def divisors(n):
    divs = [1]
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.extend([i, n / i])
    return np.sort(list(set(divs))).astype(int)


if __name__ == "__main__":
    path_to_data = '../_data/roc_analysis'

    phi_set = np.load(os.path.join(path_to_data, 'psi_set_3_2.npy'))
    n_classes = len(phi_set)
    print(phi_set)

    # List of datasets of different classes
    list_folders = np.sort(glob.glob(path_to_data + '/*/'))

    for n in divisors(100):
        df_scores = pd.DataFrame()
        y_labels = None

        for i, folder in enumerate(list_folders):
            # Concatenate likelihoods from different datasets
            df_loglh = pd.read_csv(os.path.join(folder, 'llh.csv'), index_col=0)
            df_lh = np.exp(df_loglh)
            # df_scores = df_scores.append(df_lh)
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
        # df_scores = df_scores.groupby(df_scores.index // n).mean()

        n_samples = len(df_scores)
        # Normalizing likelihoods
        y_scores = df_scores.divide(df_scores.values.sum(axis=1), axis=0).values

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_labels[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

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
        plt.savefig(path_to_data + f'/AUROC_100samples_class{c}_llh_n{n}.png')
        plt.show()
