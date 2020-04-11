import os
import glob
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

if __name__ == "__main__":
    path_to_data = '/mnt/c/Users/gizem/Desktop/master_thesis/source/_data/roc_analysis'

    phi_set = np.load(os.path.join(path_to_data, 'phi_set_3.npy'))
    n_classes = len(phi_set)
    print(phi_set)

    # List of datasets of different classes
    list_folders = glob.glob(path_to_data + '/*')

    df_scores = pd.DataFrame()
    y_labels = None

    for i, folder in enumerate(list_folders[:3]):
        # Concatenate likelihoods from different datasets
        df_ = pd.read_csv(os.path.join(folder, 'llh.csv'), index_col=0)
        df_scores = df_scores.append(np.exp(-df_))

        # Create and concatenate labels for different classes
        n_class_samples = len(df_)
        y_class_labels = np.zeros((n_class_samples, n_classes))
        y_class_labels[:, i] = 1
        if y_labels is None:
            y_labels = y_class_labels
        else:
            y_labels = np.concatenate((y_labels, y_class_labels))

    # Adding additional dataset for class 0
    folder_name_class_0 = [
        '1586085941_3sec_100traj_3model_functionPolicy',
    ]
    for f in folder_name_class_0:
        df_class_0 = pd.read_csv(os.path.join(path_to_data, f, 'llh.csv'), index_col=0)
        df_scores = df_scores.append(np.exp(-df_class_0))
        n_class_samples = len(df_class_0)

        y_class_labels = np.zeros((n_class_samples, n_classes))
        y_class_labels[:, 0] = 1
        y_labels = np.concatenate((y_labels, y_class_labels))

    # Adding additional dataset for class 1
    folder_name_class_1 = [
        '1586089193_3sec_100traj_3model_functionPolicy'
    ]
    for f in folder_name_class_1:
        df_class_1 = pd.read_csv(os.path.join(path_to_data, f, 'llh.csv'), index_col=0)
        df_scores = df_scores.append(np.exp(-df_class_1))
        n_class_samples = len(df_class_1)

        y_class_labels = np.zeros((n_class_samples, n_classes))
        y_class_labels[:, 1] = 1
        y_labels = np.concatenate((y_labels, y_class_labels))

    # Adding additional dataset for class 2
    folder_name_class_2 = [
        '1586091283_3sec_100traj_3model_functionPolicy'
    ]
    for f in folder_name_class_2:
        df_class_2 = pd.read_csv(os.path.join(path_to_data, f, 'llh.csv'), index_col=0)
        df_scores = df_scores.append(np.exp(-df_class_2))
        n_class_samples = len(df_class_2)

        y_class_labels = np.zeros((n_class_samples, n_classes))
        y_class_labels[:, 2] = 1
        y_labels = np.concatenate((y_labels, y_class_labels))

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
    c = 2
    plt.plot(fpr[c], tpr[c], color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc[c])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(path_to_data + f'/AUROC_200samples_class{c}_negllh.png')
    plt.show()
