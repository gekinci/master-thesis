import numpy as np
import pandas as pd
import os
import seaborn as sns
#
# sns.set()
import math
import matplotlib.pyplot as plt
from utils.constants import *
from utils.helpers import *
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score


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


def run_roc_results(c, n_classes, method):
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
        tick_font = 18
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


def run_auc_analsis(c=0, n_samples=5000, n_classes=10, n_run=10):
    path_to_thesis = '/home/gizem/master_thesis/docs/thesis/figures/roc_analysis'
    n_sample_per_class = int(n_samples / n_classes)
    df_runs = pd.DataFrame(columns=['Number of trajectories', 'AUROC', 'AUPR'])
    for method in [EXACT, PART_FILT]:
        path_to_data = '/home/gizem/DATA/ROC_' + str(n_classes) + 'MODEL_' + method + f'_{n_sample_per_class}samples'

        obs_model_list = [f'psi_{i}' for i in range(n_classes)]
        L_list = []
        for obs in obs_model_list:
            if method == PART_FILT:
                path_to_exp = os.path.join(path_to_data, obs)
                df_L = pd.read_csv(path_to_exp + '/llh_' + method + '.csv', index_col=0)
            else:
                df_L = pd.read_csv(path_to_data + f'/{obs.split("_")[-1]}' + '.csv', index_col=0)
            L_list += [df_L]

        for r in range(n_run):
            L_run = []
            df_r = pd.DataFrame(columns=['Number of trajectories', 'AUROC', 'AUPR'])
            auroc_run = []
            aupr_run = []

            start = int(r * (n_sample_per_class / n_run))
            end = int((n_sample_per_class / n_run) * (r + 1) - 1)
            for l in L_list:
                L_run += [l.loc[start:end]]

            for n in range(1, 31):
                df_scores = pd.DataFrame()
                y_labels = None

                for i, df_loglh in enumerate(L_run):
                    df_lh = np.exp(df_loglh)
                    df_lh = df_lh.divide(df_lh.values.sum(axis=1), axis=0)  # Normalizing likelihoods
                    if n != 1:
                        for k in range(int(n_sample_per_class / n_run)):
                            df_shuffled_ = df_lh.sample(frac=1).reset_index(drop=True)
                            df_scores = df_scores.append(df_shuffled_.loc[0:(n - 1)].mean(), ignore_index=True)
                    elif n == 1:
                        df_scores = df_scores.append(df_lh)

                    n_class_samples = int(len(df_loglh))
                    y_class_labels = np.zeros((n_class_samples, n_classes))
                    y_class_labels[:, i] = 1
                    if y_labels is None:
                        y_labels = y_class_labels
                    else:
                        y_labels = np.concatenate((y_labels, y_class_labels))

                df_scores.reset_index(drop=True, inplace=True)
                y_scores = df_scores.values

                pr_auc = dict()
                roc_auc = dict()
                for m in range(n_classes):
                    lr_precision, lr_recall, _ = precision_recall_curve(y_labels[:, m], y_scores[:, m])
                    pr_auc[m] = auc(lr_recall, lr_precision)
                    fpr, tpr, _ = roc_curve(y_labels[:, m], y_scores[:, m])
                    roc_auc[m] = auc(fpr, tpr)
                aupr_run += [pr_auc[c]]
                auroc_run += [roc_auc[c]]
            df_r['Number of trajectories'] = list(range(1, 31))
            df_r['AUROC'] = auroc_run
            df_r['AUPR'] = aupr_run
            df_r['state estimator'] = 'particle filter' if method == PART_FILT else 'exact update'
            df_runs = df_runs.append(df_r)
    df_runs.to_csv(path_to_thesis + f'/df_auc_{c}.csv')
    plt.figure()
    ax = sns.lineplot(x="Number of trajectories", y='AUROC', hue="state estimator", style="state estimator",
                      markers=True, dashes=False, data=df_runs)
    plt.savefig(path_to_thesis + f'/AUROC_{n_sample_per_class * n_classes}samples_class{c}_std.pdf')
    plt.figure()
    ax = sns.lineplot(x="Number of trajectories", y='AUPR', hue="state estimator", style="state estimator",
                      markers=True, dashes=False, data=df_runs)
    plt.savefig(path_to_thesis + f'/AUPR_{n_sample_per_class * n_classes}samples_class{c}_std.pdf')
    run_percentile(df_runs, c=c, path_to_save=path_to_thesis)


def run_percentile(df_run, c, path_to_save):
    fig_auroc = plt.figure()
    ax_auroc = fig_auroc.add_subplot(1, 1, 1)
    fig_aupr = plt.figure()
    ax_aupr = fig_aupr.add_subplot(1, 1, 1)
    for method in df_run['state estimator'].unique():
        df_perc_auroc = pd.DataFrame()
        df_perc_aupr = pd.DataFrame()
        for i in df_run['Number of trajectories'].unique():
            df_run_ = df_run[(df_run['Number of trajectories'] == i) & (df_run['state estimator'] == method)]
            df_perc_auroc[i] = df_run_.quantile([0.25, 0.5, 0.75])['AUROC']
            df_perc_aupr[i] = df_run_.quantile([0.25, 0.5, 0.75])['AUPR']
        method_marker = 'o' if method == 'exact update' else '*'
        ax_auroc.plot(df_perc_auroc.columns.astype(int), df_perc_auroc.loc[0.5], marker=method_marker, markersize=3,
                      label=method)
        ax_auroc.fill_between(df_perc_auroc.columns.astype(int), df_perc_auroc.loc[0.25], df_perc_auroc.loc[0.75],
                              alpha=0.2)
        ax_aupr.plot(df_perc_aupr.columns.astype(int), df_perc_aupr.loc[0.5], marker=method_marker, markersize=3,
                     label=method)
        ax_aupr.fill_between(df_perc_aupr.columns.astype(int), df_perc_aupr.loc[0.25], df_perc_aupr.loc[0.75],
                             alpha=0.2)
    # ax_auroc.set_ylim([0.5, 1.02])
    ax_auroc.set_ylabel('AUROC')
    ax_auroc.set_xlabel('Number of trajectories')
    ax_auroc.legend(loc='lower right')
    fig_auroc.savefig(path_to_save + f'/AUROC_perc_{c}.pdf')
    # ax_aupr.set_ylim([0, 1.02])
    ax_aupr.set_ylabel('AUPR')
    ax_aupr.set_xlabel('Number of trajectories')
    ax_aupr.legend(loc='lower right')
    fig_aupr.savefig(path_to_save + f'/AUPR_perc_{c}.pdf')
    plt.close('all')


if __name__ == "__main__":
    # for i in range(10):
    #     run_auc_analsis(c=i)
    # path_to_thesis = '/home/gizem/master_thesis/docs/thesis/figures/roc_analysis'
    # df_run = pd.read_csv(path_to_thesis + '/df_auc_0.csv', index_col=0)
    # run_percentile(df_run, c=0, path_to_save=path_to_thesis)
    run_auc_analsis(10)
