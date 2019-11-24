import os
from ctbn.generative_ctbn import GenerativeCTBN
from ctbn.learning_ctbn import *
import ctbn.config as cfg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    os.makedirs('../data', exist_ok=True)
    n_train = 8
    n_test = 2

    ctbn = GenerativeCTBN(cfg)

    df_train = ctbn.sample_and_save_trajectories(n_traj=n_train, file_name='train')
    df_test = ctbn.sample_and_save_trajectories(n_traj=n_test, file_name='test')

    df_eval = train_and_evaluate(ctbn, df_train, df_test)

    avg_n_transition = int(np.average([df_train.groupby(by='trajectory_id').apply(len).mean(),
                                       df_test.groupby(by='trajectory_id').apply(len).mean()]))

    plt.figure()
    # ax = sns.lineplot(x="number_of_trajectories", y="test_log_likelihood", err_style="bars", ci=68, data=df_L,
    #                   color='orange', label='learned network')
    # plt.axhline(L_true_model_avg, xmin=0, xmax=n_train + 1, color='b', label='optimal')
    ax = sns.lineplot(x="number_of_trajectories", y="mean_squared_error", err_style="bars", ci=68, data=df_eval,
                      color='orange', label='learned network')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Number of trajectories')
    plt.legend()
    plt.savefig(f'../data/training_plot_{n_train}Train_{n_test}Test_{avg_n_transition}trans.png')

    df_eval.to_csv(f'../data/df_eval_{n_train}Train_{n_test}Test_{avg_n_transition}trans.csv')

    print(f'Trajectories : {cfg.t_max} unit of time, about {avg_n_transition}')
    # print(f'True likelihood of test data: avg = {L_true_model_avg}, std = {L_true_model_std}')
    # print('Actual Q:', Q)
    # print('Predicted Q:', Q_pred)
