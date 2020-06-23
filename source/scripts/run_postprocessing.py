import numpy as np
import pandas as pd
import os
import seaborn as sns;

sns.set()
import matplotlib.pyplot as plt
from utils.constants import *
from scripts.run_inference_mp import visualize_llh

if __name__ == "__main__":
    path_to_data = '/home/gizem/master_thesis/source/_data/roc_analysis'
    folder_name = "1592868186_5sec_200train_0test_3model_detFunctionPolicy_['exactUpdate']_seed0_"
    path_to_exp = os.path.join(path_to_data, folder_name)

    phi_set = np.load('../configs/psi_set_81.npy')
    # print(phi_set)

    df_L = pd.read_csv(path_to_exp + '/psi_0/llh_exactUpdate.csv', index_col=0)
    dict_L = {PART_FILT: df_L}
    visualize_llh(dict_L, 200, path_to_save=path_to_exp)

    # df_llh = pd.read_csv(os.path.join(path_to_exp, 'llh.csv'), index_col=0)
    # test_data = 40
    # train_data = 160
    # df_train = df_llh.head(train_data)
    # df_test = df_llh.tail(test_data)
    # df_test_result = pd.DataFrame(columns=['Number of trajectories', 'Test log-likelihood'])
    # for i in range(1, train_data+1):
    #     df_test_run = pd.DataFrame(columns=['Number of trajectories', 'Test log-likelihood'])
    #     df_train_run = df_train.head(i)
    #     pred = phi_set[int(df_train_run.sum(axis=0).idxmax().split('_')[-1][0])]
    #     pred_tag = r'$\psi_{}$'.format(int(df_train_run.sum(axis=0).idxmax().split('_')[-1][0]))
    #     df_test_run['Test log-likelihood'] = df_test[pred_tag]
    #     df_test_run['Number of trajectories'] = i
    #     df_test_result = pd.concat([df_test_result, df_test_run])
    #     df_test_result.reset_index(drop=True, inplace=True)
    #
    # plt.figure()
    # ax = sns.lineplot(x="Number of trajectories", y="Test log-likelihood", data=df_test_result)
    # plt.savefig(os.path.join(path_to_data, folder_name)+ ('/test_likelihood.png'))
