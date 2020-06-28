import numpy as np
import pandas as pd
import os
# import seaborn as sns;
#
# sns.set()
import matplotlib.pyplot as plt
from utils.constants import *
from scripts.run_inference_mp import visualize_llh

if __name__ == "__main__":
    path_to_data = '/home/gizem/DATA'
    folder_name = "ROC_6MODEL_particleFilter_SAMECLASS_1593306382_5sec_200train_0test_6model_detFunctionPolicy_['particleFilter']_seed0_"
        # "INFER_81MODEL_exactUpdate_1593187072_5sec_200train_0test_3model_detFunctionPolicy_['exactUpdate']_seed0_"
    obs_model = 'psi_5'
    path_to_exp = os.path.join(os.path.join(path_to_data, folder_name), obs_model)
    path_to_thesis = '/home/gizem/master_thesis/docs/thesis/figures/equivalence_classes'
    print(obs_model)

    phi_set = np.load('../configs/psi_set_same_class.npy')

    df_L = pd.read_csv(path_to_exp + '/llh_particleFilter.csv', index_col=0)
    # df_L = pd.read_csv(path_to_exp + '/llh_exactUpdate.csv', index_col=0)
    dict_L = {PART_FILT: df_L}
    visualize_llh(dict_L, 200, path_to_save=path_to_thesis)

    # for s in df_L.sum(axis=0).unique():
    #     print(s)
    #     df_sum = df_L.sum(axis=0)
    #     class_list = df_L.columns[df_L.sum(axis=0) == s]
    #     # print(class_list)
    #     phi_list = []
    #     for c in class_list:
    #         ind = int(c.split('_')[-1].split('$')[0])
    #         # print(phi_set[ind])
    #         if s == -16464.560956165187:
    #             phi_list += [phi_set[ind]]
    # print(phi_list)
    # np.save('../configs/psi_set_same_class.npy', phi_list)
