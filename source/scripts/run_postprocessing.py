import numpy as np
import pandas as pd
import os


if __name__ == "__main__":
    path_to_data = '/mnt/c/Users/gizem/Desktop/master_thesis/source/_data/inference_sampling'
    folder_name = '1585524528_3sec_10traj_5model_functionPolicy'
    path_to_exp = os.path.join(path_to_data, folder_name)

    phi_set = np.load(os.path.join(path_to_exp, 'phi_set.npy'))
    print(phi_set)
    df_llh = pd.read_csv(os.path.join(path_to_exp, 'llh.csv'), index_col=0)

    predicted_model = phi_set[int(df_llh.sum(axis=0).idxmax().split('_')[-1])]
    print('PREDICTED:\n', predicted_model)
