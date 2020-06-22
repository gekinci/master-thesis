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


def main(ind):
    t0 = time.time()
    main_folder = '../_data/roc_analysis'
    config_file = '../configs/roc_analysis.yaml'

    psi_set = np.load('../configs/psi_set_3.npy')
    obs_model = psi_set[ind]

    with open(config_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    n_sample_per_class = cfg[N_TRAIN]
    run_folder = create_folder_for_experiment(folder_name=main_folder, tag=create_folder_tag(cfg))

    np.random.seed(cfg[SEED])
    pomdp = POMDPSimulation(cfg)

    # Generating all the data
    pomdp.reset_obs_model(obs_model)

    psi_folder = run_folder + f'/psi_{ind}'
    os.makedirs(psi_folder, exist_ok=True)

    if pomdp.POLICY_TYPE == DET_FUNC:
        np.save(os.path.join(psi_folder, 'policy.npy'), pomdp.policy)
    else:
        pomdp.policy.to_csv(os.path.join(psi_folder, 'policy.csv'))

    cfg['Q3'] = pomdp.Qset
    cfg['parent_Q'] = pomdp.parent_ctbn.Q

    with open(os.path.join(psi_folder, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    df_L = run(pomdp, psi_set, n_sample_per_class, psi_folder, IMPORT_DATA=IMPORT_DATA)

    visualize_llh(df_L, config[N_TRAIN], path_to_save=psi_folder)
    run_test(df_L, psi_subset, config[N_TRAIN], config[N_TEST], psi_folder)
