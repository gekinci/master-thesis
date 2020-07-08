import numpy as np
import pandas as pd
import os
import seaborn as sns
from glob import glob

import math
import matplotlib.pyplot as plt
from utils.constants import *
from utils.helpers import *

if __name__ == '__main__':
    path_to_data = '/home/gizem/DATA/q_convergence'
    path_to_thesis = '/home/gizem/master_thesis/docs/thesis/figures/q_convergence'
    csv_path_list = sorted(glob(path_to_data + '/Q_hist_1*'))
    Q1 = []
    for p in csv_path_list:
        Q1 += [pd.read_csv(p, index_col=0)]
    for col in ['01', '10']:
        plt.figure()
        for p in csv_path_list:
            df = pd.read_csv(p, index_col=0)
            plt.plot(df.index, df[col])
            df[TIME].unique()
            for i in df.reset_index().groupby(by=TIME).first()['index'].values:
                plt.axvline(i, ymin=0, ymax=2, color='orange')
            plt.savefig(path_to_thesis + f'/Q_{col}.pdf')
