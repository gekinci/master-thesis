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
    parent = 2
    path_to_data = '/home/gizem/DATA/q_convergence'
    path_to_thesis = '/home/gizem/master_thesis/docs/thesis/figures/q_convergence'
    csv_path_list = sorted(glob(path_to_data + f'/Q_hist_{parent}*'))
    tick_font = 14
    label_font = 16
    legend_font = 14
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
                plt.axvline(i, ymin=0, ymax=2, color='gray', alpha=0.075)
            if parent == 1:
                black_line = 1.117 if col == '01' else 0.836
            else:
                black_line = 1.10 if col == '01' else 2.445
            plt.axhline(black_line, xmin=-100, xmax=2700, color='black', linestyle='--')
            plt.xlabel('Number of updates', fontsize=label_font)
            col_tag = 0 if col == '01' else 1
            ylabel = r'$q' + f'_{col_tag}^{parent}$'
            plt.ylabel(ylabel, fontsize=label_font)
            plt.savefig(path_to_thesis + f'/Q{parent}_{col}.pdf')
