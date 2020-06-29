import numpy as np
import pandas as pd
import os
# import seaborn as sns;
#
# sns.set()
import math
import matplotlib.pyplot as plt
from utils.constants import *
from utils.helpers import *


def run_deg_plot():
    path_to_data = '/home/gizem/DATA/degenerate_pf'
    path_to_thesis = '/home/gizem/master_thesis/docs/thesis/figures/degenerate_pf'

    df_traj = pd.read_csv(path_to_data + '/df_traj.csv', index_col=0)
    df_belief_exact = pd.read_csv(path_to_data + '/belief_exactUpdate_.csv', index_col=0)
    df_belief_part = pd.read_csv(path_to_data + '/belief_particleFilter_.csv', index_col=0)

    n_axis = 5
    tick_font = 14
    label_font = 18
    legend_font = 14
    df_b_cols = ['00', '01', '10', '11']
    fig, ax = plt.subplots(n_axis, 1, sharex=True, figsize=(12, 9))
    t_max=5

    dict_b = {'exactUpdate': df_belief_exact.truncate(before=to_decimal(0), after=to_decimal(t_max)),
              'particleFilter': df_belief_part.truncate(before=to_decimal(0), after=to_decimal(t_max))}

    ax[0].step(df_traj[TIME], df_traj['y'], where='post')
    ax[0].set_ylim([-.5, 2.5])
    ax[0].set_ylabel('y(t)', fontsize=label_font)
    ax[0].axvspan(3, 3.2, color='r', alpha=0.3)
    sec = ax[0].secondary_yaxis(location='right')
    sec.set_ylabel('  (a)', fontsize=label_font, rotation='horizontal', ha='left')
    sec.set_yticks([])
    ax[0].set_ylim([-.5, 2.5])
    ax[0].set_yticks([0, 1, 2])
    ax[0].set_yticklabels([0, 1, 2], fontsize=tick_font)

    count = 1
    for col in df_b_cols:
        for m, df in dict_b.items():
            ax[count].plot(df.index, df[col]) if m == EXACT else ax[count].step(df.index, df[col], where='post')
        ax[count].set_ylabel(r'$b(x_{p}$' + f' = {col};t)', fontsize=label_font)
        ax[count].set_ylim([-0.1, 1.1])
        ax[count].set_yticks([0, 0.5, 1])
        ax[count].set_yticklabels([0, 0.5, 1], fontsize=tick_font)
        sec = ax[count].secondary_yaxis(location='right')
        if count == 0:
            sec.set_ylabel('  (a)', fontsize=label_font, rotation='horizontal', ha='left')
        elif count == 1:
            sec.set_ylabel('  (b)', fontsize=label_font, rotation='horizontal', ha='left')
        elif count == 2:
            sec.set_ylabel('  (c)', fontsize=label_font, rotation='horizontal', ha='left')
        elif count == 3:
            sec.set_ylabel('  (d)', fontsize=label_font, rotation='horizontal', ha='left')
        else:
            sec.set_ylabel('  (e)', fontsize=label_font, rotation='horizontal', ha='left')
        sec.set_yticks([])
        count += 1
    ax[1].legend(['exact update', 'particle filter'], loc='best',  fontsize=legend_font)
    ax[count - 1].set_xlabel('t / s', fontsize=label_font)
    ax[count - 1].set_xticks([0, 1, 2, 3, 4, 5])
    ax[count - 1].set_xticklabels([0, 1, 2, 3, 4, 5], fontsize=tick_font)
    plt.tight_layout()
    fig.savefig(os.path.join(path_to_thesis, 'belief_traj.pdf'))


if __name__ == "__main__":
    run_deg_plot()
