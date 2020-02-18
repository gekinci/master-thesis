from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import pandas as pd


def visualize_optimal_policy_map(df, path_to_save='../data/'):
    step_cols = df.columns[df.columns.str.startswith('step')]

    for s in step_cols:
        fig = plt.figure()
        ax = Axes3D(fig)
        groupedbyAction = df.groupby(by=s)

        for a, df_a in list(groupedbyAction):
            ax.scatter(df_a['b1'], df_a['b2'], df_a['b3'], label=a)
            ax.set_xlabel('b1')
            ax.set_ylabel('b2')
            ax.set_zlabel('b3')
            ax.legend()
        fig.savefig(path_to_save + f'OPM_{s.split("_")[-1]}.png')
        plt.show()
    return


def visualize_trajectories(df, path_to_save='../data/'):
    node_list = ['X', 'Y', 'Z'] if 'Z' in df.columns else ['X', 'Y']

    # Saving and plotting the trajectories
    fig, ax = plt.subplots(len(node_list))
    for i, node in enumerate(node_list):
        ax[i].step(df['time'], df[node])
        ax[i].set_ylim([-.5, 1.5])
        ax[i].set_ylabel(node)
        ax[i].set_xlabel('time')

    fig.savefig(path_to_save + 'trajectory_plot.png')

    return


if __name__ == '__main__':
    folder = '../pomdp/data/1581945386/'
    df_belief = pd.read_csv(folder + 'df_belief.csv')
    df_traj = pd.read_csv(folder + 'env_traj.csv')

    visualize_optimal_policy_map(df_belief, path_to_save=folder)
