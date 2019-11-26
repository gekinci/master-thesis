from ctbn.generative_ctbn import GenerativeCTBN
import constants
import matplotlib.pyplot as plt


def segmentation_function(df):
    return df.max()


def get_pomdp_trajectories(ctbn):
    df_traj = ctbn.sample_and_save_trajectories(n_traj=1, file_name='parent1')
    df_traj.drop(columns=['3'], inplace=True)

    df_traj.loc[:, constants.OBS] = df_traj[['1', '2']].apply(segmentation_function, axis=1)

    return df_traj


if __name__ == "__main__":
    cfg = {
        constants.PARENTS: {'1': [],
                            '2': [],
                            '3': ['1', '2']},
        constants.T_MAX: 20,
        constants.N_VALUES: 2
    }

    parents_ctbn = GenerativeCTBN(cfg)

    df_pomdp = get_pomdp_trajectories(parents_ctbn)

    # Saving and plotting the trajectories
    fig, ax = plt.subplots(parents_ctbn.num_nodes)
    for i, var in enumerate(['1','2', constants.OBS]):
        ax[i].step(df_pomdp[constants.TIME], df_pomdp[var])
        ax[i].set_ylim([-.5, 1.5])
        ax[i].set_ylabel(var)
        ax[i].set_xlabel('time')

    fig.savefig(f'../data/pomdp.png')

