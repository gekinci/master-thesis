import pandas as pd
import numpy as np
import os

df_1 = pd.read_csv('/mnt/c/Users/gizem/Desktop/master_thesis/source/pomdp/data/1581945386/df_belief.csv', index_col=0)
# df_2 = pd.read_csv('/mnt/c/Users/gizem/Desktop/master_thesis/source/pomdp/data/1581945575/df_belief.csv', index_col=0)
# df_2 = pd.read_csv('/mnt/c/Users/gizem/Desktop/master_thesis/source/pomdp/data/1581948052/df_belief.csv', index_col=0)
df_2 = pd.read_csv('/mnt/c/Users/gizem/Desktop/master_thesis/source/pomdp/data/1581948382/df_belief.csv', index_col=0)

step_col_1 = df_1.columns[df_1.columns.str.startswith('step')]
step_col_2 = df_2.columns[df_2.columns.str.startswith('step')]

common_step_col = list(set(step_col_1).intersection(step_col_2))

print('Comparing the policies in different step in one trajectory...')
for col1 in step_col_1:
    print(col1)
    for col2 in step_col_1:
        if col1 == col2:
            continue
        else:
            print(col2)
            if (df_1[col1] == df_1[col2]).all():
                print('policies same...')
            else:
                print('policies different...')

print('Comparing the policies from two different trajectories...')
for col in common_step_col:
    print(col)
    if (df_1[col] == df_2[col]).all():
        print('policies same...')
    else:
        print('policies different...')

