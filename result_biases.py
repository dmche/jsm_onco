"""
Table for all biases

Writes to csv


added EZK count

"""

import csv
import pandas as pd


# read ready:
s = 'Mel_mut_new_pfs2.csv'
f = 'Mel_mut_new_Effect_pfs2.csv'

strat_names = ['sim', 'simc', 'dif', 'difc']

# than we'll use full 16 strategies:
strategies = [(x + '_' + y) for x in strat_names for y in strat_names]


# Data preparation

test_batch_size = 14  # handy, once

# j - first test object's index:
start_j = 392  # open for 1-time. For full CV - make 0
end_j = start_j + test_batch_size  # - this is for single. or bf - for full, or for last objects


test_el_first = start_j


# global result:

F = pd.read_csv(f, sep=';', header=None)
F_fact = F[test_el_first:(test_el_first + test_batch_size)]

df = pd.DataFrame(index=range(test_el_first, test_el_first + test_batch_size))
df['fact'] = F_fact


for strategy in strategies:

    df_new = pd.read_csv(strategy + '/' + 'protocol_signs.csv', header=None, index_col=0)
    #print(df_new)

    df = pd.merge(df_new, df, left_index=True, right_index=True)

    # is correct?
    df[strategy] = ''
    df.loc[df[1] == df['fact'], strategy] = 1

    #del df[1]

    df[(strategy + '_sig1')] = df[2]
    df[(strategy + '_sig2')] = df[3]

    #df[(strategy + '_ezk1')] = df[4]
    #df[(strategy + '_ezk2')] = df[5]



    # delete after
    cols_to_del = [1, 2, 3] #, 4, 5]
    df.drop(cols_to_del, axis=1, inplace=True)
    print(df)


#df = df.astype(int)

print(df)
#df.to_csv('df_all_examples_result.csv')
df.to_csv('df_all_examples_result_with_values_with_EZK.csv')

