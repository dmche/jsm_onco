'''

only ttest

working (with delay)

combined to one: 2 of 2 files in total JSM solver (Testing module)

'''

import pandas as pd
import csv
from multiprocessing import Pool, Manager, Process

from time import time
from time import sleep
import multiprocessing
import gc
import os
gc.enable()



# read ready:
s = 'Mel_mut_new_pfs2.csv'
f = 'Mel_mut_new_Effect_pfs2.csv'

# immune for comparative
#s = 'Mel_mut_new_pfs2_immune.csv'
#f = 'Mel_mut_new_Effect_pfs2_immune.csv'

# start jsm: -)

# allbatches = list(all_perms(el))
# but not need to make all permuts (n!) - because matters only first digit of the batch  - need check!!!


bf = 414  #  handy - just know from database


def signword(sign):
    # returns word plus or minus for filename
    if sign == 1:
        signword = 'plus'
    else:
        signword = 'minus'
    return signword


# here testing begins:


def get_objects(sign, n_tau):
    # numbers of objects in list format
    F = pd.read_csv(f, sep=';', header=None)
    # print(F.loc[38,0])

    F.loc[n_tau, 0] = 0
    return F[0][F[0] == sign].index.tolist()


def collect_primers_reas(primers, reas):
    # getting reasons s for objects (examples)
    S = []
    for element in (csv.reader(open(s, 'r'), delimiter=',')):
        S.append([str(i) for i, e in enumerate(element) if e == '1'])
    for i in primers:
        reas.append(S[int(i)])


def pir2(sign):
    # counting enterings of reasons in tau
    # getting reasons s for tau example:
    csv.writer(open(strategy + '/' + 'protocol.csv', 'a')).writerow(["start pir2"])
    tau_reas = []
    signov_tau1 = 0

    # take who have effect 0:
    collect_primers_reas(get_objects(0, n_tau), tau_reas)

    # calculate how many of each reasons contains in tau-example reasons:
    for element in csv.reader(open(strategy + '/' + signword(sign) + '_reas_2.csv', 'r')):
        if all(x in tau_reas[0] for x in element):  # tau has reasons
            signov_tau1 += 1
            csv.writer(open(strategy + '/' + signword(sign) + '_reas_2_used_hyps.csv', 'a')).writerow(element)

    csv.writer(open(strategy + '/' + 'enterings_in_tau.csv', 'a')).writerow([signword(sign), signov_tau1])
    csv.writer(open(strategy + '/' + 'protocol.csv', 'a')).writerow(["Reasons in tau-example:", signword(sign), signov_tau1])

    sleep(1)


def put_tau(strategy):
    # compare enterings of reasons in tau-example. single-way (not parallel).
    df = pd.read_csv(strategy + '/' + 'enterings_in_tau.csv', header=None)
    plus_in_tau = df[df[0] == 'plus'].iloc[0, 1]
    minus_in_tau = df[df[0] == 'minus'].iloc[0, 1]

    plus_objects = get_objects(1, n_tau)
    minus_objects = get_objects(-1, n_tau)

    if plus_in_tau > minus_in_tau:
        plus_objects += get_objects(0, n_tau)
        csv.writer(open(strategy + '/' + 'protocol.csv', 'a')).writerow([n_tau, '1'])
        csv.writer(open(strategy + '/' + 'protocol_signs.csv', 'a')).writerow([n_tau, '1', plus_in_tau, minus_in_tau])

        csv.writer(open('all_protocol_signs.csv', 'a')).writerow([n_tau, strategy + '_' + strategy, '1', plus_in_tau, minus_in_tau])

    elif minus_in_tau > plus_in_tau:
        minus_objects += get_objects(0, n_tau)
        csv.writer(open(strategy + '/' + 'protocol.csv', 'a')).writerow([n_tau, '-1'])
        csv.writer(open(strategy + '/' + 'protocol_signs.csv', 'a')).writerow([n_tau, '-1', minus_in_tau, plus_in_tau])

        csv.writer(open('all_protocol_signs.csv', 'a')).writerow([n_tau, strategy + '_' + strategy, '-1', minus_in_tau, plus_in_tau])

    else:
        csv.writer(open(strategy + '/' + 'protocol_signs.csv', 'a')).writerow([n_tau, '0', minus_in_tau, plus_in_tau])

        csv.writer(open('all_protocol_signs.csv', 'a')).writerow([n_tau, strategy + '_' + strategy, '0', minus_in_tau, plus_in_tau])


    csv.writer(open(strategy + '/' + 'final_plus_objects.csv', 'w'), ).writerow(plus_objects)
    csv.writer(open(strategy + '/' + 'final_minus_objects.csv', 'w')).writerow(minus_objects)




def check_causal(sign):
    # checking causal explanation (abduction). Checking whether all examples were used for hypotheses generation
    count = 0
    countNo = 0
    new_primers_reas = []
    primers = list(csv.reader(open(strategy + '/' + 'final_' + signword(sign) + '_objects.csv', 'r')))[0]  # need adopt: element in csv.reader not int
    collect_primers_reas(primers, new_primers_reas)

    causal_approved = []
    for primer in new_primers_reas:
        for reason in csv.reader(open(strategy + '/' + signword(sign) + '_reas_2.csv', 'r')):
            if all(x in primer for x in reason):
                causal_approved.append(primer)
                count += 1
                break
            else:
                continue
        else:
            csv.writer(open(strategy + '/' + 'protocol.csv', 'a')).writerow([signword(sign), 'example number', primers[count], ' NOT explained'])
            count += 1
            countNo += 1
    if causal_approved == new_primers_reas:
        csv.writer(open(strategy + '/' + 'protocol.csv', 'a')).writerow(['ALL', count, signword(sign), 'examples are totally explained'])
    else:
        csv.writer(open(strategy + '/' + 'protocol.csv', 'a')).writerow(['NOT all:', count - countNo, signword(sign), 'explained', countNo,
                                                        'NOT', len(new_primers_reas), ' / ', len(causal_approved)])


def create_fill_folders_16strats(strategy1):
    # only 4 strat names
    for strategy2 in strat_names:
        plus = []
        for element in csv.reader(open(strategy1 + '/' + 'plus_reas_2.csv', 'r')):
            plus.append(element)

        minus = []
        for element in csv.reader(open(strategy2 + '/' + 'minus_reas_2.csv', 'r')):
            minus.append(element)

        folder_name = strategy1 + '_' + strategy2

        newpath = folder_name
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        plus_only = [x for x in plus if x not in minus]
        minus_only = [x for x in minus if x not in plus]

        csv.writer(open(folder_name + '/' + 'plus_reas_2.csv', 'w')).writerows(plus_only)
        csv.writer(open(folder_name + '/' + 'minus_reas_2.csv', 'w')).writerows(minus_only)


def erase_test(strategy):
    # clear working files:
    filenames = ['enterings_in_tau', 'final_plus_objects', 'final_minus_objects']
    for filename in filenames:
        csv.writer(open(strategy + '/' + filename + '.csv', 'w')).writerows('')


# For parallel implementation of testing:


def csv_write_ntau(strategy):
    csv.writer(open(strategy + '/' + 'protocol.csv', 'a')).writerow(["tau n =", n_tau])


def multi_pir2():
    pool = multiprocessing.Pool(processes=8)
    pool.map(pir2, [(sign) for sign in signs])
    pool.close()
    pool.join()


def multi_check_causal():
    pool = multiprocessing.Pool(processes=8)
    pool.map(check_causal, [(sign) for sign in signs])
    pool.close()
    pool.join()


def csv_write_end(strategy):
    csv.writer(open(strategy + '/' + 'protocol.csv', 'a')).writerow(['---end of block---'])


if __name__ == '__main__':

    t1 = time()
    csv.writer(open('global_protocol.csv', 'w')).writerow(['Started'])
    csv.writer(open('all_protocol_signs.csv', 'w')).writerows('')

    # similarity, similarity with counter-example, difference, difference with counter-example
    strat_names = ['sim', 'simc', 'dif', 'difc']

    p0 = Pool(4)
    p0.map(create_fill_folders_16strats, strat_names)
    p0.close()
    p0.join()

    # than we'll use full 16 strategies:
    strategies = [(x + '_' + y) for x in strat_names for y in strat_names]

    signs = [1, -1]

    # clean workin protocols:
    for strategy in strategies:
        csv.writer(open(strategy + '/' + 'protocol.csv', 'w')).writerows('')
        csv.writer(open(strategy + '/' + 'protocol_signs.csv', 'w')).writerows('')


    # Data preparation

    test_batch_size = 14  # handy, once

    # j - first test object's index:
    start_j = 392  # open for 1-time. For full CV - make 0
    end_j = start_j + test_batch_size  # - this is for single. or bf - for full, or for last objects

    # restriction from past (could be deleted)
    if bf - start_j < test_batch_size:
        test_batch_size = bf - start_j

    test_el_first = start_j

    # now we need only 8 processes: no extensions (4x), and we only process lists of each sign (8):
    allstrategies = [(x, y) for x in strategies for y in strategies]

    t2 = time()

    csv.writer(open('global_protocol.csv', 'a')).writerow(['All hypotheses written. time=', t2 - t1])

    # testing starts:

    for n_tau in range(test_el_first, test_el_first + test_batch_size):

        p1 = Pool(16)
        p1.map(erase_test, strategies)
        p1.close()
        p1.join()

        p2 = Pool(16)
        p2.map(csv_write_ntau, strategies)
        p2.close()
        p2.join()

        processes = [None] * 32
        for strategy in strategies:
            processes = multiprocessing.Process(target=multi_pir2)
            processes.start()
        processes.join()

        p3 = Pool(16)
        p3.map(put_tau, strategies)
        p3.close()
        p3.join()

        processes2 = [None] * 32
        for strategy in strategies:
            processes2 = multiprocessing.Process(target=multi_check_causal)
            processes2.start()
        processes2.join()

        p4 = Pool(16)
        p4.map(csv_write_end, strategies)
        p4.close()
        p4.join()

    # global result:

    F = pd.read_csv(f, sep=';', header=None)
    F_fact = F[test_el_first:(test_el_first + test_batch_size)]

    df = pd.DataFrame(index=range(test_el_first, test_el_first + test_batch_size))
    df['fact'] = F_fact

    res = pd.DataFrame(index=strategies, columns=['l0+', 'l0-', 'a+', 'a-', 'b+', 'b-', 'c+', 'c-'])

    for strategy in strategies:

        df_new = pd.read_csv(strategy + '/' + 'protocol_signs.csv', header=None, index_col=0)
        # print(df_new)

        df = pd.merge(df_new, df, left_index=True, right_index=True)

        # is correct?
        df['l0+'] = 0
        df.loc[((df[1] == df['fact']) & (df['fact'] == 1)), 'l0+'] = 1
        df['l0-'] = 0
        df.loc[((df[1] == df['fact']) & (df['fact'] == -1)), 'l0-'] = 1

        # is error?
        df['a+'] = 0
        df.loc[((df[1] != df['fact']) & (df['fact'] == 1) & (df[1] != 0)), 'a+'] = 1
        df['a-'] = 0
        df.loc[((df[1] != df['fact']) & (df['fact'] == -1) & (df[1] != 0)), 'a-'] = 1

        # is zero?
        df['b+'] = 0
        df.loc[((df[1] == 0) & (df['fact'] == 1)), 'b+'] = 1
        df['b-'] = 0
        df.loc[((df[1] == 0) & (df['fact'] == -1)), 'b-'] = 1

        # is reject?
        df['c+'] = 0
        df.loc[((df[2] == 0) & (df[2] == 0) & (df['fact'] == 1)), 'c+'] = 1
        df.loc[((df[2] == 0) & (df[2] == 0) & (df['fact'] == 1)), 'b+'] = 0

        df['c-'] = 0
        df.loc[((df[2] == 0) & (df[2] == 0) & (df['fact'] == -1)), 'c-'] = 1
        df.loc[((df[2] == 0) & (df[2] == 0) & (df['fact'] == -1)), 'b-'] = 0

        for col in res.columns:
            res.at[strategy, col] = df[col].sum()

        # delete after
        cols_to_del = [1, 2, 3]
        df.drop(cols_to_del, axis=1, inplace=True)


    #df = df.astype(int)

    #print(df)
    #df.to_csv('df_all_examples_result.csv')


    #print(res)
    res.to_csv('res_strategies_scores.csv')

    t3 = time()
    csv.writer(open('global_protocol.csv', 'a')).writerow(['Total time=', t3 - t1])






