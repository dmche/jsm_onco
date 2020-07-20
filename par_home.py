'''


Homme - for test


Parallel version on final data

Adopted for AWS machine

**************

Handy parameters:
 1. input files - 2
 2. k - 9 13
 3. bf  - 414 ..
 4. allbatches - [[..]..[..]]
 5. batches_sizes - [., . , .]
 6. test_batch_size - 14
 7. start_j - 392 0

Input files preprocessing - 'Format_muts_new_JSM.py'

*************

To do:

-rename global lists
-names for overlap function
-names for all files and lists
-parallel with signs


'''

import pandas as pd
import csv
from multiprocessing import Pool, Manager, Process

from time import time
from time import sleep
import multiprocessing
import gc
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

# length of each extension:
# handy: - for ascending order!
batches_sizes = [100, 100, 100, 100]


def extensions_apply(cons, sign, strategy):  # get away sign  - or think for multiprocessing

    # define lists for writing: (maybe as function - #1)
    if sign == 1:
        if strategy == 'sim':
            global_list = global_list_plus_sim
            global_list_ezk = global_list_ezk_plus_sim
        elif strategy == 'simc':
            global_list = global_list_plus_simc
            global_list_ezk = global_list_ezk_plus_simc
        elif strategy == 'dif':
            global_list = global_list_plus_dif
            global_list_ezk = global_list_ezk_plus_dif
        elif strategy == 'difc':
            global_list = global_list_plus_difc
            global_list_ezk = global_list_ezk_plus_difc
    elif sign == -1:
        if strategy == 'sim':
            global_list = global_list_minus_sim
            global_list_ezk = global_list_ezk_minus_sim
        elif strategy == 'simc':
            global_list = global_list_minus_simc
            global_list_ezk = global_list_ezk_minus_simc
        elif strategy == 'dif':
            global_list = global_list_minus_dif
            global_list_ezk = global_list_ezk_minus_dif
        elif strategy == 'difc':
            global_list = global_list_minus_difc
            global_list_ezk = global_list_ezk_minus_difc

    # get S
    S = []
    for element in (csv.reader(open(s, 'r'), delimiter=',')):           # do not forget turn to ,
        S.append([str(i) for i, e in enumerate(element) if e == '1'])

    # delete test objects:
    del S[test_el_first:(test_el_first + test_batch_size)]

    # take empty F
    F = pd.Series(data=None, index=None).astype('int64')

    # read signs before excluding test objs:
    read_F = pd.read_csv(f, sep=';', header=None)

    # exclude tests:
    read_F.drop(read_F.index[test_el_first:(test_el_first + test_batch_size)], inplace=True)
    read_F.reset_index(inplace=True, drop=True)

    # with who we need to intersect new seen:
    seen_result_ezk = set([])

    # working files
    seen_dict = {}
    seen = set([])

    for ext in cons:  # cons:
        # in [0, 1, 2] - for one extension set (permutation):
        t_ext_start = time()

        # seen in extension - for EZK collecting and checking
        seen_ext = set([])

        bfbatch = batches_sizes[ext]

        csv.writer(open(strategy + '/' + 'protocol.csv', 'a')).writerow(['    extension index', ext, 'extension size:', bfbatch, 'is processing'])

        if ext == 0:
            batchF = read_F[0:bfbatch]
        else:
            batchF = read_F[sum(batches_sizes[:ext]):(sum(batches_sizes[:ext]) + bfbatch)]

        #  ALL F elements series - for counter-example testing
        F = pd.concat([F, batchF], axis=0)

        # current batch F elements series - for new hypotheses generation:

        # objects numbers:
        o_all = F[0][F[0] == sign].index.tolist()
        o_batch = batchF[0][batchF[0] == sign].index.tolist()

        contro = F[0][F[0] == -sign].index.tolist()

        def counter_el_generator():
            for primer in contro:
                yield S[primer]

        # Strategies - reason for NOT including this hyp-candidate into hypotheses:
        def simc(comb_add):
            # similarity with counter-examples
            strat = 'similarity with counter-examples'
            gen_contr_elems = counter_el_generator()
            for contr_element in gen_contr_elems:
                if all(x in contr_element for x in reas):
                    break
            else:
                seen_ext.add(tuple(reas))
                len_bef = len(seen)
                seen.add(tuple(reas))
                if len(seen) != len_bef:
                    seen_k[tuple(comb_add)] = reas
                    gc.collect()

        def sim(comb_add):
            # simple similarity
            strat = 'simple similarity'
            len_bef = len(seen)
            seen.add(tuple(reas))
            seen_ext.add(tuple(reas))
            if len(seen) != len_bef:
                seen_k[tuple(comb_add)] = reas
                gc.collect()

        def dif(comb_add):
            # difference
            strat = 'difference'
            # 1. get our part os S
            S_elements = []
            for primer in o_all:
                S_elements.append(S[primer])
            # 2. get parents of reas:
            par_elems = []
            for element in S_elements:
                if all(x in element for x in reas):
                    par_elems.append(element)
            # 3. excluding parents from rest elements:
            S_elems_rest = [x for x in S_elements if x not in par_elems]
            # 4. checking whether residual from each parent is in rest elements of S (than reject):
            for parent in par_elems:
                resid = [x for x in parent if x not in reas]
                if len(resid) > 0:
                    for element in S_elems_rest:
                        if all(x in element for x in resid):
                            # print('residual meets in rest, not in reasons, first break with the second immediately')
                            break
                    else:
                        # print('still ok, to next element')
                        continue
                    # print('and second break')
                    break
            else:
                seen_ext.add(tuple(reas))
                len_bef = len(seen)
                seen.add(tuple(reas))
                if len(seen) != len_bef:
                    seen_k[tuple(comb_add)] = reas

        def difc(comb_add):
            # difference with counter-examples
            strat = 'difference with counter-examples'
            gen_contr_elems = counter_el_generator()
            for contr_element in gen_contr_elems:
                if all(x in contr_element for x in reas):
                    break
            else:
                # 1. get our part os S
                S_elements = []
                for primer in o_all:
                    S_elements.append(S[primer])
                # 2. get parents of reas:
                par_elems = []
                for element in S_elements:
                    if all(x in element for x in reas):
                        par_elems.append(element)
                # 3. excluding parents from rest elements:
                S_elems_rest = [x for x in S_elements if x not in par_elems]
                # 4. checking whether residual from each parent is in rest elements of S (than reject):
                for parent in par_elems:
                    resid = [x for x in parent if x not in reas]
                    if len(resid) > 0:
                        for element in S_elems_rest:
                            if all(x in element for x in resid):
                                # print('residual meets in rest, not in reasons, first break with the second immediately')
                                break
                        else:
                            # print('still ok, to next element')
                            continue
                        # print('and second break')
                        break
                else:
                    seen_ext.add(tuple(reas))
                    len_bef = len(seen)
                    seen.add(tuple(reas))
                    if len(seen) != len_bef:
                        seen_k[tuple(comb_add)] = reas


        '''
        # strategy combinations - 16
        # similarity, similarity with counter-example prohibition, difference, difference with counter-example prohibition
        strategies = ['sim', 'scep', 'dif', 'dcep']
        # strategies combinations:
        strat_no = 0
        for strat_plus in strategies:
            for strat_minus in strategies:
                strat_no += 1
                #print(strat_no, 'strat_plus=', strat_plus, 'strat_minus=', strat_minus)

        '''
        # manually:
        k = 9

        ext_index = cons.index(ext)

        csv.writer(open(strategy + '/' + 'protocol.csv', 'a')).writerow(['    Current number (index) of ext:', ext_index, 'input number of seen:', len(seen_dict)])

        # combination block:
        for i in range(2, k + 1):
            tk_start = time()
            seen_k = {}
            if i == 2:
                for y in o_batch:
                    for x in o_all:
                        if (x != y) & (sorted([x, y]) == [x, y]):
                            # getting intersection:
                            reas = tuple(a for a in S[x] if a in S[y])
                            # function with strategy:
                            if strategy == 'dif':
                                dif([x, y])
                            elif strategy == 'difc':
                                difc([x, y])
                            elif strategy == 'sim':
                                sim([x, y])
                            else:
                                simc([x, y])
                            del reas

            else:
                # take only len=i-1 list, to make intersection with them:
                def combs_generator():
                    for a in seen_dict.keys():
                        if len(a) == i - 1:
                            yield a

                gen_combnminus1 = combs_generator()

                for comb in gen_combnminus1:
                    for y in o_batch:
                        if y in comb:
                            break
                        else:
                            # getting intersection:
                            reas = tuple(a for a in seen_dict[comb] if a in S[y])
                            # function with strategy:
                            if strategy == 'dif':
                                dif(list(comb) + [y])
                            elif strategy == 'difc':
                                difc(list(comb) + [y])
                            elif strategy == 'sim':
                                sim(list(comb) + [y])
                            else:
                                simc(list(comb) + [y])
                            del reas

            # current k is finished
            seen_dict.update(seen_k)
            tk_end = time()
            csv.writer(open(strategy + '/' + 'protocol.csv', 'a')).writerow(['      done k =', i, 'for ext', ext, 'done:', len(seen_dict), 'in time:', round(tk_end - tk_start)])
            gc.collect()

        # current extension is finished
        # Finding EZK
        t_ext_end = time()
        csv.writer(open(strategy + '/' + 'protocol.csv', 'a')).writerow(['    ext number', ext_index, 'done, current len of EZK:', len(seen_result_ezk), ', hyps for this ext to intersect:', len(seen_ext), 'in time:', round(t_ext_end-t_ext_start)])

        # check
        #csv.writer(open('CHECK_1_reas_EZK_batch_' + str(allbatches.index(cons)) + '.csv', 'w')).writerows(list(seen_result_ezk))
        #csv.writer(open('CHECK_2_reas_EZK_batch_' + str(allbatches.index(cons)) + '.csv', 'w')).writerows(list(seen_ext))

        # if it is a first iteration - in second extension will go ALL the hyps (just seen):
        if len(seen_result_ezk) == 0:
            seen_result_ezk = seen_ext.copy()
        else:
            # then check the intersection between seen and previous seen_result. It will be resulting and go further:
            seen_result_ezk = set.intersection(seen_result_ezk, seen_ext)

        csv.writer(open(strategy + '/' + 'protocol.csv', 'a')).writerow(['    and output len of EZK:', len(seen_result_ezk), ', total current seen:', len(seen)])

        del seen_ext
        gc.collect()

    # extensions' consequence (permutation set) is over, so it's time to print results:
    csv.writer(open(strategy + '/' + 'protocol.csv', 'a')).writerow(['  cross-validation set No ', allbatches.index(cons) + 1, ' from', len(allbatches), '(', cons,') finished with result len: ', len(seen)])

    # write every list of Global list inta separate file - for error traction
    csv.writer(open(strategy + '/' + signword(sign) + '_reas_EZK_batch_' + str(allbatches.index(cons)) + '.csv', 'w')).writerows(list(seen_result_ezk))
    csv.writer(open(strategy + '/' + signword(sign) + '_reas_ALL_batch_' + str(allbatches.index(cons)) + '.csv', 'w')).writerows(list(seen))

    global_list.append(seen)
    global_list_ezk.append(seen_result_ezk)
    del seen
    del seen_result_ezk
    gc.collect()

def signword(sign):
    # returns word plus or minus for filename
    if sign == 1:
        signword = 'plus'
    else:
        signword = 'minus'
    return signword


def process_sign_consequent(sign, strategy):
    """processing of all the procedure for non-parallel realization - For comparison """

    # define lists for writing: (maybe as function - #2)
    if sign == 1:
        if strategy == 'sim':
            global_list = global_list_plus_sim
            global_list_ezk = global_list_ezk_plus_sim
        elif strategy == 'simc':
            global_list = global_list_plus_simc
            global_list_ezk = global_list_ezk_plus_simc
        elif strategy == 'dif':
            global_list = global_list_plus_dif
            global_list_ezk = global_list_ezk_plus_dif
        elif strategy == 'difc':
            global_list = global_list_plus_difc
            global_list_ezk = global_list_ezk_plus_difc

    elif sign == -1:
        if strategy == 'sim':
            global_list = global_list_minus_sim
            global_list_ezk = global_list_ezk_minus_sim
        elif strategy == 'simc':
            global_list = global_list_minus_simc
            global_list_ezk = global_list_ezk_minus_simc
        elif strategy == 'dif':
            global_list = global_list_minus_dif
            global_list_ezk = global_list_ezk_minus_dif
        elif strategy == 'difc':
            global_list = global_list_minus_difc
            global_list_ezk = global_list_ezk_minus_difc


    for cons in allbatches:
        extensions_apply(cons, sign, strategy)

    # find unique hypotheses between all the reasons lists:
    inters = set.union(*global_list)
    inters_ezk = set.intersection(*global_list_ezk)

    csv.writer(open(strategy + '/' + 'EZK_' + signword(sign) + '_reas.csv', 'w')).writerows(inters_ezk)  # - for EZK
    csv.writer(open(strategy + '/' + signword(sign) + '_reas_2.csv', 'w')).writerows(list(inters))  # - all hyps for predict


    csv.writer(open(strategy + '/' + 'protocol.csv', 'a')).writerow(['** Done: ', signword(sign), 'all:', len(inters), 'ezk:', len(inters_ezk)])


def lists_analysis(sign, strategy):
    """find unique hypotheses between all the reasons lists:"""

    # define lists for writing: (maybe as function - #3)
    if sign == 1:
        if strategy == 'sim':
            global_list = global_list_plus_sim
            global_list_ezk = global_list_ezk_plus_sim
        elif strategy == 'simc':
            global_list = global_list_plus_simc
            global_list_ezk = global_list_ezk_plus_simc
        elif strategy == 'dif':
            global_list = global_list_plus_dif
            global_list_ezk = global_list_ezk_plus_dif
        elif strategy == 'difc':
            global_list = global_list_plus_difc
            global_list_ezk = global_list_ezk_plus_difc
    elif sign == -1:
        if strategy == 'sim':
            global_list = global_list_minus_sim
            global_list_ezk = global_list_ezk_minus_sim
        elif strategy == 'simc':
            global_list = global_list_minus_simc
            global_list_ezk = global_list_ezk_minus_simc
        elif strategy == 'dif':
            global_list = global_list_minus_dif
            global_list_ezk = global_list_ezk_minus_dif
        elif strategy == 'difc':
            global_list = global_list_minus_difc
            global_list_ezk = global_list_ezk_minus_difc


    inters = set.union(*global_list)
    inters_ezk = set.intersection(*global_list_ezk)

    csv.writer(open(strategy + '/' + 'EZK_' + signword(sign) + '_reas.csv', 'w')).writerows(inters_ezk)  # - for EZK
    csv.writer(open(strategy + '/' + signword(sign) + '_reas_2.csv', 'w')).writerows(list(inters))  # - all hyps for predict

    csv.writer(open(strategy + '/' + 'protocol.csv', 'a')).writerow(['** Done: ', signword(sign), 'all:', len(inters), 'ezk:', len(inters_ezk)])


# eliminate common hypotheses before testing:


def eliminate_overap(strategy):
    """As Olga told, left only non-overlapping hyps, - applies for non-counterexample strategies only"""

    if (strategy == 'dif') or (strategy == 'sim'):

        # take files with reasons:
        plus = []
        for element in csv.reader(open(strategy + '/' + 'plus_reas_2.csv', 'r')):
            plus.append(element)

        minus = []
        for element in csv.reader(open(strategy + '/' + 'minus_reas_2.csv', 'r')):
            minus.append(element)

        # checking intersection and keep only clean hypotheses:
        plus_only = [x for x in plus if x not in minus]

        minus_only = [x for x in minus if x not in plus]

        csv.writer(open(strategy + '/' + 'plus_reas_2.csv', 'w')).writerows(plus_only)
        csv.writer(open(strategy + '/' + 'minus_reas_2.csv', 'w')).writerows(minus_only)

    else:
        return


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
    elif minus_in_tau > plus_in_tau:
        minus_objects += get_objects(0, n_tau)
        csv.writer(open(strategy + '/' + 'protocol.csv', 'a')).writerow([n_tau, '-1'])
        csv.writer(open(strategy + '/' + 'protocol_signs.csv', 'a')).writerow([n_tau, '-1', minus_in_tau, plus_in_tau])
    else:
        csv.writer(open(strategy + '/' + 'protocol_signs.csv', 'a')).writerow([n_tau, '0', minus_in_tau, plus_in_tau])

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

    # similarity, similarity with counter-example, difference, difference with counter-example
    strategies = ['sim', 'simc', 'dif', 'difc']
    signs = [1, -1]
    allbatches = [[0, 1, 2, 3], [1, 0, 2, 3], [2, 3, 0, 1], [3, 2, 1, 0]]

    # clean workin protocols:
    for strategy in strategies:
        csv.writer(open(strategy + '/' + 'protocol.csv', 'w')).writerows('')
        csv.writer(open(strategy + '/' + 'protocol_signs.csv', 'w')).writerows('')

    # function arguments
    arguments = [(x, y, z) for x in allbatches for y in signs for z in strategies]

    # Data preparation

    test_batch_size = 14  # handy, once

    # j - first test object's index:
    start_j = 392  # open for 1-time. For full CV - make 0
    end_j = start_j + test_batch_size  # - this is for single. or bf - for full, or for last objects

    # restriction from past (could be deleted)
    if bf - start_j < test_batch_size:
        test_batch_size = bf - start_j

    test_el_first = start_j

    # Here starts first stage: hypotheses generation for cand-EZK
    manager = Manager()

    # lists initiation
    global_list_plus_sim = manager.list()
    global_list_minus_sim = manager.list()
    global_list_plus_simc = manager.list()
    global_list_minus_simc = manager.list()
    global_list_plus_dif = manager.list()
    global_list_minus_dif = manager.list()
    global_list_plus_difc = manager.list()
    global_list_minus_difc = manager.list()

    global_list_ezk_plus_sim = manager.list()
    global_list_ezk_minus_sim = manager.list()
    global_list_ezk_plus_simc = manager.list()
    global_list_ezk_minus_simc = manager.list()
    global_list_ezk_plus_dif = manager.list()
    global_list_ezk_minus_dif = manager.list()
    global_list_ezk_plus_difc = manager.list()
    global_list_ezk_minus_difc = manager.list()

    # main function for hypotheses generation:
    p = Pool(32)
    p.starmap(extensions_apply, arguments)
    p.close()
    p.join()

    # now we need only 8 processes: no extensions (4x), and we only process lists of each sign (8):
    arguments2 = [(x, y) for x in signs for y in strategies]

    p = Pool(8)
    p.starmap(lists_analysis, arguments2)
    p.close()
    p.join()

    # now to exclude overlapping hypotheses (only in strategies without counter-examples):
    p = Pool(2)
    async_eliminate_overap = p.map_async(eliminate_overap, ['sim', 'dif'])
    p.close()
    p.join()

    t2 = time()

    csv.writer(open('global_protocol.csv', 'a')).writerow(['All hypotheses written. time=', t2 - t1])

    '''
    # testing starts:

    for n_tau in range(test_el_first, test_el_first + test_batch_size):

        p = Pool(4)
        p.map(erase_test, strategies)
        p.close()
        p.join()

        p = Pool(4)
        p.map(csv_write_ntau, strategies)
        p.close()
        p.join()

        processes = [None] * 16
        for strategy in strategies:
            processes = multiprocessing.Process(target=multi_pir2)
            processes.start()
        processes.join()

        # new
        p = Pool(4)
        p.map(put_tau, strategies)
        p.close()
        p.join()


        processes = [None] * 16
        for strategy in strategies:
            processes = multiprocessing.Process(target=multi_check_causal)
            processes.start()
        processes.join()

        p = Pool(4)
        p.map(csv_write_end, strategies)
        p.close()
        p.join()

    t3 = time()
    csv.writer(open('global_protocol.csv', 'a')).writerow(['Total time=', t3 - t1])
    '''





