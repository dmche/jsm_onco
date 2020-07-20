# -*- coding: utf-8 -*-
"""
Full library for System interface
Melanoma Mutations edition


"""
import pandas as pd
import csv
from multiprocessing import Pool
from time import time

# expression:
# f = 'F.csv'
# s = 'S.csv'

# mutations:
f = 'M_effect.csv'
s = 'M.csv'


def comb(sign):
    '''
    full amount of hypotheses
    :return: 3 files
    '''
    t1 = time()

    erase(1)

    S = []
    # how many elements in BF? 107 - for expression
    bf = 360

    items = 0
    for element in (csv.reader(open(s, 'r'), delimiter=',')):
        if items <= bf:
            S.append([str(i) for i, e in enumerate(element) if e == '1'])
        items += 1
    del S[0]

    F = pd.read_csv(f, sep=';', header=None)[0:bf]
    # objects numbers:
    o = get_objects(sign, bf)
    contro = get_objects(-sign, bf)

    contr_elements = []
    for primer in contro:
        contr_elements.append(S[primer])

    # manually:
    k = 8

    seen = []
    combs = []
    freq = []

    def add_reas(comb_app):
        # function that checks reas and updates all the lists
        if reas not in seen:
            for contr_element in contr_elements:
                if all(x in contr_element for x in reas):
                    break
            else:
                combs.append(comb_app)
                seen.append(reas)
                freq.append([i])
        # writing no-parents
        else:
            if i > 2:
                freq[seen.index(reas)] = [int(freq[seen.index(reas)][0]) + 1]

    for i in range(2, k + 1):
        if i == 2:
            for y in o:
                for x in o:
                    if (x != y) & (sorted([x, y]) == [x, y]):
                        # getting intersection:
                        reas = [a for a in S[x] if a in S[y]]
                        add_reas([x, y])
        else:
            # take only len=i-1 list, to make intersection with them:
            combnminus1 = [a for a in combs if len(a) == i - 1]
            # and colocative reasons, to get intersection in them:
            seen_ind = [index for index in range(len(combs)) if combs[index] in combnminus1]
            seenminus1 = [seen[ind] for ind in seen_ind]

            for x in range(len(combnminus1)):
                for y in o:
                    if y in combnminus1[x]:
                        break
                    else:
                        # getting intersection:
                        reas = [a for a in seenminus1[x] if a in S[y]]
                        add_reas(combnminus1[x] + [y])

    csv.writer(open(signword(sign) + '_reas' + '.csv', 'w')).writerows(seen)
    csv.writer(open(signword(sign) + '_comb' + '.csv', 'w')).writerows(combs)
    csv.writer(open(signword(sign) + '_freq' + '.csv', 'w')).writerows(freq)

    # t2 = time()
    # print('Block 1 - calculation', t2-t1)

    # indexing

    listofhypinds = []
    for reason_par in seen:
        listofhypinds.append([seen.index(reason_par)])
        for reason in seen:
            if reason != reason_par:
                # reason contains in reason_par
                if all(x in reason_par for x in reason):
                    listofhypinds[seen.index(reason_par)].append(seen.index(reason))

    # t3 = time()
    # print('Block 2 - diags and upper', t3 - t2)

    # deleting duplicated subsets (that are including in somewhere), leaving only those NOT included nowhere
    upper_hyps = []

    # carcas diagrams:
    nested_indices = []
    for subset_par in listofhypinds:
        hypn = [int(x) for x in subset_par][0]
        for subset_ch in listofhypinds:
            if subset_ch != subset_par:
                if hypn in [int(x) for x in subset_ch]:
                    break
        else:
            nested_indices.append(listofhypinds[hypn])
            upper_hyps.append(seen[int(listofhypinds[hypn][0])])

    csv.writer(open(signword(sign) + '_diags_1' + '.csv', 'w')).writerows(nested_indices)
    csv.writer(open(signword(sign) + '_upper_hyps_2' + '.csv', 'w')).writerows(upper_hyps)

    # t4 = time()
    # print('Block 3 - indexing', t4 - t3)

    indexed_hyps = nested_indices.copy()
    for chain in indexed_hyps:
        for ind in chain:
            if chain.index(ind) == 0:
                chain[0] = []
            else:
                chain[chain.index(ind)] = sorted(
                    int(x) for x in list(set(upper_hyps[nested_indices.index(chain)]) - set(seen[int(ind)])))

    # making ordered and delete empty
    indexed_hyps_ord = []
    for chain in indexed_hyps:
        indexed_hyps_ord.append(sorted(list(filter(None, chain)), key=len, reverse=True))

    csv.writer(open(signword(sign) + '_indexed_hyps_3' + '.csv', 'w')).writerows(indexed_hyps_ord)

    # print(get_indexed_hyp(indexed_hyps, 122, 1))

    return seen


def get_indexed_hyp(indexed_hyps, upper_no, chain_no):
    # returns hypotheses by her coordinates: number of border hyp and number in sequence
    return sorted(int(x) for x in list(set(indexed_hyps[upper_no][0]) - set(indexed_hyps[upper_no][chain_no])))


def search_fragment_in_upper(sign, fragment):
    # seek fragment in upper hypotheses
    # load from file:
    upper_hyps = []
    get_fromfile(sign, upper_hyps, '_upper_hyps_2')

    # making new list for filtered hyps:
    new_hyp_list = []
    for up_h in upper_hyps:
        if all(x in up_h for x in fragment):
            new_hyp_list.append(up_h)

    print(new_hyp_list, len(new_hyp_list))


def search_len_in_upper(sign, len, compare):
    # seek hypotheses length in upper hypotheses
    upper_hyps = []
    get_fromfile(sign, upper_hyps, '_upper_hyps_2')

    new_hyp_list = []
    for up_h in upper_hyps:
        if compare == 'less':
            print(1)


def get_fromfile(sign, listt, filename):
    # getting smth from file: list must be []
    for stringg in csv.reader(open(signword(sign) + filename + '.csv', 'r')):
        listt.append([int(x) for x in stringg])


def signword(sign):
    # returns word plus or minus for filename
    if sign == 1:
        signword = 'plus'
    else:
        signword = 'minus'
    return signword


def get_objects(sign, bf):
    F = pd.read_csv(f, sep=';', header=None)[0:bf]
    # objects numbers:
    o = F[0][F[0] == sign].index.tolist()
    return o


def erase(sign):
    # clear working files:
    filenames = ['protocol', signword(sign) + '_reas', signword(sign) + '_comb', signword(sign) + '_freq',
                 signword(sign) + '_diags_1', signword(sign) + '_upper_hyps_2', signword(sign) + '_indexed_hyps_3']
    for filename in filenames:
        csv.writer(open(filename + '.csv', 'w')).writerows('')


def get_parents(a):
    # writing a file with parents
    a = 1


# CHAINS ******************************************************

def get_S(sign):
    '''
    :param sign:
    :return list of lists of attributes (obrazuyushchih), in string format:
    '''
    S = []
    # how many elements in BF? firstly we'll make without o_actual / o_old
    bf = 360
    # preparing list of all objects:
    items = 0
    for element in (csv.reader(open(s, 'r'), delimiter=',')):
        if items <= bf:
            S.append([str(i) for i, e in enumerate(element) if e == '1'])
        items += 1
    del S[0]
    # list of objects via sign
    o = get_objects(sign, bf)

    return [x for x in S if S.index(x) in o]


def get_o(sign):
    '''
    :param sign:
    :return list of lists of attributes (obrazuyushchih), in string format:
    '''
    S = []
    # how many elements in BF? firstly we'll make without o_actual / o_old
    bf = 360
    # preparing list of all objects:
    items = 0
    for element in (csv.reader(open(s, 'r'), delimiter=',')):
        if items <= bf:
            S.append([str(i) for i, e in enumerate(element) if e == '1'])
        items += 1
    del S[0]
    # list of objects via sign

    return get_objects(sign, bf)


def lower1(sign):
    '''
    :param sign:
    :return lower border list:
    '''
    lower_bord = []
    orbits = []
    S_actual = get_S(sign)
    S_contr = get_S(-sign)

    for feat in range(0, 41):
        ex_with_feat = []  # all examples that contains this feature
        for example in S_actual:
            intexam = [int(x) for x in example]
            if feat in intexam:
                ex_with_feat.append(intexam)
        # intersection
        ex_set = (set(x) for x in ex_with_feat)
        clos = set.intersection(*ex_set)

        # now we will remove those who in counter examples:
        for contr_element in S_contr:
            intcontr = [int(x) for x in contr_element]
            if all(x in intcontr for x in list(clos)):
                # print intcontr, list(clos)
                break
        else:
            lower_bord.append(list(clos))
            orbits.append(len(ex_with_feat))

    # in case when lower border in empty:
    if len(lower_bord) > 0:
        lower_bord_orbsort = [x for _, x in sorted(zip(orbits, lower_bord), reverse=True)]
        return lower_bord_orbsort
    else:
        return lower2(sign)


# def chech_if_lowest(hyp):


def lower2(sign):
    '''
    :param sign:
    :return lower border list:
    '''
    S_actual = get_S(sign)
    S_contr = get_S(-sign)

    clos = range(0, 41)
    low2 = []
    for x in clos:
        for y in clos:
            if (x != y) & (sorted([x, y]) == [x, y]):
                low2.append([x, y])
    lower_bord2 = []
    orbits = []
    for el in low2:
        ex_with_feat = []  # all examples that contains this featureS
        for example in S_actual:
            intexam = [int(x) for x in example]
            if all(x in intexam for x in el):
                ex_with_feat.append(intexam)

        # intersection
        if len(ex_with_feat) > 1:
            ex_set = (set(x) for x in ex_with_feat)
            closure = set.intersection(*ex_set)
        else:
            break
        # print(el, list(sorted(closure)))
        # now we will remove those who in counter examples:
        for contr_element in S_contr:
            intcontr = [int(x) for x in contr_element]
            if all(x in intcontr for x in list(closure)):
                break
        else:
            lower_bord2.append(list(sorted(closure)))
            orbits.append(len(ex_with_feat))

    # delete those who nested in each other
    real_lower_bord2 = lower_bord2.copy()
    delete_nested(lower_bord2, real_lower_bord2, orbits)

    lower_bord2_orbsort = [x for _, x in sorted(zip(orbits, real_lower_bord2), reverse=True)]
    return lower_bord2_orbsort


def upper(sign):
    S_actual = get_S(sign)
    S_contr = get_S(-sign)
    upp = []
    for el1 in S_actual:
        sel1 = set(el1)
        for el2 in S_actual:
            if (el1 != el2) & (sorted([el1, el2]) == [el1, el2]):
                sel2 = set(el2)
                up_el = set.intersection(sel1, sel2)
                int_up_el = sorted([int(x) for x in up_el])
                if int_up_el not in upp:
                    upp.append(int_up_el)
    upper_bord = []
    orbits = []
    for el in upp:
        ex_with_feat = []  # all examples that contains this featureS
        for example in S_actual:
            intexam = [int(x) for x in example]
            if all(x in intexam for x in el):
                ex_with_feat.append(intexam)
        if len(ex_with_feat) == 2:
            potent_hyp = el
        else:
            ex_set = (set(x) for x in ex_with_feat)
            closure = set.intersection(*ex_set)
            potent_hyp = list(closure)
        # now we will remove those who in counter examples:
        for contr_element in S_contr:
            intcontr = [int(x) for x in contr_element]
            if all(x in intcontr for x in potent_hyp):
                break
        else:
            upper_bord.append(sorted(potent_hyp))
            orbits.append(len(ex_with_feat))

    # delete those who nested in each other
    real_bord = upper_bord.copy()
    delete_nested(upper_bord, real_bord, orbits)

    upper_bord_orbsort = [x for _, x in sorted(zip(orbits, real_bord), reverse=True)]
    return upper_bord_orbsort


def delete_nested(upper_bord, real_bord, orbits):
    # check whether they are including in each other:
    for elem_real in upper_bord:
        for elem2 in real_bord:
            if elem_real != elem2:
                # reason contains in reason_par
                if all(x in elem_real for x in elem2):
                    del orbits[real_bord.index(elem2)]
                    real_bord.remove(elem2)


def orbit_closure(hyp, sign, nav_butt_active):
    '''
    Checking the status of the hypothesis, and outputs 4 cases
    Also changing the parameter for button visibility
    :param hyp: hypothesis to check its status
    :param new_hyps:
    :return: prints 4 cases
    '''
    S_actual = get_S(sign)
    S_contr = get_S(-sign)

    hyp_orbit = []
    # zkp = 0
    # all who has the hyp:
    for ex in S_actual:
        reasi = [int(x) for x in ex]
        if all(x in reasi for x in hyp):
            hyp_orbit.append(reasi)
    if len(hyp_orbit) < 2:
        # zkp = 1
        return 'Такой гипотезы не существует. Попробуйте уменьшить число совпадений с введенным Вами набором'
    else:
        ex_set = (set(x) for x in hyp_orbit)
        inters = set.intersection(*ex_set)
        # ZKP:
        for contr_element in S_contr:
            intcontr = [int(x) for x in contr_element]
            if all(x in intcontr for x in list(inters)):
                return 'Гипотеза не прошла запрет на контр-примеры. Введите другую или снизьте совпадения'
        else:
            if sorted(set(hyp)) == sorted(inters):
                nav_butt_active.append(1)
                # zkp = 1
                return 'Гипотеза. Число примеров-родителей - ' + str(len(hyp_orbit))
            else:
                # zkp = 1
                return 'Ближайшая гипотеза содержит дополнительно признаки: ' + str(
                    list(inters - set(hyp))) + ' , и имеет вид ' + str(
                    list(sorted(inters))) + ' , с числом родителей ' + str(len(hyp_orbit))
    # if zkp == 0:
    # return 'Гипотеза не прошла запрет на контр-примеры. Введите другую или снизьте совпадения'


def save_only_closures(hyp, sign, new_hyps):
    '''
    Function - is a simplier variant of orbit_closure: does not output 4 cases, and just saves closures of the hyp
    :param new_hyps:
    :return: Modifies inputted list
    '''
    S_actual = get_S(sign)
    S_contr = get_S(-sign)
    hyp_orbit = []
    # all who has the hyp:
    for ex in S_actual:
        reasi = [int(x) for x in ex]
        if all(x in reasi for x in hyp):
            hyp_orbit.append(reasi)
    if len(hyp_orbit) >= 2:
        ex_set = (set(x) for x in hyp_orbit)
        inters = set.intersection(*ex_set)
        # ZKP:
        for contr_element in S_contr:
            intcontr = [int(x) for x in contr_element]
            if all(x in intcontr for x in list(inters)):
                break
        else:
            if list(sorted(inters)) not in new_hyps:
                new_hyps.append(list(sorted(inters)))


def go_up_reasons(hypo, sign):
    '''
    Level up, for navigation block. Creates own list
    '''
    low_to_add1 = []
    for obr in range(0, 41):
        if obr in hypo:
            continue
        else:
            low_to_add1.append(obr)
    # print low_to_add1
    new_level_hyps = []
    for obr in low_to_add1:
        save_only_closures(hypo + [obr], sign, new_level_hyps)
    return new_level_hyps


def go_down_objects(hyp, sign):
    '''
        Level down, for navigation block. Writes new list
    '''
    down_level = []
    S_actual = get_S(sign)
    S_contr = get_S(-sign)
    # check whether our intersection inputs in some element
    for ex in S_actual:
        exi = [int(x) for x in ex]
        inters = set.intersection(set(exi), set(hyp))
        if (len(list(inters)) > 0) and (len(list(inters)) < len(hyp)) and (list(sorted(inters)) not in down_level):
            # ZKP:
            for contr_element in S_contr:
                intcontr = [int(x) for x in contr_element]
                if all(x in intcontr for x in list(inters)):
                    break
            else:
                down_level.append(list(sorted(inters)))

    return down_level


def get_partial(hyp, sign, n_part):
    '''
    :param n_part: number of reasons that must be equal
    :return: all the hyps associated with part
    '''
    if n_part == 0:
        return ['Задайте величину окрестности']
    else:
        hyp_orbit = []
        S_actual = get_S(sign)
        for ex in S_actual:
            exi = [int(x) for x in ex]
            inters = set.intersection(set(exi), set(hyp))
            if len(list(inters)) >= len(hyp) - n_part:
                if sorted(list(inters)) not in hyp_orbit:
                    hyp_orbit.append(list(inters))

    nearest_hyps_from_part = []
    for hyp in hyp_orbit:
        save_only_closures(hyp, sign, nearest_hyps_from_part)

    return nearest_hyps_from_part


def crypt_names():
    '''
    :return: list with ALL the reasons and their numbers
    '''
    list_pairs = []
    tab = pd.read_csv('keys.csv', sep=';')
    for i in tab.index:
        list_pairs.append([str(tab.loc[i, 'number']) + ' - ' + str(tab.loc[i, 'reas'])])
    return list_pairs


def decrypt_numbers(number):
    tab = pd.read_csv('keys.csv', sep=';')

    return tab.loc[number, 'reas']


# testing only - launch alone
if __name__ == '__main__':
    t1 = time()
    comb(1)
    t2 = time()
    print('time=', t2 - t1)


