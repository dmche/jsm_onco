"""
Encrypter of the hypotheses

"""

import csv
import pandas as pd


#s = 'Mel_mut_new_pfs2.csv'  #342
s = 'Mel_mut_new_pfs2_immune.csv'  #491

a = 'Mel_mut_new_ALL_immune.csv'

S = pd.read_csv(s)
A = pd.read_csv(a, index_col=0)

meanings = list(A.columns)
#print(S)
#print(meanings)

# Take all the EZK from all the hyps:

strategies = ['sim', 'simc', 'dif', 'difc']

plus = set([])

for strategy in strategies:
    for element in csv.reader(open(strategy + '/' + 'EZK_plus_reas.csv', 'r')):
        plus.add(tuple(element))


minus = set([])

for strategy in strategies:
    for element in csv.reader(open(strategy + '/' + 'EZK_minus_reas.csv', 'r')):
        minus.add(tuple(element))

#print(plus)
#print(minus)


# Take used in correct predictions:

strategies16 = [(x + '_' + y) for x in strategies for y in strategies]

plus_used = set([])
for strategy in strategies16:
    for element in csv.reader(open(strategy + '/' + 'plus_reas_2_used_hyps.csv', 'r')):
        plus_used.add(tuple(element))

minus_used = set([])
for strategy in strategies16:
    for element in csv.reader(open(strategy + '/' + 'minus_reas_2_used_hyps.csv', 'r')):
        minus_used.add(tuple(element))


# seek for common

plus_common = list(set.intersection(plus_used, plus))
minus_common = list(set.intersection(minus_used, minus))

print('Common plus')

for p in plus_common:
    p_int = [int(x) for x in p]
    print([e for e in meanings if meanings.index(e) in p_int])

print('Common minus')

for m in minus_common:
    m_int = [int(x) for x in m]
    print([e for e in meanings if meanings.index(e) in m_int])

