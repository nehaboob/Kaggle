#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 10:59:27 2018

@author: neha
"""


import pandas as pd

ins = pd.read_csv('installments_payments.csv')
pos = pd.read_csv('POS_CASH_balance.csv')
cc = pd.read_csv('credit_card_balance.csv')
prev = pd.read_csv('previous_application.csv')

prev_ins = pd.merge(prev, ins, on='SK_ID_PREV', how='inner')
prev_pos = pd.merge(prev, pos, on='SK_ID_PREV', how='inner')
prev_cc = pd.merge(prev, cc, on='SK_ID_PREV', how='inner')


prev[prev.NAME_CONTRACT_TYPE.isin(['Consumer loans', 'Cash loans'])].SK_ID_PREV.head()
2030495 - consumer loan
2802425 - cash loan
2523466
2819243
1784265

prev[prev.NAME_CONTRACT_TYPE.isin(['Revolving loans'])].SK_ID_PREV.head()

1285768
1629736
2621158
1371540
2261993

prev_ins[prev_ins.SK_ID_PREV == 2030495] - Cash loan
prev_ins[prev_ins.SK_ID_PREV == 2802425] - consumer loan
prev_ins[prev_ins.SK_ID_PREV == 1285768] - revloving loan
        
ins[ins.SK_ID_PREV == 2030495] 
ins[ins.SK_ID_PREV == 2802425]
ins[ins.SK_ID_PREV == 1285768]

pos[pos.SK_ID_PREV == 2030495] 
pos[pos.SK_ID_PREV == 2802425]


prev_pos[prev_pos.SK_ID_PREV == 2030495] - cash loan
prev_pos[prev_pos.SK_ID_PREV == 2802425] - consumer loan
prev_pos[prev_pos.SK_ID_PREV == 1285768] - revolving loan

prev_cc[prev_cc.SK_ID_PREV == 2030495] - cash loan
prev_cc[prev_cc.SK_ID_PREV == 2802425] - consumer loan
prev_cc[prev_cc.SK_ID_PREV == 1285768] - revolving loan
       
cc[cc.SK_ID_PREV == 1285768]

cc[cc.SK_ID_PREV == 1285768].
cc.loc[cc.SK_ID_PREV == 1285768, ['MONTHS_BALANCE',
'AMT_BALANCE',
'AMT_CREDIT_LIMIT_ACTUAL',
'AMT_DRAWINGS_ATM_CURRENT',
'AMT_DRAWINGS_CURRENT',
'AMT_DRAWINGS_OTHER_CURRENT',
'AMT_DRAWINGS_POS_CURRENT',
'AMT_INST_MIN_REGULARITY',
'AMT_PAYMENT_CURRENT',
'AMT_PAYMENT_TOTAL_CURRENT',
'AMT_RECEIVABLE_PRINCIPAL',
'AMT_RECIVABLE',
'AMT_TOTAL_RECEIVABLE']].sort_values('MONTHS_BALANCE')


(cc.AMT_BALANCE != cc.AMT_TOTAL_RECEIVABLE).sum()
(cc.AMT_RECIVABLE != cc.AMT_TOTAL_RECEIVABLE).sum()
(cc.AMT_PAYMENT_CURRENT != cc.AMT_PAYMENT_TOTAL_CURRENT).sum()



