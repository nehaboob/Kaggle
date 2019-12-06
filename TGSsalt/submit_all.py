#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 01:29:35 2018

@author: neha
"""

import numpy as np 

preds_test0 = np.load('preds_test_rev_sgd_lh_0.npy')    
preds_test1 = np.load('preds_test_rev_sgd_lh_1.npy')    
preds_test2 = np.load('preds_test_rev_sgd_lh_2.npy')    
preds_test3 = np.load('preds_test_rev_sgd_lh_3.npy')    
preds_test4 = np.load('preds_test_rev_sgd_lh_4.npy')

sub1 = (preds_test0+preds_test1+preds_test2+preds_test3+preds_test4)/5
       
del preds_test0, preds_test1, preds_test2, preds_test3, preds_test4
       
preds_test5 = np.load('preds_test_rev_sgd_lh_aug_0.npy')    
preds_test6 = np.load('preds_test_rev_sgd_lh_aug_1.npy')    
preds_test7 = np.load('preds_test_rev_sgd_lh_aug_2.npy')    
preds_test8 = np.load('preds_test_rev_sgd_lh_aug_3.npy')    
preds_test9 = np.load('preds_test_rev_sgd_lh_aug_4.npy') 

sub2 = (preds_test5+preds_test6+preds_test7+preds_test8+preds_test9)/5
   
del  preds_test5,preds_test6,preds_test7,preds_test8,preds_test9      
       
preds_test10 = np.load('preds_test_rev_sgd_lh_aug_fixed001_3.npy')    
preds_test11 = np.load('preds_test_rev_sgd_lh_aug_lowlr_1.npy')    
preds_test12 = np.load('preds_test_rev_sgd_lh_aug_lowlr_2.npy')    
preds_test13 = np.load('preds_test_rev_sgd_lh_woflip_lh_fixed005_3.npy')    

sub3 = (preds_test10+preds_test11+preds_test12+preds_test13)/4

del preds_test10,preds_test11,preds_test12,preds_test13

sumbit_prediction = (sub1+sub2+sub3)/3