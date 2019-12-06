#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 19:53:22 2018

@author: neha
"""

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    # Jiaxin fin that if all zeros, then, the background is treated as object
    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))
    #temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))
    #print(temp1)
    intersection = temp1[0]
    #print("temp2 = ",temp1[1])
    #print(intersection.shape)
    #print(intersection)
    #Compute areas (needed for finding the union between all objects)
    #print(np.histogram(labels, bins = true_objects))
    area_true = np.histogram(labels,bins=[0,0.5,1])[0]
    #print("area_true = ",area_true)
    area_pred = np.histogram(y_pred, bins=[0,0.5,1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
  
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    intersection[intersection == 0] = 1e-9
    
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

preds_valid0 = np.load('preds_valid_rev_sgd_lh_aug_0.npy')    
preds_valid1 = np.load('preds_valid_rev_sgd_lh_aug_1.npy')    
preds_valid2 = np.load('preds_valid_rev_sgd_lh_aug_2.npy')    
preds_valid3 = np.load('preds_valid_rev_sgd_lh_aug_3.npy')     
preds_valid4 = np.load('preds_valid_rev_sgd_lh_aug_4.npy')    

y_valid0 = np.load('y_valid_rev_sgd_lh_aug_0.npy')    
y_valid1 = np.load('y_valid_rev_sgd_lh_aug_1.npy')    
y_valid2 = np.load('y_valid_rev_sgd_lh_aug_2.npy')    
y_valid3 = np.load('y_valid_rev_sgd_lh_aug_3.npy')     
y_valid4 = np.load('y_valid_rev_sgd_lh_aug_4.npy')

X = preds_valid0[:200].flatten()
y = y_valid0[:200].flatten()
clf_0 = LogisticRegression().fit(X.reshape(-1,1), np.round(y))
preds_valid0 = clf_0.predict_proba(preds_valid0.flatten().reshape(-1,1))
preds_valid0 = np.reshape(preds_valid0[:,1], (800, 128, 128))

X = preds_valid1[:200].flatten()
y = y_valid1[:200].flatten()
clf_1 = LogisticRegression().fit(X.reshape(-1,1), np.round(y))
preds_valid1 = clf_1.predict_proba(preds_valid1.flatten().reshape(-1,1))
preds_valid1 = np.reshape(preds_valid1[:,1], (800, 128, 128))

X = preds_valid2[:200].flatten()
y = y_valid2[:200].flatten()
clf_2 = LogisticRegression().fit(X.reshape(-1,1), np.round(y))
preds_valid2 = clf_2.predict_proba(preds_valid2.flatten().reshape(-1,1))
preds_valid2 = np.reshape(preds_valid2[:,1], (800, 128, 128))

X = preds_valid3[:200].flatten()
y = y_valid3[:200].flatten()
clf_3 = LogisticRegression().fit(X.reshape(-1,1), np.round(y))
preds_valid3 = clf_3.predict_proba(preds_valid3.flatten().reshape(-1,1))
preds_valid3 = np.reshape(preds_valid3[:,1], (800, 128, 128))

X = preds_valid4[:200].flatten()
y = y_valid4[:200].flatten()
clf_4 = LogisticRegression().fit(X.reshape(-1,1), np.round(y))
preds_valid4 = clf_4.predict_proba(preds_valid4.flatten().reshape(-1,1))
preds_valid4 = np.reshape(preds_valid4[:,1], (800, 128, 128))

final_prediction = np.concatenate([preds_valid0, preds_valid1, preds_valid2, preds_valid3, preds_valid4])
final_truth = np.concatenate([y_valid0, y_valid1, y_valid2, y_valid3, y_valid4])

final_prediction = np.array(final_prediction)
final_truth =np.array(final_truth)
## Scoring for last model, choose threshold by validation data 
thresholds_ori = np.linspace(0.3, 0.7, 31)
thresholds_ori = np.linspace(0.2, 0.9, 40)

# Reverse sigmoid function: Use code below because the  sigmoid activation was removed
thresholds = np.log(thresholds_ori/(1-thresholds_ori)) 

ious = np.array([iou_metric_batch(final_truth, final_prediction > threshold) for threshold in thresholds])
print(ious)

# instead of using default 0 as threshold, use validation data to find the best threshold.
threshold_best_index = np.argmax(ious) 
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend()