#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 13:05:33 2017

@author: neha
"""

import pandas as pd
import matplotlib.pyplot as plt
aisles = pd.read_csv('../data/aisles.csv')
dept = pd.read_csv('../data/departments.csv')
products = pd.read_csv('../data/products.csv')

prior = pd.read_csv('../data/order_products__prior.csv')
train = pd.read_csv('../data/order_products__train.csv')
orders = pd.read_csv('../data/orders.csv')

# check the info about the files/tabels 
print(aisles.info())
print(dept.info())
print(products.info())
print(od_pd_p.info())
print(od_pd_t.info())
print(orders.info())


## check the null columns

print(pd.isnull(aisles).sum() > 0)
print(pd.isnull(dept).sum() > 0)
print(pd.isnull(products).sum() > 0)
print(pd.isnull(od_pd_p).sum() > 0)
print(pd.isnull(od_pd_t).sum() > 0)
print(pd.isnull(orders).sum() > 0)
## prodct belongs to aisle and department
# check the mapping between Aisle and depatment
dept_aisle = products.groupby(['aisle_id', 'department_id'])

print("Total number of Dept Aisle combinations: "+str(len(dept_aisle)))
print("Total unique Aisle: "+str(products.aisle_id.nunique()))

print("Each Asile belongs to only one department. So its like dept->asile->product")

# Things to check for
# Train set orders are most recent orders of the users ?
# How many users are in test set and train set
print("order count breakup:")
print(orders.eval_set.value_counts())
print("prior orders: "+str(od_pd_p.order_id.nunique()))
print("train orders: "+str(od_pd_t.order_id.nunique()))

print("unique users in eval_sets: ")
print(orders.groupby('eval_set').user_id.nunique())
print(orders.groupby('eval_set').order_id.nunique())

print("All users are unique in train and test set")


train_users = orders[orders.eval_set == 'train'].user_id
test_users = orders[orders.eval_set == 'test'].user_id

common_users= test_users.isin(train_users)

print("test users which are in train set")
print(common_users.value_counts())

common_users= train_users.isin(test_users)

print("test users which are in train set")
print(common_users.value_counts())

print("Train and test set contain unique users which do not overlap")
print("we have 131209 train users and 75000 test users")
print("We have 206209 prior users which is sum total of train and test users")

print(orders.user_id.nunique())

## check out distribution of orders per userid

user_order_counts = orders.user_id.value_counts()

user_order_counts.plot.hist(bins=96)
user_order_counts.value_counts().plot(kind='bar')
print("Users have orders between 4 and 100")


## train and test users histogram

train_orders = orders.user_id.isin(train_users)
test_orders = orders.user_id.isin(test_users)

orders[train_orders].user_id.value_counts().plot.hist(bins=96)
orders[test_orders].user_id.value_counts().plot.hist(bins=96)

print("Train and test user id have almost same distribution on order numbers")

## Max ordered products

product_count = od_pd_p.product_id.value_counts()

product_cumsum = product_count.cumsum()

product_cumsum_percent= (product_cumsum/32434489.00)*100
product_cumsum_percent.plot(use_index=False)

# train products
product_count_t = od_pd_t.product_id.value_counts()

product_cumsum_t = product_count_t.cumsum()

product_cumsum_percent_t= (product_cumsum_t/1384617.00)*100
product_cumsum_percent_t.plot(use_index=False)

## reordered products
product_count_re = od_pd_p[od_pd_p.reordered == 1].product_id.value_counts()
product_count_t_re = od_pd_t[od_pd_t.reordered == 1].product_id.value_counts()


## check number of order wrt day of week
## check number of order wrt hour of day

## check number of orders wrt day of week and hour of day both

## check the day since orders distribution 

## distribution of number of products bought

## check the important Asiles

## check the department distribution 

## reorder percentage of department

## add to cart and reorder relation

## time based variables and reorder relation

## most rordered products

## check for the missing data in the dataframes 

## distribution of products withing departments 

## organic vs non organic

a = od_pd_p.groupby('product_id').agg({'add_to_cart_order':'mean', 'reordered':'sum'})


plt.plot(a.add_to_cart_order, a.reordered, 'ro')
